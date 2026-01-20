import os
import uvicorn
import logging
import httpx
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from io import BytesIO

# --- 0. 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("JL-Backend-REST")

# --- 1. 核心配置 ---
# 必须在 Render 后台 Environment 变量中配置 GOOGLE_API_KEY
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

app = FastAPI()

# --- 2. 跨域配置 (CORS) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
@app.head("/")
async def root():
    # 根路由返回，用于 Render 健康检查和诊断
    return {
        "status": "active", 
        "engine": "REST-v1-Stable", 
        "region_mode": "US-West-Bypass"
    }

def extract_text_from_pdf(file_content: bytes) -> str:
    """提取 PDF 文本并截断，保护内存和 API 限制"""
    try:
        reader = PdfReader(BytesIO(file_content))
        text = ""
        max_pages = 30 # 针对投研演示，前 30 页通常足够
        for i, page in enumerate(reader.pages):
            if i >= max_pages: break
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"PDF 提取失败: {str(e)}")
        return ""

async def call_gemini_rest(prompt: str, retries=5):
    """
    【核心修复逻辑】
    不再使用 SDK，直接通过 HTTP 请求呼叫 Google 的 v1 稳定版 REST 接口。
    这能彻底解决由于 SDK 自动选择 v1beta 导致的 404 错误。
    """
    # 锁死 v1 稳定路径
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={GOOGLE_API_KEY}"
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    async with httpx.AsyncClient() as client:
        for i in range(retries):
            try:
                logger.info(f">>> 正在通过 REST v1 接口连接模型 (重试 {i+1})...")
                response = await client.post(url, json=payload, timeout=90.0)
                
                if response.status_code == 200:
                    data = response.json()
                    # 解析 Google REST API 返回的 JSON 结构
                    return data['candidates'][0]['content']['parts'][0]['text']
                
                # 处理限流或临时错误
                if response.status_code in [429, 500, 503]:
                    wait_time = (2 ** i)
                    logger.warning(f"API 繁忙 ({response.status_code})，正在进行退避重试...")
                    await asyncio.sleep(wait_time)
                    continue
                
                # 其他错误直接抛出报错详情
                error_data = response.json().get('error', {})
                error_msg = error_data.get('message', 'Unknown API Error')
                raise Exception(f"Google API 报错: {error_msg}")
                
            except Exception as e:
                if i == retries - 1:
                    raise e
                await asyncio.sleep(2 ** i)
    return None

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...), analysis_type: str = Form(...)):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="API Key 缺失，请检查环境变量设置")

    logger.info(f">>> 接收到分析请求: {file.filename}, 模式: {analysis_type}")
    
    # 1. 读取并提取文本
    content = await file.read()
    raw_text = extract_text_from_pdf(content)
    
    if not raw_text or len(raw_text) < 10:
        raise HTTPException(status_code=400, detail="无法从 PDF 中提取有效文字，请检查文件格式")

    # 2. 截断文本防止请求过载 (40,000 字符)
    safe_text = raw_text[:40000]

    # 3. 提示词匹配
    prompts = {
        "comprehensive": "你是一位华夏基金资深投研分析师。请对这份财报进行深度投资分析。使用精美 Markdown 格式。",
        "compliance": "你是一位资深合规官。请审查文档的风险提示、违规用语及 ESG 披露。列出建议。",
        "quick": "你是一位基金经理助理。极速提取核心数据：一句话总结、三个关键数字、亮点和风险。"
    }
    
    base_prompt = prompts.get(analysis_type, prompts["comprehensive"])
    final_input = f"{base_prompt}\n\n[内容如下]:\n{safe_text}"
    
    # 4. 执行 REST 调用
    try:
        analysis_result = await call_gemini_rest(final_input)
        if not analysis_result:
            raise HTTPException(status_code=500, detail="AI 返回结果为空")
            
        # 返回格式兼容你目前的前端
        return {"analysis": analysis_result}
    except Exception as e:
        logger.error(f">>> 引擎运行故障: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI 分析失败: {str(e)}")

if __name__ == "__main__":
    # 获取 Render 端口
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
