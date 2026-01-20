import os
import uvicorn
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from PyPDF2 import PdfReader
from io import BytesIO

# --- 0. 终极日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("JL-Backend")

# --- 1. 核心配置 ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info(">>> Gemini API 基础配置已完成")
    except Exception as e:
        logger.error(f">>> API 配置崩溃: {str(e)}")
else:
    logger.warning(">>> 警告: 缺少 GOOGLE_API_KEY，请在 Render 后台 Environment 设置")

app = FastAPI()

# --- 2. 跨域配置 (CORS) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. 首页与诊断路由 ---
@app.get("/")
@app.head("/")
async def root():
    return {"status": "active", "agent": "JL Intelligence AI Engine"}

# 🟢 秘密诊断接口：如果你还是遇到 404，请在浏览器访问此地址
@app.get("/debug/models")
async def list_available_models():
    if not GOOGLE_API_KEY:
        return {"error": "API Key not configured"}
    try:
        # 获取 Google 允许你使用的所有模型列表
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        return {"available_models": models}
    except Exception as e:
        return {"error": str(e)}

# --- 4. 稳健的 PDF 提取 ---
def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(file_content))
        text = ""
        # 限制解析前 30 页，确保免费版内存不崩溃
        max_pages = 30 
        for i, page in enumerate(reader.pages):
            if i >= max_pages: break
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f">>> PDF 提取异常: {str(e)}")
        return ""

# --- 5. 核心分析接口 (带全自动容错重试) ---
@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...), 
    analysis_type: str = Form(...)
):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="API Key 缺失")

    logger.info(f">>> 收到请求: {file.filename}, 模式: {analysis_type}")
    
    # 提取文本
    file_bytes = await file.read()
    raw_text = extract_text_from_pdf(file_bytes)
    
    if not raw_text or len(raw_text) < 10:
        raise HTTPException(status_code=400, detail="无法从 PDF 提取文字，请检查文件格式")
    
    # 截断文本防止 Token 超限 (约 45,000 字符)
    safe_text = raw_text[:45000] 

    prompts = {
        "comprehensive": "你是一位华夏基金资深投研分析师。请对这份财报进行深度投资分析。使用精美的 Markdown 格式。",
        "compliance": "你是一位资深合规官。请审查文档的风险提示、违规用语及 ESG 披露。列出建议。",
        "quick": "你是一位基金经理助理。极速提取核心：一句话总结、三个关键数字、最大利好和风险。"
    }
    
    final_prompt = f"{prompts.get(analysis_type, prompts['comprehensive'])}\n\n内容概要:\n{safe_text}"
    
    # 🟡 核心改动：多模型名字轮询测试 (针对之前的 404 错误)
    # 顺序：带 latest 的稳定版 -> 标准版 -> 深度 Pro 版 -> 旧版 Pro
    candidate_models = [
        'gemini-1.5-flash-latest', 
        'gemini-1.5-flash', 
        'gemini-1.5-pro-latest',
        'gemini-pro'
    ]
    
    last_error = ""
    
    for model_name in candidate_models:
        try:
            logger.info(f">>> 尝试使用模型: {model_name} ...")
            model_instance = genai.GenerativeModel(model_name)
            response = model_instance.generate_content(final_prompt)
            
            if response and response.text:
                logger.info(f">>> 成功！使用的模型是: {model_name}")
                return {
                    "analysis": f"> **引擎诊断：** 已自动匹配最兼容引擎 ({model_name})\n\n" + response.text
                }
        except Exception as e:
            last_error = str(e)
            logger.warning(f">>> 模型 {model_name} 失败: {last_error}")
            # 继续尝试下一个模型名字
            continue

    # 如果所有候选者都试过了还是不行
    logger.error(">>> 致命错误: 所有 AI 引擎路径均不可用")
    raise HTTPException(status_code=500, detail=f"AI 引擎连接全线失败。最后一次报错: {last_error}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
