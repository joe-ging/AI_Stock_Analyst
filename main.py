import os
import uvicorn
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai  # 🟢 使用 1.55.0+ 版本的最新引用
from PyPDF2 import PdfReader
from io import BytesIO

# --- 0. 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("JL-Backend-Interactions")

# --- 1. 核心配置 ---
# 根据最新文档，优先读取 GEMINI_API_KEY
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

app = FastAPI()

# 🟢 初始化最新的 Client
# 这一步会自动处理 API 版本路径（v1beta/v1），无需我们手动拼接
client = genai.Client(api_key=API_KEY)

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
    return {
        "status": "active", 
        "mode": "Interactions-API-Beta", 
        "sdk_version": ">=1.55.0"
    }

def extract_text_from_pdf(file_content: bytes) -> str:
    """PDF 文字提取"""
    try:
        reader = PdfReader(BytesIO(file_content))
        text = ""
        for i, page in enumerate(reader.pages[:30]): # 演示限制 30 页
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"PDF 提取失败: {e}")
        return ""

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...), analysis_type: str = Form(...)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing")

    logger.info(f">>> 正在处理文件: {file.filename}, 模式: {analysis_type}")
    
    file_bytes = await file.read()
    raw_text = extract_text_from_pdf(file_bytes)
    
    if not raw_text or len(raw_text) < 10:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    # 准备 Prompt
    prompts = {
        "comprehensive": "你是一位华夏基金资深分析师。请对这份财报进行深度投资分析。使用精美 Markdown 格式。",
        "compliance": "你是一位资深合规官。请审查文档的风险提示和合规性。",
        "quick": "你是一位基金经理助理。极速提取核心：一句话总结、三个关键数字、亮点和风险。"
    }
    
    final_prompt = f"{prompts.get(analysis_type, '分析这份财报')}\n\n内容概要:\n{raw_text[:40000]}"
    
    try:
        # 🟢 【核心修复】：使用最新的 Interactions API 语法
        # 这会自动寻找最适合的 v1beta/interactions 路径
        logger.info(">>> 正在发起交互式推理请求...")
        interaction = client.interactions.create(
            model="gemini-2.5-flash", # 文档中提到的新模型 ID
            input=final_prompt
        )
        
        # 🟢 从 Interaction 对象中提取最后一次输出的文本
        analysis_text = interaction.outputs[-1].text
        
        logger.info(">>> 分析成功")
        return {"analysis": analysis_text}

    except Exception as e:
        logger.error(f">>> AI 响应失败: {str(e)}")
        # 如果 gemini-2.5-flash 还没完全开放，尝试自动降级到常用模型
        try:
            logger.info(">>> 尝试降级到 gemini-1.5-flash...")
            interaction = client.interactions.create(
                model="gemini-1.5-flash",
                input=final_prompt
            )
            return {"analysis": interaction.outputs[-1].text}
        except:
            raise HTTPException(status_code=500, detail=f"AI 引擎拒绝请求: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
