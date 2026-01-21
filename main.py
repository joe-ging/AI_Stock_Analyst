import os
import uvicorn
import logging
import gc  # 内存回收垃圾箱
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai  # 2026 最新 SDK
from PyPDF2 import PdfReader
from io import BytesIO
import time

# --- 0. 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("JL-Ultimate-Stable")

# --- 1. 核心配置 ---
# 确保在 Render 中设置了 GEMINI_API_KEY
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

app = FastAPI()

# 初始化 Client
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
    return {"status": "online", "mode": "Interactions-Memory-Hybrid"}

def extract_text_lightweight(file_content: bytes) -> str:
    """【内存优化】解析 PDF 并在提取后立即释放内存"""
    try:
        reader = PdfReader(BytesIO(file_content))
        text = ""
        # 🟢 维持 10 页限制，这是 512MB 内存的最安全红线
        for i, page in enumerate(reader.pages[:10]):
            content = page.extract_text()
            if content:
                text += content + "\n"
        
        # 强制清理对象
        del reader
        gc.collect()
        return text.strip()
    except Exception as e:
        logger.error(f"PDF 解析失败: {e}")
        return ""

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...), analysis_type: str = Form(...)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY_MISSING")

    logger.info(f">>> 正在处理文件: {file.filename}")
    
    # 读取并提取
    file_bytes = await file.read()
    raw_text = extract_text_lightweight(file_bytes)
    
    # 立刻清理原始字节流
    del file_bytes
    gc.collect()
    
    if not raw_text or len(raw_text) < 10:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    # 提示词
    prompts = {
        "comprehensive": "你是一位华夏基金资深投研分析师。请对这份财报进行深度投资分析。使用精美 Markdown 格式。",
        "compliance": "你是一位资深合规官。请审查文档的风险提示和合规性。",
        "quick": "你是一位基金经理助理。极速提取核心：一句话总结、三个关键数字、亮点和风险。"
    }
    
    # 进一步截断文本以确保 Payload 安全
    final_prompt = f"{prompts.get(analysis_type)}\n\n[内容概要]:\n{raw_text[:20000]}"
    
    # 释放文本内存
    del raw_text
    gc.collect()

    try:
        # 🟢 【恢复成功点】：使用之前测试成功的 Interactions API 语法
        # 这会自动映射到 /v1beta/interactions，绕过 404 错误
        logger.info(">>> 发起交互式 AI 推理 (Interactions API)...")
        interaction = client.interactions.create(
            model="gemini-1.5-flash", 
            input=final_prompt
        )
        
        analysis_result = interaction.outputs[-1].text
        
        # 最终内存清理
        gc.collect()
        
        logger.info(">>> 分析成功，内存已回收")
        return {"analysis": analysis_result}

    except Exception as e:
        logger.error(f">>> AI 推理异常: {str(e)}")
        gc.collect()
        raise HTTPException(status_code=500, detail=f"AI Engine Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
