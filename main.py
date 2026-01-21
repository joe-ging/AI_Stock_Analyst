import os
import uvicorn
import logging
import gc  # 🟢 引入内存管理
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from PyPDF2 import PdfReader
from io import BytesIO
import time

# --- 0. 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("JL-Memory-Guard")

# --- 1. 核心配置 ---
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

app = FastAPI()
client = genai.Client(api_key=API_KEY)
START_TIME = time.time()

# --- 2. 跨域配置 ---
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
    """轻量级健康检查"""
    return {"status": "online", "mem_optimization": "active"}

def extract_text_lightweight(file_content: bytes) -> str:
    """【核心优化】极简 PDF 提取，处理完立刻释放内存"""
    try:
        reader = PdfReader(BytesIO(file_content))
        text = ""
        # 🟢 限制为 8 页。对于 512MB 内存来说，这是最安全的红线。
        for i, page in enumerate(reader.pages[:8]): 
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        
        # 销毁对象，强制回收
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

    logger.info(f">>> 正在处理: {file.filename}")
    
    # 1. 提取文字并释放原始字节流
    content = await file.read()
    raw_text = extract_text_lightweight(content)
    del content # 立刻删除巨大的原始字节流
    gc.collect()
    
    if not raw_text:
        raise HTTPException(status_code=400, detail="CANNOT_READ_PDF")

    # 2. 提示词路由
    prompts = {
        "comprehensive": "你是一位华夏基金资深分析师。请对这份财报进行深度投资分析。使用精美 Markdown 格式。",
        "compliance": "你是一位资深合规官。请审查文档的风险提示和合规性。",
        "quick": "你是一位助理。极速提取核心：一句话总结、三个关键数字、亮点和风险。"
    }
    
    final_prompt = f"{prompts.get(analysis_type)}\n\n[内容概要]:\n{raw_text[:20000]}"
    
    # 清理掉 raw_text，因为我们已经组装好 prompt 了
    del raw_text
    gc.collect()

    try:
        logger.info(">>> 启动 AI 推理引擎...")
        # 🟢 始终使用 flash 模型，它比 pro 模型省电、省内存、速度快
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=final_prompt
        )
        
        analysis_result = response.text
        gc.collect() # 推理完再扫一遍
        
        logger.info(">>> 分析成功，已强行回收内存")
        return {"analysis": analysis_result}

    except Exception as e:
        logger.error(f">>> 引擎异常: {str(e)}")
        gc.collect()
        raise HTTPException(status_code=500, detail=f"AI Engine Busy: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
