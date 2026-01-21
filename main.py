import os
import uvicorn
import logging
import gc
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from PyPDF2 import PdfReader
from io import BytesIO
import time

# --- 0. 日志与初始化 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("JL-Final-Shield")

API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

app = FastAPI()
client = genai.Client(api_key=API_KEY)

# --- 1. 跨域配置 ---
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
    return {"status": "ready", "engine": "Dual-Engine-Hybrid", "auth": API_KEY is not None}

def extract_text_safely(file_content: bytes) -> str:
    """极限内存优化版 PDF 提取"""
    try:
        reader = PdfReader(BytesIO(file_content))
        text = ""
        # 面试演示：只读前 8 页，保证 100% 不爆内存且 AI 响应极速
        for i, page in enumerate(reader.pages[:8]):
            content = page.extract_text()
            if content: text += content + "\n"
        
        del reader
        gc.collect()
        return text.strip()
    except Exception as e:
        logger.error(f"PDF解析失败: {e}")
        return ""

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...), analysis_type: str = Form(...)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API Key Missing")

    logger.info(f">>> 接收文件: {file.filename}")
    
    # 步骤 1: 提取文本并释放字节流
    content = await file.read()
    raw_text = extract_text_safely(content)
    del content
    gc.collect()
    
    if not raw_text:
        raise HTTPException(status_code=400, detail="Text extraction failed")

    # 步骤 2: 准备 Prompt
    prompts = {
        "comprehensive": "你是一位资深投研分析师。请对这份财报进行深度投资分析。使用精美 Markdown 格式。",
        "compliance": "你是一位资深合规官。请审查文档的风险提示和合规性。",
        "quick": "你是一位助理。极速提取核心亮点和风险。"
    }
    final_prompt = f"{prompts.get(analysis_type, '分析这份财报')}\n\n[内容摘要]:\n{raw_text[:20000]}"
    
    del raw_text
    gc.collect()

    # 步骤 3: 【保命逻辑】双路径尝试
    try:
        # 第一路径：2026 Interaction API (最新)
        try:
            logger.info(">>> 尝试路径 A: Interactions (gemini-2.5-flash)...")
            interaction = client.interactions.create(
                model="gemini-2.5-flash",
                input=final_prompt
            )
            return {"analysis": interaction.outputs[-1].text}
        except Exception as e_a:
            if "404" in str(e_a) or "not found" in str(e_a).lower():
                # 第二路径：Legacy Generation (最稳)
                logger.warning(">>> 路径 A 不匹配模型，尝试路径 B: GenerateContent (gemini-1.5-flash)...")
                response = client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=final_prompt
                )
                return {"analysis": response.text}
            else:
                raise e_a

    except Exception as e:
        logger.error(f">>> 最终执行失败: {str(e)}")
        gc.collect()
        raise HTTPException(status_code=500, detail=f"AI Engine Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
