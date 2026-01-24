import os
import uvicorn
import logging
import gc
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from PyPDF2 import PdfReader
from io import BytesIO

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
    return {"status": "ready", "engine": "Language-Aware-Engine", "auth": API_KEY is not None}

def extract_text_safely(file_content: bytes) -> str:
    """极限内存优化版 PDF 提取"""
    try:
        reader = PdfReader(BytesIO(file_content))
        text = ""
        # 演示模式：读取前 10 页以平衡深度与内存
        for i, page in enumerate(reader.pages[:10]):
            content = page.extract_text()
            if content: text += content + "\n"
        
        del reader
        gc.collect()
        return text.strip()
    except Exception as e:
        logger.error(f"PDF解析失败: {e}")
        return ""

@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...), 
    analysis_type: str = Form(...),
    language: str = Form(...) # 接收前端传来的语言代码 (en, zh_cn, zh_hk)
):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API Key Missing")

    logger.info(f">>> 接收请求: {file.filename} | 目标语言: {language}")
    
    # 步骤 1: 提取文本
    content = await file.read()
    raw_text = extract_text_safely(content)
    del content
    gc.collect()
    
    if not raw_text:
        raise HTTPException(status_code=400, detail="Text extraction failed")

    # 步骤 2: 语言映射
    lang_map = {
        "en": "English",
        "zh_cn": "Simplified Chinese (简体中文)",
        "zh_hk": "Traditional Chinese (繁體中文)"
    }
    target_lang = lang_map.get(language, "English")

    # 步骤 3: 准备 Prompt
    prompts = {
        "comprehensive": "You are a senior investment analyst. Perform a deep institutional research analysis on this financial report.",
        "compliance": "You are a senior compliance officer. Audit this document for risk disclosures and regulatory red flags.",
        "quick": "You are a fund manager's assistant. Provide a high-speed 3-minute executive brief."
    }
    
    # 强制语言指令
    language_instruction = f"IMPORTANT: The user has selected {target_lang} as their preferred language. You MUST generate the entire report in {target_lang}. Use professional financial terminology appropriate for that language."
    
    final_prompt = f"{language_instruction}\n\n{prompts.get(analysis_type, 'Analyze this report')}\n\n[DOCUMENT CONTENT]:\n{raw_text[:25000]}"
    
    del raw_text
    gc.collect()

    # 步骤 4: 双路径尝试
    try:
        try:
            # 路径 A: 最新 Interactions API
            interaction = client.interactions.create(
                model="gemini-2.5-flash",
                input=final_prompt
            )
            return {"analysis": interaction.outputs[-1].text}
        except Exception:
            # 路径 B: 稳定版 GenerateContent
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=final_prompt
            )
            return {"analysis": response.text}

    except Exception as e:
        logger.error(f">>> 执行失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI Engine Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
