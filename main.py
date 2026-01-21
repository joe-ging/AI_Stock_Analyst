import os
import uvicorn
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai  # 使用 2026 最新版官方 SDK
from PyPDF2 import PdfReader
from io import BytesIO

# --- 0. 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("JL-Backend-GenAI-v2")

# --- 1. 核心配置 ---
# 按照最新官方文档，优先读取 GEMINI_API_KEY
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

app = FastAPI()

# 初始化最新版 Client
# 如果环境变量叫 GEMINI_API_KEY，genai.Client() 会自动识别
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
        "sdk": "google-genai-2026", 
        "auth_configured": API_KEY is not None
    }

def extract_text_from_pdf(file_content: bytes) -> str:
    """从 PDF 中提取文字"""
    try:
        reader = PdfReader(BytesIO(file_content))
        text = ""
        # 演示建议提取前 30 页
        for i, page in enumerate(reader.pages[:30]):
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"PDF 提取失败: {str(e)}")
        return ""

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...), analysis_type: str = Form(...)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="环境变量 GEMINI_API_KEY 缺失")

    logger.info(f">>> 正在处理文件: {file.filename}, 模式: {analysis_type}")
    
    # 1. 提取文字
    file_bytes = await file.read()
    raw_text = extract_text_from_pdf(file_bytes)
    
    if not raw_text or len(raw_text) < 10:
        raise HTTPException(status_code=400, detail="无法从 PDF 中提取有效文本")

    # 2. 准备提示词
    prompts = {
        "comprehensive": "你是一位华夏基金资深投研分析师。请对这份财报进行深度投资分析。使用精美 Markdown 格式。",
        "compliance": "你是一位资深合规官。请审查文档的风险提示和合规性。指出不当措辞并给出建议。",
        "quick": "你是一位基金经理助理。极速提取核心：一句话总结、三个关键数字、亮点和风险。"
    }
    
    instruction = prompts.get(analysis_type, "请分析这份财报")
    final_content = f"{instruction}\n\n[财报正文内容]:\n{raw_text[:40000]}"
    
    try:
        # 3. 使用最新官方 SDK 调用方式
        logger.info(">>> 正在呼叫 Gemini 1.5 Flash 引擎...")
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=final_content
        )
        
        if not response.text:
            raise Exception("AI 返回内容为空")
            
        logger.info(">>> 分析成功完成")
        return {"analysis": response.text}

    except Exception as e:
        logger.error(f">>> SDK 调用失败: {str(e)}")
        # 抛出具体错误供调试
        raise HTTPException(status_code=500, detail=f"AI 引擎错误: {str(e)}")

if __name__ == "__main__":
    # Render 端口配置
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
