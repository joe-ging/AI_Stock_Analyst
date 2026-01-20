import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai  # 使用成熟稳定的 SDK
from PyPDF2 import PdfReader
from io import BytesIO

# --- 配置 ---
# 从环境变量获取 Key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    # 使用成熟的模型初始化方式
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    model = None
    print("⚠️ 警告: 未检测到 GOOGLE_API_KEY，请在 Render 环境变量中配置")

app = FastAPI()

# 允许跨域 (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 🚀 解决 Render 的 404 健康检查报错 ---
@app.get("/")
async def root():
    return {"status": "JL Intelligence AI Engine is active", "version": "1.5.2"}

def extract_text_from_pdf(file_content: bytes) -> str:
    """从 PDF 中提取文本"""
    try:
        reader = PdfReader(BytesIO(file_content))
        text = ""
        # 限制读取页数，保证演示速度
        max_pages = 50 
        for i, page in enumerate(reader.pages):
            if i >= max_pages: break
            extracted = page.extract_text()
            if extracted: text += extracted + "\n"
        return text
    except Exception as e:
        print(f"PDF 解析错误: {e}")
        return ""

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...), analysis_type: str = Form(...)):
    if not GOOGLE_API_KEY or not model:
        raise HTTPException(status_code=500, detail="Server API Key not configured")

    print(f"收到分析请求: {file.filename}, 模式: {analysis_type}")
    
    # 1. 提取内容
    content = await file.read()
    text = extract_text_from_pdf(content)
    
    if not text:
        raise HTTPException(status_code=400, detail="无法读取 PDF 内容，请确认不是纯图片扫描件。")

    print(f"文本提取成功，长度: {len(text)} 字符")

    # 2. 设定指令
    prompts = {
        "comprehensive": "你是一位华夏基金的资深投研分析师。请对这份财报进行深度分析，包含核心观点、财务指标解读、业务亮点与风险。请使用专业且排版精美的 Markdown 格式。",
        "compliance": "你是一位风控合规官。请依据监管要求审查这份文档：1.是否有风险披露？2.是否有绝对化违规用语（如“零风险”）？3.ESG披露情况。请列出具体的风险点。",
        "quick": "你是一位基金经理。请用3分钟阅读量提取核心：一句话总结、三个关键数字、最大亮点和最大风险。"
    }
    
    # 3. 调用 AI (增加重试逻辑的稳定性)
    try:
        system_instruction = prompts.get(analysis_type, prompts["comprehensive"])
        # 限制发送给 AI 的文本长度，防止超出免费版 Token 限制
        final_prompt = f"{system_instruction}\n\n以下是文档内容摘要:\n{text[:30000]}"
        
        response = model.generate_content(final_prompt)
        
        if not response.text:
            raise Exception("AI 返回内容为空")
            
        return {"analysis": response.text}
        
    except Exception as e:
        error_msg = str(e)
        print(f"AI 调用出错: {error_msg}")
        raise HTTPException(status_code=500, detail=f"AI 处理失败: {error_msg}")

if __name__ == "__main__":
    # 获取 Render 分配的端口
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
