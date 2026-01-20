import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from PyPDF2 import PdfReader
from io import BytesIO

# --- 🟢 第一步：在这里填入你的 Google API Key ---
# 去这里申请免费 Key: https://aistudio.google.com/app/apikey
# 将下面的 "你的_API_KEY_粘贴在这里" 替换成以 AIza 开头的字符串
MY_API_KEY = "AIzaSyCw374pksMMP0LPHxIIzOrNFArhGYd9mQs" 

# 自动配置环境
os.environ["GOOGLE_API_KEY"] = MY_API_KEY
genai.configure(api_key=MY_API_KEY)

# 初始化模型 (使用 Flash 模型，速度快且免费)
model = genai.GenerativeModel('gemini-1.5-flash')

app = FastAPI(title="Ant Bank AI Backend")

# --- 🟢 关键配置：允许跨域 ---
# 这让你的 HTML 网页可以访问这个 Python 程序
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisResponse(BaseModel):
    analysis: str

def extract_text_from_pdf(file_content: bytes) -> str:
    """从 PDF 提取文字"""
    try:
        reader = PdfReader(BytesIO(file_content))
        text = ""
        # 限制读取前 30 页，防止演示时等待太久
        max_pages = 300 
        for i, page in enumerate(reader.pages):
            if i >= max_pages: break
            extracted = page.extract_text()
            if extracted: text += extracted + "\n"
        return text
    except Exception as e:
        print(f"PDF 读取错误: {e}")
        return ""

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_document(
    file: UploadFile = File(...), 
    analysis_type: str = Form(...)
):
    print(f"收到文件: {file.filename}, 模式: {analysis_type}")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="请上传 PDF 文件")
    
    # 1. 读取内容
    content = await file.read()
    pdf_text = extract_text_from_pdf(content)
    
    if not pdf_text:
        raise HTTPException(status_code=400, detail="无法读取 PDF 文本，可能是扫描件。")

    print(f"成功提取文本: {len(pdf_text)} 字符")

    # 2. 设定 AI 角色
    roles = {
        "comprehensive": "你是一位华夏基金的资深分析师。请对这份财报做深度分析：核心观点（买入/卖出）、关键财务指标变化、业务亮点、估值逻辑。请用 Markdown 格式。",
        "compliance": "你是一位严格的风控官。请检查文档合规性：是否有风险提示？是否有绝对化用语（如'零风险'）？ESG 披露情况如何？请列出具体风险点。",
        "quick": "你是一位基金经理。请用 3 分钟阅读量提取核心：一句话总结、三个关键数字、最大利好 (Bull Case) 和最大风险 (Bear Case)。"
    }
    
    system_prompt = roles.get(analysis_type, roles["comprehensive"])
    final_prompt = f"{system_prompt}\n\n以下是文档内容摘要:\n{pdf_text[:30000]}" # 限制长度

    # 3. 发送给 Gemini
    try:
        print("正在请求 Google Gemini API...")
        response = model.generate_content(final_prompt)
        print("AI 分析完成！发送结果给前端。")
        return {"analysis": response.text}
    except Exception as e:
        print(f"AI 错误: {e}")
        raise HTTPException(status_code=500, detail=f"AI 处理失败: {str(e)}")

if __name__ == "__main__":
    # Render 部署需要动态获取端口
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)