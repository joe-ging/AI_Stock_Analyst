import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from PyPDF2 import PdfReader
from io import BytesIO

# --- 1. 核心配置 ---
# 提醒：请确保在 Render 的 Environment Variables 中添加了 GOOGLE_API_KEY
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    # 使用最稳定的 gemini-1.5-flash 模型
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    model = None
    print("⚠️ 警告: 未检测到 API Key，请检查 Render 环境变量配置")

app = FastAPI()

# --- 2. 跨域配置 (CORS) ---
# 必须允许所有源，否则你的 Netlify 前端无法访问这个 Render 后端
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. 首页路由 (解决 Render 404 健康检查问题) ---
@app.get("/")
async def health_check():
    return {
        "status": "online",
        "service": "JL Intelligence AI Engine",
        "founder": "Margot Jing Zhou",
        "company": "Jingling Education & Intelligence"
    }

# --- 4. PDF 文本提取函数 ---
def extract_text_from_pdf(file_content: bytes) -> str:
    """从 PDF 字节流中提取文本"""
    try:
        reader = PdfReader(BytesIO(file_content))
        text = ""
        # 限制解析前 50 页，确保演示时的响应速度
        max_pages = 50 
        for i, page in enumerate(reader.pages):
            if i >= max_pages: break
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text
    except Exception as e:
        print(f"PDF 解析失败: {str(e)}")
        return ""

# --- 5. 核心分析接口 ---
@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...), 
    analysis_type: str = Form(...)
):
    if not GOOGLE_API_KEY or not model:
        raise HTTPException(status_code=500, detail="后端 API Key 未配置，请联系 Margot")

    print(f"收到请求: {file.filename}, 模式: {analysis_type}")
    
    # 读取文件
    file_bytes = await file.read()
    raw_text = extract_text_from_pdf(file_bytes)
    
    if not raw_text or len(raw_text.strip()) < 10:
        raise HTTPException(status_code=400, detail="无法从 PDF 中提取有效文字，请确保不是加密文件或纯图片扫描件")

    # 🟢 优化：截断文本 (约 80,000 字符)
    # 这是为了防止财报太长导致发送给 Google 的数据量超标，造成 404 或超时
    safe_text = raw_text[:80000] 
    print(f"文本提取成功，发送至 AI 的长度: {len(safe_text)} 字符")

    # 设定针对华夏基金的专业 Prompt
    role_prompts = {
        "comprehensive": "你是一位华夏基金 (ChinaAMC) 的资深投研分析师。请对这份财报进行深度投资分析，包括核心观点、财务指标解读、业务亮点与潜在风险。请使用专业、优雅的 Markdown 格式输出。",
        "compliance": "你是一位资深合规风控官。请依据监管要求审查该文档：1.是否有充分风险提示？2.是否有“零风险/必赚”等违规用语？3.ESG 披露情况。请列出具体的风险改进建议。",
        "quick": "你是一位基金经理。请用 3 分钟阅读量极速提取：一句话总结公司现状、三个最关键财务数字、最大利好因素和最大风险点。"
    }
    
    target_prompt = role_prompts.get(analysis_type, role_prompts["comprehensive"])
    final_input = f"{target_prompt}\n\n[待分析文档内容如下]:\n{safe_text}"
    
    # 6. 调用 Google Gemini 引擎
    try:
        response = model.generate_content(final_input)
        
        # 处理可能的安全拦截
        if not response or not response.text:
            return {"analysis": "### 分析中断\nAI 检测到文档中包含某些受到策略限制的内容，无法完成自动化分析。"}
            
        return {"analysis": response.text}
        
    except Exception as e:
        err_str = str(e)
        print(f"AI 调用异常: {err_str}")
        # 如果报错包含 404，通常是接口版本问题
        if "404" in err_str:
            raise HTTPException(status_code=500, detail="AI 模型路径匹配错误，正在尝试修复中")
        raise HTTPException(status_code=500, detail=f"AI 服务暂时不可用: {err_str}")

# --- 7. 启动入口 ---
if __name__ == "__main__":
    # Render 环境必须绑定 PORT 环境变量
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
