import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from PyPDF2 import PdfReader
from io import BytesIO

# --- 1. 核心配置 ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    # 初始化两个模型实例：一个追求性能(Flash)，一个追求深度(Pro)
    model_flash = genai.GenerativeModel('gemini-1.5-flash')
    model_pro = genai.GenerativeModel('gemini-1.5-pro')
else:
    model_flash = None
    model_pro = None
    print("⚠️ 警告: 未检测到 API Key，请检查 Render 环境变量配置")

app = FastAPI()

# --- 2. 跨域配置 (CORS) ---
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
        "models": ["gemini-1.5-pro", "gemini-1.5-flash"],
        "founder": "Margot Jing Zhou"
    }

# --- 4. PDF 文本提取函数 ---
def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(file_content))
        text = ""
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
    if not GOOGLE_API_KEY or not model_flash:
        raise HTTPException(status_code=500, detail="后端 API Key 未配置")

    print(f"收到请求: {file.filename}, 模式: {analysis_type}")
    
    file_bytes = await file.read()
    raw_text = extract_text_from_pdf(file_bytes)
    
    if not raw_text or len(raw_text.strip()) < 10:
        raise HTTPException(status_code=400, detail="无法从 PDF 中提取有效文字")

    # 针对不同任务选择最合适的模型
    # 深度分析使用 Pro 模型，其他使用 Flash 模型
    selected_model = model_pro if analysis_type == "comprehensive" else model_flash
    model_label = "Gemini 1.5 Pro (Deep Reasoning)" if analysis_type == "comprehensive" else "Gemini 1.5 Flash (High Speed)"

    # 优化：截断文本 (约 80,000 字符)
    safe_text = raw_text[:80000] 

    role_prompts = {
        "comprehensive": "你是一位华夏基金 (ChinaAMC) 的首席资深分析师。请对这份财报进行极其严谨的投资分析。你的受众是专业基金经理，请重点关注财务造假风险、现金流回款质量以及核心竞争力的护城河。请使用专业、优雅的 Markdown 格式输出。",
        "compliance": "你是一位资深合规风控官。请依据监管要求审查该文档：1.是否有充分风险提示？2.是否有“零风险/必赚”等违规用语？3.ESG 披露情况。请列出具体的风险改进建议。",
        "quick": "你是一位基金经理助理。请用 3 分钟阅读量极速提取：一句话总结公司现状、三个最关键财务数字、最大利好因素和最大风险点。"
    }
    
    target_prompt = role_prompts.get(analysis_type, role_prompts["comprehensive"])
    final_input = f"{target_prompt}\n\n[待分析文档内容如下]:\n{safe_text}"
    
    try:
        # 尝试使用选定的模型
        response = selected_model.generate_content(final_input)
        
        if not response or not response.text:
            # 如果 Pro 出错（通常是限流），自动降级到 Flash
            if selected_model == model_pro:
                print("Pro 模型繁忙，降级至 Flash 模式")
                response = model_flash.generate_content(final_input)
            else:
                return {"analysis": "### 分析中断\nAI 引擎响应异常，请重试。"}
            
        # 加上模型标注，增加技术含量感
        return {"analysis": f"> **引擎反馈：** 已调用 {model_label} 进行处理\n\n" + response.text}
        
    except Exception as e:
        err_str = str(e)
        print(f"AI 调用异常: {err_str}")
        # 再次尝试降级
        try:
            response = model_flash.generate_content(final_input)
            return {"analysis": f"> **警告：** 由于资源占用，已自动切换至快速引擎\n\n" + response.text}
        except:
            raise HTTPException(status_code=500, detail=f"AI 服务暂时不可用: {err_str}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
