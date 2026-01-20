import os
import uvicorn
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from PyPDF2 import PdfReader
from io import BytesIO

# --- 0. 日志配置 (增强部署可见性) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("JL-Intelligence-Backend")

logger.info(">>> 后端引擎正在启动...")

# --- 1. 核心配置 ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        # 初始化模型实例
        model_flash = genai.GenerativeModel('gemini-1.5-flash')
        model_pro = genai.GenerativeModel('gemini-1.5-pro')
        logger.info(">>> Gemini AI 模型配置成功")
    except Exception as e:
        logger.error(f">>> 模型配置失败: {str(e)}")
        model_flash = None
        model_pro = None
else:
    model_flash = None
    model_pro = None
    logger.warning(">>> 未检测到 API Key，请检查 Render 环境变量配置")

app = FastAPI()

# --- 2. 跨域配置 (CORS) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. 首页路由 (极速响应，通过 Render 健康检查) ---
@app.get("/")
async def health_check():
    logger.info(">>> 接收到 Render 健康检查请求")
    return {
        "status": "online",
        "service": "JL Intelligence AI Engine",
        "ready": model_flash is not None,
        "founder": "Margot Jing Zhou"
    }

# --- 4. PDF 文本提取函数 ---
def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(file_content))
        text = ""
        # 针对演示，解析前 50 页已足够
        max_pages = 50 
        for i, page in enumerate(reader.pages):
            if i >= max_pages: break
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text
    except Exception as e:
        logger.error(f">>> PDF 解析失败: {str(e)}")
        return ""

# --- 5. 核心分析接口 ---
@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...), 
    analysis_type: str = Form(...)
):
    if not GOOGLE_API_KEY or not model_flash:
        raise HTTPException(status_code=500, detail="后端 API Key 未配置，分析功能暂时不可用")

    logger.info(f">>> 收到分析请求: 文件={file.filename}, 模式={analysis_type}")
    
    # 1. 读取并提取内容
    file_bytes = await file.read()
    raw_text = extract_text_from_pdf(file_bytes)
    
    if not raw_text or len(raw_text.strip()) < 10:
        logger.warning(">>> PDF 提取结果为空")
        raise HTTPException(status_code=400, detail="无法从 PDF 中提取有效文字，请确保文件非加密且非纯图片")

    # 2. 确定模型与标签
    selected_model = model_pro if analysis_type == "comprehensive" else model_flash
    model_label = "Gemini 1.5 Pro (Deep Reasoning)" if analysis_type == "comprehensive" else "Gemini 1.5 Flash (High Speed)"

    # 3. 文本截断优化
    safe_text = raw_text[:80000] 
    logger.info(f">>> 文本提取成功，准备调用 {model_label}")

    # 4. 设定专业 Prompt
    role_prompts = {
        "comprehensive": "你是一位华夏基金 (ChinaAMC) 的首席资深分析师。请对这份财报进行极其严谨的投资分析。请使用专业、优雅的 Markdown 格式输出。",
        "compliance": "你是一位资深合规风控官。请依据监管要求审查该文档的风险提示、违规用语及 ESG 披露。请列出改进建议。",
        "quick": "你是一位基金经理助理。请用 3 分钟阅读量极速提取一句话总结、三个关键数字、最大利好和最大风险。"
    }
    
    target_prompt = role_prompts.get(analysis_type, role_prompts["comprehensive"])
    final_input = f"{target_prompt}\n\n[待分析文档摘要]:\n{safe_text}"
    
    # 5. 调用 AI 引擎并处理异常
    try:
        response = selected_model.generate_content(final_input)
        
        if not response or not response.text:
            if selected_model == model_pro:
                logger.warning(">>> Pro 模型限流或异常，尝试降级到 Flash")
                response = model_flash.generate_content(final_input)
            else:
                return {"analysis": "### 分析中断\nAI 引擎暂时无法生成结果，请重试。"}
            
        logger.info(f">>> {model_label} 分析完成")
        return {"analysis": f"> **引擎状态：** 已调用 {model_label} 完成智能解析\n\n" + response.text}
        
    except Exception as e:
        logger.error(f">>> AI 调用崩溃: {str(e)}")
        # 最后的保命降级
        try:
            response = model_flash.generate_content(final_input)
            return {"analysis": f"> **系统降级提示：** 由于云端资源占用，已自动切换至快速引擎\n\n" + response.text}
        except:
            raise HTTPException(status_code=500, detail="AI 服务暂时不可用，请稍后再试")

if __name__ == "__main__":
    # 获取端口，默认为 10000 (Render 标准端口)
    port = int(os.environ.get("PORT", 10000))
    logger.info(f">>> 服务准备就绪，正在绑定端口: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
