import streamlit as st
import os
import whisper
import time
from openai import OpenAI
import shutil
import gc
import io
from docx import Document

# --- 【环境修复】锁定云端 FFmpeg 路径 ---
ffmpeg_bin = shutil.which("ffmpeg")
if ffmpeg_bin:
    os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_bin)

# ============================================================
# 1. 基础配置
# ============================================================
LANG_NAMES = {"zh": "中文", "en": "English", "ja": "日本語", "ko": "한국어", "fr": "Français"}

st.set_page_config(
    page_title="SISU口译教练 AI", 
    layout="wide", 
    page_icon="🏫",
    initial_sidebar_state="auto"
)

def extract_text_from_docx(file):
    """专门从 docx 中提取文字"""
    doc = Document(io.BytesIO(file.read()))
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

# ============================================================
# 2. 核心 AI 逻辑函数
# ============================================================

def polish_transcript(client, model_name, raw_text, lang, is_source=True, src_context=""):
    lang_name = LANG_NAMES.get(lang, lang)
    instruction = f"你是专业的转录校对助手。请修正 {lang_name} 文本的标点、同音字并分段。"
    if not is_source:
        instruction += "务必保留学生的改口、重复、停顿、犹豫词（嗯、呃），只修正录音转写错误。"
    prompt = f"{instruction}\n\n=== 参考上下文 ===\n{src_context}\n\n=== 待修正文本 ===\n{raw_text}"
    try:
        response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], temperature=0.2)
        return response.choices[0].message.content.strip()
    except: return raw_text

def analyze_logic(client, model_name, src_text, interp_text, academic=False):
    instruction = "你是一位资深口译教练。请对以下练习进行评估。"
    if academic:
        instruction += "目前处于【学术分析模式】，请提取 15-20 个具体的【源语 vs 译语】对比案例（跨度1-4句），并结合口译理论进行深度因果分析，最后推荐3篇真实学术文献。"
    
    prompt = f"{instruction}\n\n## 1. 源语原文\n{src_text}\n## 2. 学生译语\n{interp_text}\n\n请严格按模块输出报告。"
    try:
        response = client.chat.completions.create(
            model=model_name, 
            messages=[{"role": "system", "content": "你是一位严谨的口译教练。"}, {"role": "user", "content": prompt}], 
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e: return f"分析失败: {e}"

# ============================================================
# 3. 网页界面
# ============================================================

with st.sidebar:
    st.title("⚙️ 配置与指引")

# --- API 配置区 (仅修改此部分) ---
    user_api_key = st.text_input(
        "🔑 输入 API Key", 
        type="password", 
        placeholder="在此粘贴你的 sk-...",
        key="ds_api_key_v2",
        help="""API Key 是敏感信息，已自动加密。
        
**简单引导：**
1. 请先前往你选择的 AI 服务商官网申请 Key。
2. 即使暂时不测，也建议先领取一个备用！"""
    )
    
    api_base = st.text_input(
        "🌐 API Base URL", 
        value="",  # 已移除默认值
        placeholder="例如: https://api.deepseek.com/v1",
        help="请输入服务商提供的 API 基础地址（兼容 OpenAI 格式）。"
    )
    
    model_id = st.text_input(
        "🤖 模型名称 (Model ID)", 
        value="",  # 已移除默认值
        placeholder="例如: deepseek-chat 或 gpt-4o",
        help="请输入你想要调用的具体模型 ID。"
    )
    
    model_size = st.selectbox("🎯 Whisper 精度", ["base", "small"], index=1, help="💰有限，其他size缺货中...")
    academic_mode = st.checkbox("🎓 开启学术分析模式")
    
    st.markdown("---")
    
    if 'whisper_model' not in st.session_state:
        if st.button("🚀 第一步：激活 AI 引擎"):
            with st.spinner("正在云端初始化引擎..."):
                st.session_state.whisper_model = whisper.load_model(model_size)
                st.success("✅ 引擎就绪！")
                st.balloons()
    else:
        st.success(f"✅ 引擎 ({model_size}) 运行中")

    st.markdown("---")
    
    # --- 表情包显示 ---
    st.write("赶论文前中后期的Spencer真实写照：")
    if os.path.exists("sisu_meme.png"):
        st.image("sisu_meme.png", use_container_width=True)
    else:
        st.info("🖼️ 待上传：sisu_meme.png")

    st.markdown("### 💌 Spencer寄语")
    st.markdown("""
    正能量来了！

    **格高志远，学贯中外。**

    祝各位同学练习顺利，早日上岸！
    """)

# --- 主界面 ---
st.title("🏫 SISU 口译练习评估系统")
st.markdown("> **“格高志远 学贯中外”** —— 为上外学子定制的 AI 助学实验项目")

st.info("📱 **手机端用户：** 点击下方上传框后，可直接选择“录音机”现场录制你的翻译。")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📤 源语材料")
    src_file = st.file_uploader("上传原语音频", type=["mp3", "m4a", "wav", "mp4", "docx", "txt"], key="src_up")
    if src_file: st.audio(src_file)

with col2:
    st.subheader("📥 你的译语")
    interp_file = st.file_uploader("上传口译录音", type=["mp3", "m4a", "wav", "mp4", "docx", "txt"], key="int_up")
    if interp_file: st.audio(interp_file)

if st.button("🏁 第二步：开始自动评估反馈", use_container_width=True):
    if not user_api_key or not src_file or not interp_file:
        st.error("❌ 请检查 API Key 并上传文件！")
    elif 'whisper_model' not in st.session_state:
        st.error("❌ 请先点击激活 AI 引擎！")
    else:
        client = OpenAI(api_key=user_api_key, base_url=api_base)
        
        with st.spinner("AI 正在解析内容..."):
            try:
                # 处理源语
                if src_file.name.lower().endswith('.docx'):
                    p_src_raw = extract_text_from_docx(src_file)
                elif src_file.name.lower().endswith('.txt'):
                    p_src_raw = src_file.read().decode("utf-8")
                else:
                    with open("s.mp3", "wb") as f: f.write(src_file.getbuffer())
                    p_src_raw = st.session_state.whisper_model.transcribe("s.mp3", fp16=False)["text"]

                # 处理译语
                if interp_file.name.lower().endswith('.docx'):
                    p_interp_raw = extract_text_from_docx(interp_file)
                elif interp_file.name.lower().endswith('.txt'):
                    p_interp_raw = interp_file.read().decode("utf-8")
                else:
                    with open("i.mp3", "wb") as f: f.write(interp_file.getbuffer())
                    p_interp_raw = st.session_state.whisper_model.transcribe("i.mp3", fp16=False)["text"]

                # 文本处理与报告生成
                p_src = polish_transcript(client, model_id, p_src_raw, "auto", is_source=True)
                p_interp = polish_transcript(client, model_id, p_interp_raw, "auto", is_source=False, src_context=p_src)
                eval_report = analyze_logic(client, model_id, p_src, p_interp, academic=academic_mode)

                st.markdown(f"### 📊 评估报告\n\n{eval_report}")
                st.download_button(label="📥 下载报告 (.md)", data=eval_report, file_name=f"Report_{int(time.time())}.md")
                st.success("✅ 分析完成！")
            except Exception as e:
                st.error(f"处理出错: {e}")
            finally:
                gc.collect() 
                for f in ["s.mp3", "i.mp3"]:
                    if os.path.exists(f): os.remove(f)
