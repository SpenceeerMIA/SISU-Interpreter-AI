import streamlit as st
import os
import whisper
import time
from openai import OpenAI

# ============================================================
# 1. 基础配置
# ============================================================
LANG_NAMES = {
    "zh": "中文", "en": "English", "ja": "日本語",
    "ko": "한국어", "fr": "Français", "de": "Deutsch",
    "es": "Español", "ru": "Русский", "ar": "العربية"
}

# ============================================================
# 2. 核心算法函数
# ============================================================

def polish_transcript(client, raw_text, lang, is_source=True, src_context=""):
    """使用 DeepSeek 修正转写文本，利用大模型弥补 Small 模型的精度损失"""
    lang_name = LANG_NAMES.get(lang, lang)
    if is_source:
        instruction = f"你是专业的转录校对助手。请对以下 {lang_name} 文本进行修正：添加标点、修正同音字、智能分段并加粗小标题。"
    else:
        # 保留口译过程中的犹豫词、重复等特征，这对口译分析至关重要
        instruction = f"你是专业的转录校对助手。请对照源语上下文消歧。务必保留学生的改口、重复、停顿、犹豫词（嗯、呃、那个），只修正转写错误。"

    prompt = f"{instruction}\n\n=== 源语参考 ===\n{src_context}\n\n=== 待修正文本 ===\n{raw_text}"
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "你擅长结合上下文判断同音字，绝不改变原文措辞。"},
                      {"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except:
        return raw_text

def analyze_logic(client, src_text, interp_text, src_lang, interp_lang, academic=False):
    """结合 SISU 口译质量评估维度的深度分析"""
    src_name = LANG_NAMES.get(src_lang, src_lang)
    interp_name = LANG_NAMES.get(interp_lang, interp_lang)
    
    prompt = f"""你是一位资深口译教练。请对以下练习进行评估：
    翻译方向：{src_name} -> {interp_name}
    
    ## 源语原文
    {src_text}
    ## 学生译语
    {interp_text}
    
    请从 1.信息传达效果 2.口译策略运用 3.译语表达质量 4.流利度与稳定性 进行深度点评并打分。"""
    
    if academic:
        prompt += "\n⚠️ 请额外提供【学术分析表】，包含错误分类统计（漏译、错译、冗余）及流利度指标。"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "你是一位严谨的口译教练，点评需具体且具有学术深度。"},
                      {"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"评估分析环节调用失败: {e}"

# ============================================================
# 3. Streamlit 网页界面
# ============================================================
st.set_page_config(page_title="SISU口译教练 AI", layout="wide", page_icon="🎯")

# 侧边栏设计
with st.sidebar:
    st.title("⚙️ 助手配置")
    
    with st.expander("🤔 什么是 Whisper Small？"):
        st.write("""
        为了保证网页端运行流畅，我们选用了 **Small** 级别的 Whisper 模型。
        虽是“轻量级”，但配合 **DeepSeek AI** 的语义纠错，其实际表现非常接近专业水准。
        """)

    st.markdown("---")
    
    # API 输入
    user_api_key = st.text_input("🔑 填入 DeepSeek API Key", type="password", placeholder="sk-...", help="需自备 Key，本站不存储任何隐私数据。")
    
    # 模型选择
    model_size = st.selectbox("选择 AI 引擎精度", ["tiny", "base", "small"], index=2, help="云端环境建议最高选 small")
    
    academic_mode = st.checkbox("🎓 开启学术分析模式 (毕业论文用)")

    st.markdown("---")
    
    # 显式加载按钮
    if 'whisper_model' not in st.session_state:
        if st.button("🚀 第一步：激活 AI 引擎"):
            with st.spinner(f"正在云端初始化 {model_size} 模型..."):
                st.session_state.whisper_model = whisper.load_model(model_size)
                st.success("✅ 引擎已就绪，可以开始评估了！")

# 主界面：始终显示上传框，提升用户体验
st.title("🎯 AI 口译练习评估系统")
st.caption("上海国际外国语大学 (SISU) 英研口译实践专用 | 网页测试版")
st.write("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📤 源语材料 (Source)")
    src_file = st.file_uploader("上传原语音频/视频", type=["mp3", "m4a", "wav", "mp4"], key="src_upload")
    if src_file: st.audio(src_file)

with col2:
    st.subheader("📥 你的译语 (Interpreting)")
    interp_file = st.file_uploader("上传口译录音", type=["mp3", "m4a", "wav", "mp4"], key="interp_upload")
    if interp_file: st.audio(interp_file)

st.markdown("<br>", unsafe_allow_html=True)

# 评估执行
if st.button("🏁 第二步：开始自动评估反馈", use_container_width=True):
    if not user_api_key:
        st.error("❌ 缺少 API Key！请在左侧边栏填入后再试。")
    elif not src_file or not interp_file:
        st.error("❌ 请同时上传源语和译语文件！")
    elif 'whisper_model' not in st.session_state:
        st.error("❌ 请先在左侧边栏点击按钮激活 AI 引擎！")
    else:
        client = OpenAI(api_key=user_api_key, base_url="https://api.deepseek.com")
        
        with st.spinner("AI 正在认真听取并分析你的翻译，请保持页面开启..."):
            # 云端临时存盘
            with open("temp_src.mp3", "wb") as f: f.write(src_file.getbuffer())
            with open("temp_interp.mp3", "wb") as f: f.write(interp_file.getbuffer())

            try:
                # 1. 语音转文字 (强制 CPU 模式以保证云端稳定)
                model = st.session_state.whisper_model
                res_src = model.transcribe("temp_src.mp3", fp16=False)
                res_interp = model.transcribe("temp_interp.mp3", fp16=False)

                # 2. AI 文本修正
                st.info("✨ 正在进行同音字修正与语义对齐...")
                p_src = polish_transcript(client, res_src["text"], res_src["language"], is_source=True)
                p_interp = polish_transcript(client, res_interp["text"], res_interp["language"], is_source=False, src_context=p_src)

                # 3. 生成深度报告
                report = analyze_logic(client, p_src, p_interp, res_src["language"], res_interp["language"], academic=academic_mode)

                # 4. 结果展示
                st.success("✅ 评估报告已生成！")
                st.markdown("### 📊 反馈结果")
                st.markdown(report)
                
                st.download_button("📥 下载完整评估报告 (.md)", report, file_name=f"SISU口译反馈_{int(time.time())}.md")

            except Exception as e:
                st.error(f"运行出错: {e}")
            finally:
                # 清理临时文件
                if os.path.exists("temp_src.mp3"): os.remove("temp_src.mp3")
                if os.path.exists("temp_interp.mp3"): os.remove("temp_interp.mp3")
