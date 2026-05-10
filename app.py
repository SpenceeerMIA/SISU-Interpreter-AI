import streamlit as st
import os
import whisper
import time
from openai import OpenAI
import shutil

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
    """使用 DeepSeek 修正转写文本"""
    lang_name = LANG_NAMES.get(lang, lang)
    if is_source:
        instruction = f"你是专业的转录校对助手。请对以下 {lang_name} 文本进行修正：添加标点、修正同音字、智能分段并加粗小标题。"
    else:
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
    """多维口译评估 - 优化输出格式"""
    src_name = LANG_NAMES.get(src_lang, src_lang)
    interp_name = LANG_NAMES.get(interp_lang, interp_lang)
    
    prompt = f"""你是一位资深口译教练。请对以下练习进行评估：
    翻译方向：{src_name} -> {interp_name}
    
    ## 源语原文
    {src_text}
    ## 学生译语
    {interp_text}
    
    任务要求：
    1. 请先列出一个【评分摘要表】，包含：信息传达、策略运用、语言质量、流利度（每项满分25分）。
    2. 接着进行【详细点评】，分点列出优缺点。
    3. 最后给出【改进建议】。
    注意：使用 Markdown 表格和加粗标题，确保排版在 Obsidian 等工具中清晰美观。"""
    
    if academic:
        prompt += "\n4. 额外增加【学术分析】，统计错译、漏译及犹豫词频率。"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "你是一位严谨的口译教练，点评需具体且排版专业。"},
                      {"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"评估调用失败: {e}"

# ============================================================
# 3. Streamlit 网页界面
# ============================================================
st.set_page_config(page_title="SISU口译教练", layout="wide", page_icon="🎯")

# 侧边栏
with st.sidebar:
    st.title("⚙️ 助手配置")
    user_api_key = st.text_input("🔑 DeepSeek API Key", type="password", placeholder="sk-...")
    model_size = st.selectbox("Whisper 精度", ["tiny", "base", "small"], index=2)
    academic_mode = st.checkbox("🎓 开启学术分析模式")
    
    if 'whisper_model' not in st.session_state:
        if st.button("🚀 第一步：激活 AI 引擎"):
            with st.spinner(f"正在加载 {model_size} 模型..."):
                st.session_state.whisper_model = whisper.load_model(model_size)
                st.success("✅ 引擎就绪！")

# 主界面
st.title("🎯 AI 口译练习评估系统")
st.write("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📤 源语材料")
    src_file = st.file_uploader("上传原语音频/视频", type=["mp3", "m4a", "wav", "mp4"], key="src")
    if src_file: st.audio(src_file)

with col2:
    st.subheader("📥 你的译语")
    interp_file = st.file_uploader("上传口译录音", type=["mp3", "m4a", "wav", "mp4"], key="interp")
    if interp_file: st.audio(interp_file)

if st.button("🏁 开始自动评估反馈", use_container_width=True):
    if not user_api_key or not src_file or not interp_file or 'whisper_model' not in st.session_state:
        st.error("❌ 请检查：1.API Key 2.文件上传 3.点击左侧加载引擎")
    else:
        client = OpenAI(api_key=user_api_key, base_url="https://api.deepseek.com")
        
        with st.spinner("AI 正在听写并分析中...（长音频请耐心等待）"):
            # 临时保存
            with open("t_src.mp3", "wb") as f: f.write(src_file.getbuffer())
            with open("t_interp.mp3", "wb") as f: f.write(interp_file.getbuffer())

            try:
                # 1. 转写
                model = st.session_state.whisper_model
                res_src = model.transcribe("t_src.mp3", fp16=False)
                res_interp = model.transcribe("t_interp.mp3", fp16=False)

                # 2. 修正
                p_src = polish_transcript(client, res_src["text"], res_src["language"], is_source=True)
                p_interp = polish_transcript(client, res_interp["text"], res_interp["language"], is_source=False, src_context=p_src)

                # 3. 评估
                eval_report = analyze_logic(client, p_src, p_interp, res_src["language"], res_interp["language"], academic=academic_mode)

                # 4. 【关键修正】缝合所有内容到最终报告
                final_report = f"""# 🎧 SISU 口译练习反馈报告
生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}

## 📝 1. 源语转写原文 (Source)
{p_src}

---

## 🎙️ 2. 学生译语文本 (Interpretation)
{p_interp}

---

## 📊 3. AI 深度评估
{eval_report}
"""

                # 5. 展示与下载
                st.success("✅ 分析完成！")
                st.markdown(final_report)
                
                st.download_button(
                    label="📥 下载完整练习记录 (.md)",
                    data=final_report,
                    file_name=f"SISU_Practice_{int(time.time())}.md",
                    mime="text/markdown"
                )

            except Exception as e:
                st.error(f"发生错误: {e}")
            finally:
                if os.path.exists("t_src.mp3"): os.remove("t_src.mp3")
                if os.path.exists("t_interp.mp3"): os.remove("t_interp.mp3")
