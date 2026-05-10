import streamlit as st
import os
import whisper
import time
from openai import OpenAI
import shutil
import gc

# --- 强制修复 FFmpeg 路径问题 ---
# 这段代码会告诉 Python 去哪里找 ffmpeg，解决 RuntimeError
ffmpeg_bin = shutil.which("ffmpeg")
if ffmpeg_bin:
    os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_bin)

# ============================================================
# 1. 基础配置
# ============================================================
LANG_NAMES = {"zh": "中文", "en": "English", "ja": "日本語", "ko": "한국어", "fr": "Français"}

# ============================================================
# 2. 核心算法函数
# ============================================================

def polish_transcript(client, raw_text, lang, is_source=True, src_context=""):
    """使用 DeepSeek 修正转写文本"""
    lang_name = LANG_NAMES.get(lang, lang)
    instruction = f"你是专业的转录校对助手。请修正 {lang_name} 文本的标点、同音字并分段。"
    if not is_source:
        instruction += "务必保留学生的改口、重复、犹豫词（嗯、呃），只修正录音转写错误。"

    prompt = f"{instruction}\n\n=== 参考上下文 ===\n{src_context}\n\n=== 待修正文本 ===\n{raw_text}"
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except:
        return raw_text

def analyze_logic(client, src_text, interp_text, src_lang, interp_lang, academic=False):
    """多维口译评估 - 强化表格格式"""
    prompt = f"""你是一位资深口译教练。请对以下练习进行评估：
    
    ## 源语 (Source): {src_text[:500]}...
    ## 译语 (Interpretation): {interp_text[:500]}...
    
    请严格按以下格式输出：
    1. ### 📊 评分摘要表
    使用 Markdown 表格，包含维度（信息传达、策略运用、语言质量、流利度）、得分（满分25）和核心评价。
    2. ### 📝 详细点评
    分点列出【表现优秀】与【改进空间】。
    3. ### 💡 针对性建议
    给出 3 条提升建议。"""
    
    if academic:
        prompt += "\n4. ### 🎓 学术指标分析\n统计错译、漏译及犹豫词频率。"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "你是一位排版严谨的口译教练，必须使用表格呈现分数。"},
                      {"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"评估失败: {e}"

# ============================================================
# 3. 界面逻辑
# ============================================================
st.set_page_config(page_title="SISU口译教练", layout="wide")

with st.sidebar:
    st.title("⚙️ 配置")
    user_key = st.text_input("🔑 API Key", type="password")
    model_size = st.selectbox("Whisper 精度", ["base", "small"], index=1)
    academic_mode = st.checkbox("🎓 开启学术模式")
    
    if 'whisper_model' not in st.session_state:
        if st.button("🚀 激活 AI 引擎"):
            with st.spinner("正在加载..."):
                st.session_state.whisper_model = whisper.load_model(model_size)
                st.success("引擎就绪")

st.title("🎯 AI 口译练习评估系统")
col1, col2 = st.columns(2)
with col1:
    src_file = st.file_uploader("上传源语", type=["mp3", "m4a", "wav", "mp4"])
with col2:
    interp_file = st.file_uploader("上传译语", type=["mp3", "m4a", "wav", "mp4"])

if st.button("🏁 开始自动评估反馈", use_container_width=True):
    if not user_key or not src_file or not interp_file or 'whisper_model' not in st.session_state:
        st.error("请检查配置、文件及引擎激活状态")
    else:
        client = OpenAI(api_key=user_key, base_url="https://api.deepseek.com")
        with st.spinner("AI 正在深度分析中（长音频可能需要数分钟）..."):
            # 保存临时文件
            with open("s.mp3", "wb") as f: f.write(src_file.getbuffer())
            with open("i.mp3", "wb") as f: f.write(interp_file.getbuffer())

            try:
                # 1. 转写与清理内存
                model = st.session_state.whisper_model
                raw_s = model.transcribe("s.mp3", fp16=False)["text"]
                raw_i = model.transcribe("i.mp3", fp16=False)["text"]
                gc.collect() 

                # 2. AI 修正
                p_src = polish_transcript(client, raw_s, "auto", is_source=True)
                p_interp = polish_transcript(client, raw_i, "auto", is_source=False, src_context=p_src)

                # 3. 评估
                eval_report = analyze_logic(client, p_src, p_interp, "zh", "en", academic=academic_mode)

                # 4. 【核心改进】构建完整报告
                final_output = f"""# 🎧 SISU 口译练习反馈报告\n\n## 1. 源语转写 (Source)\n{p_src}\n\n---\n\n## 2. 学生译语 (Interpretation)\n{p_interp}\n\n---\n\n## 3. AI 深度评估\n{eval_report}"""

                st.success("✅ 分析完成！")
                st.markdown(final_output)
                
                st.download_button(
                    label="📥 下载完整练习记录 (.md)",
                    data=final_output,
                    file_name=f"SISU_Practice_{int(time.time())}.md",
                    mime="text/markdown"
                )
            except Exception as e:
                st.error(f"出错: {e}")
            finally:
                for f in ["s.mp3", "i.mp3"]:
                    if os.path.exists(f): os.remove(f)
