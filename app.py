import streamlit as st
import os
import whisper
import time
from openai import OpenAI
import shutil
import gc

# --- 【环境修复】锁定云端 FFmpeg 路径 ---
ffmpeg_bin = shutil.which("ffmpeg")
if ffmpeg_bin:
    os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_bin)

# ============================================================
# 1. 基础配置与移动端适配
# ============================================================
LANG_NAMES = {"zh": "中文", "en": "English", "ja": "日本語", "ko": "한국어", "fr": "Français"}

st.set_page_config(
    page_title="SISU口译教练 AI", 
    layout="wide", 
    page_icon="🏫",
    initial_sidebar_state="auto"
)

# ============================================================
# 2. 核心 AI 逻辑函数
# ============================================================

def polish_transcript(client, raw_text, lang, is_source=True, src_context=""):
    lang_name = LANG_NAMES.get(lang, lang)
    instruction = f"你是专业的转录校对助手。请修正 {lang_name} 文本的标点、同音字并分段。"
    if not is_source:
        instruction += "务必保留学生的改口、重复、停顿、犹豫词（嗯、呃），只修正录音转写错误。"
    prompt = f"{instruction}\n\n=== 参考上下文 ===\n{src_context}\n\n=== 待修正文本 ===\n{raw_text}"
    try:
        response = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.2)
        return response.choices[0].message.content.strip()
    except: return raw_text

def analyze_logic(client, src_text, interp_text, src_lang, interp_lang, academic=False):
    prompt = f"你是一位资深口译教练。请对以下练习进行评估：\n\n## 1. 源语原文\n{src_text}\n## 2. 学生译语\n{interp_text}\n\n请严格按模块输出报告。"
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
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

    # 修复点 1：删掉 index=1
    user_api_key = st.text_input(
        "🔑 API Key", 
        help="建议到DeepSeek官网申请一个API Key，不测也给我去领一个！", 
        type="password", 
        placeholder="sk-...",
        autocomplete="new-password" # 💡 关键点：告诉浏览器这是“新密码”，不要尝试自动填入旧密码
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
    
    # Spencer 寄语区
    st.write("赶论文前中后期的Spencer真实写照：")
    if os.path.exists("sisu_meme.png"):
        st.image("sisu_meme.png", use_container_width=True)
    else:
        st.info("🖼️ 待上传：sisu_meme.png")

    st.markdown("### 💌 Spencer寄语")
    
    # 修复点 2：使用空行确保分三行显示
    st.markdown("""
    正能量来了！

    **格高志远，学贯中外。**

    祝各位同学练习顺利，早日上岸！
    """)

# --- 主界面 ---
st.title("🏫 SISU 口译练习评估系统")
st.markdown("> **“格高志远 学贯中外”** —— 为上外学子定制的 AI 助学实验项目")

current_hour = time.localtime().tm_hour
if 0 <= current_hour < 6:
    st.warning("🌙 **很晚了！！** 求你，赶紧睡吧！")
elif 6 <= current_hour < 9:
    st.info("☕ **早啊！** 一日之计在于晨，看到这个，说明你起得很早了，先睡会儿吧（")

st.write("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📤 源语材料")
    src_file = st.file_uploader("上传原语音频", type=["mp3", "m4a", "wav", "mp4"], key="src_up")
    if src_file: st.audio(src_file)

with col2:
    st.subheader("📥 你的译语")
    interp_file = st.file_uploader("上传口译录音", type=["mp3", "m4a", "wav", "mp4"], key="int_up")
    if interp_file: st.audio(interp_file)

if st.button("🏁 第二步：开始自动评估反馈", use_container_width=True):
    if not user_api_key or not src_file or not interp_file or 'whisper_model' not in st.session_state:
        st.error("❌ 请检查配置和文件。")
    else:
        client = OpenAI(api_key=user_api_key, base_url="https://api.deepseek.com")
        with st.spinner("AI 正在认真听取并分析中..."):
            with open("s.mp3", "wb") as f: f.write(src_file.getbuffer())
            with open("i.mp3", "wb") as f: f.write(interp_file.getbuffer())
            try:
                model = st.session_state.whisper_model
                res_s = model.transcribe("s.mp3", fp16=False)["text"]
                res_i = model.transcribe("i.mp3", fp16=False)["text"]
                gc.collect() 
                p_src = polish_transcript(client, res_s, "auto")
                p_interp = polish_transcript(client, res_i, "auto", is_source=False, src_context=p_src)
                eval_report = analyze_logic(client, p_src, p_interp, "zh", "en", academic=academic_mode)
                
                final_output = f"# 🎧 反馈报告\n\n## 📝 源语原文\n{p_src}\n\n## 🎙️ 学生译语\n{p_interp}\n\n## 📊 AI 评估\n{eval_report}"
                st.success("✅ 分析完成！")
                st.snow() 
                st.markdown(final_output)
                st.download_button("📥 下载记录 (.md)", final_output, file_name="SISU_Practice.md")
            except Exception as e:
                st.error(f"出错: {e}")
            finally:
                for f in ["s.mp3", "i.mp3"]:
                    if os.path.exists(f): os.remove(f)
