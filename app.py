import streamlit as st
import os
import whisper
import time
from openai import OpenAI
import shutil
import gc

# --- 【核心修复】强制锁定 FFmpeg 路径 ---
ffmpeg_bin = shutil.which("ffmpeg")
if ffmpeg_bin:
    os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_bin)

# ============================================================
# 1. 基础配置
# ============================================================
LANG_NAMES = {
    "zh": "中文", "en": "English", "ja": "日本語",
    "ko": "한국어", "fr": "Français", "de": "Deutsch"
}

# ============================================================
# 2. 核心 AI 逻辑函数
# ============================================================

def polish_transcript(client, raw_text, lang, is_source=True, src_context=""):
    """使用 DeepSeek 弥补 Whisper Small 的精度损失"""
    lang_name = LANG_NAMES.get(lang, lang)
    if is_source:
        instruction = f"你是专业的转录校对助手。请对以下 {lang_name} 文本进行修正：添加标点、修正同音字、智能分段。"
    else:
        instruction = f"你是专业的转录校对助手。请对照源语上下文消歧。务必保留学生的改口、重复、停顿、犹豫词（嗯、呃、那个），只修正转写错误。"

    prompt = f"{instruction}\n\n=== 参考背景 ===\n{src_context}\n\n=== 待修正文本 ===\n{raw_text}"
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
    """生成结构化的口译评估报告"""
    prompt = f"""你是一位资深口译教练。请对以下练习进行评估：
    
    ## 1. 源语原文
    {src_text}
    ## 2. 学生译语
    {interp_text}
    
    请严格按以下模块输出（使用 Markdown 排版）：
    
    ### 📊 评分摘要表
    | 维度 | 分数 (满分25) | 核心评价 |
    | :--- | :--- | :--- |
    | 信息传达 | | |
    | 策略运用 | | |
    | 语言质量 | | |
    | 流利稳定 | | |

    ### 📝 详细点评
    - **【表现优秀】**：...
    - **【存在不足】**：...

    ### 💡 针对性建议
    1. ...
    """
    if academic:
        prompt += "\n### 🎓 学术指标分析\n列出误译/漏译统计及犹豫词频率分析。"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "你是一位排版严谨的口译教练，必须使用表格呈现分数。"},
                      {"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"评估调用失败: {e}"

# ============================================================
# 3. 网页界面 (恢复友善提示与 SISU 风格)
# ============================================================
st.set_page_config(page_title="SISU 口译教练 AI", layout="wide", page_icon="🎯")

# 侧边栏：友善引导
with st.sidebar:
    st.title("⚙️ 助手配置与指引")
    st.info("👋 **Hello SISUer!** 本工具旨在辅助口译练习，结合 Whisper 转写与 DeepSeek 深度分析。")
    
    with st.expander("📖 如何开始？"):
        st.write("""
        1. 填入你的 **DeepSeek API Key**。
        2. 点击下方按钮 **'激活 AI 引擎'**。
        3. 上传音频后，点击页面底部的 **'开始评估'**。
        """)

    user_api_key = st.text_input("🔑 DeepSeek API Key", type="password", placeholder="sk-...")
    model_size = st.selectbox("🎯 Whisper 转写精度", ["base", "small"], index=1, help="Small 模型更精准，但处理较慢")
    academic_mode = st.checkbox("🎓 开启学术模式 (增加量化指标)")
    
    st.markdown("---")
    
    if 'whisper_model' not in st.session_state:
        if st.button("🚀 第一步：激活 AI 引擎"):
            with st.spinner(f"正在加载 {model_size} 引擎..."):
                st.session_state.whisper_model = whisper.load_model(model_size)
                st.success("✅ 引擎就绪！")
    else:
        st.success(f"✅ 引擎 ({model_size}) 运行中")

# 主界面
st.title("🎯 AI 口译练习评估系统")
st.caption("上海国际外国语大学 (SISU) 翻译学院练习专用 | Powered by Whisper & DeepSeek")
st.write("---")

# 上传区
col1, col2 = st.columns(2)
with col1:
    st.subheader("📤 源语材料 (Source)")
    src_file = st.file_uploader("上传原语音频", type=["mp3", "m4a", "wav", "mp4"], key="src_up")
    if src_file: st.audio(src_file)

with col2:
    st.subheader("📥 你的译语 (Interpretation)")
    interp_file = st.file_uploader("上传口译录音", type=["mp3", "m4a", "wav", "mp4"], key="int_up")
    if interp_file: st.audio(interp_file)

st.markdown("<br>", unsafe_allow_html=True)

# 评估执行
if st.button("🏁 第二步：开始自动评估反馈", use_container_width=True):
    if not user_api_key:
        st.error("❌ 请先在左侧侧边栏填入 API Key！")
    elif not src_file or not interp_file:
        st.error("❌ 请同时上传源语和译语文件！")
    elif 'whisper_model' not in st.session_state:
        st.error("❌ 请先点击左侧按钮激活 AI 引擎！")
    else:
        client = OpenAI(api_key=user_api_key, base_url="https://api.deepseek.com")
        
        with st.spinner("AI 正在认真听取并分析中（5-10分钟长音频可能需要较长时间，请勿关闭页面）..."):
            # 临时保存
            with open("s.mp3", "wb") as f: f.write(src_file.getbuffer())
            with open("i.mp3", "wb") as f: f.write(interp_file.getbuffer())

            try:
                # 1. 转写
                model = st.session_state.whisper_model
                raw_s = model.transcribe("s.mp3", fp16=False)["text"]
                raw_i = model.transcribe("i.mp3", fp16=False)["text"]
                
                # 显式回收内存，防止长音频撑爆服务器
                gc.collect()

                # 2. 文本校对
                st.info("✨ 正在进行文本校对与消歧...")
                p_src = polish_transcript(client, raw_s, "auto", is_source=True)
                p_interp = polish_transcript(client, raw_i, "auto", is_source=False, src_context=p_src)

                # 3. 评估
                eval_report = analyze_logic(client, p_src, p_interp, "zh", "en", academic=academic_mode)

                # 4. 构建完整下载内容
                final_output = f"""# 🎧 SISU 口译练习反馈报告
生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}

---

## 📝 1. 源语转写原文 (Source)
{p_src}

---

## 🎙️ 2. 学生译语文本 (Interpretation)
{p_interp}

---

## 📊 3. AI 深度评估
{eval_report}
"""

                # 5. 结果展示
                st.success("✅ 评估完成！请查看下方报告或下载记录。")
                st.markdown(final_output)
                
                st.download_button(
                    label="📥 下载完整练习记录 (Obsidian 友好版)",
                    data=final_output,
                    file_name=f"SISU_Practice_{int(time.time())}.md",
                    mime="text/markdown"
                )

            except Exception as e:
                st.error(f"运行出错: {e}")
            finally:
                # 清理文件
                for f in ["s.mp3", "i.mp3"]:
                    if os.path.exists(f): os.remove(f)
