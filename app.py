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

def analyze_logic(client, model_name, src_text, interp_text, academic=False, research_question=""):
    system_msg = "你是一位精通口译理论与实证研究的博导。你必须从提供的上下文中提取大量具体的句子作为例子进行对比分析。"
    
    prompt = f"请评估以下口译练习。\n源语：{src_text}\n译语：{interp_text}\n\n"
    
    if academic:
        prompt += f"""--- 🎓 学术分析模式（上外毕业论文级别数据分析） ---
【指定研究维度】：{research_question}

要求：
1. **多维度实例对比**：请从文中提取 15-20 个具体的【源语片段 vs 学生译语】对比案例。
   - **选例标准**：案例需具备代表性，且包含不同长度的跨度：
     * **微观案例**：5-8 个 1-2 句的短例（侧重词汇、术语或短语处理）。
     * **中观案例**：5-7 个 3-4 句的中长例（侧重句式调整、逻辑衔接）。
     * **宏观案例**：3-5 个 5-6 句的长段落案例（侧重篇章结构、信息整合及脱壳处理）。
   - 请用表格形式呈现：| 编号 | 跨度类型 | 源语片段 | 学生译语 | 推荐译法 |
2. **研究方向深度剖析**：
   - 针对上述 15-20 个案例，深度分析哪些翻译错误是由于「{research_question}」造成的。尤其针对长段落例子，分析在高认知负荷下，该因素如何导致了译语质量下滑。
3. **真实学术文献推荐**：
   - 请列举 3-5 篇与「{research_question}」相关的**真实可查**的学术论文。
   - 格式：[作者]. [标题]. [刊名]. [年份].
"""
    else:
        prompt += "要求：从信息完整度、流利度给出评分，并提供 5 个以上的具体重点词汇对照积累。"

    try:
        response = client.chat.completions.create(
            model=model_name, 
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}], 
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e: return f"分析失败: {e}"

# ============================================================
# 3. 网页界面
# ============================================================

with st.sidebar:
    st.title("⚙️ 配置与指引")

    user_api_key = st.text_input(
        "🔑 输入 API Key", type="password", placeholder="在此粘贴你的 sk-...", key="ds_api_key_v2",
        help="API Key 是敏感信息，已自动加密。\n\n建议到官网申请一个 API Key，不测也给我去领一个！万一哪天要用呢？"
    )
    api_base = st.text_input("🌐 API Base URL", value="", placeholder="例如: https://api.deepseek.com/v1", help="请输入服务商提供的 API 基础地址（兼容 OpenAI 格式）。")
    model_id = st.text_input("🤖 模型名称 (Model ID)", value="", placeholder="例如: deepseek-chat 或 gpt-4o", help="填入具体的模型代号即可。")
    
    model_size = st.selectbox("🎯 Whisper 精度", ["base", "small"], index=1, help="💰有限，其他size缺货中...")
    academic_mode = st.checkbox("🎓 开启学术分析模式")
    research_q = ""
    if academic_mode:
        research_q = st.text_input("🔍 请输入你的研究分析方向", placeholder="如：听辨障碍、脱壳处理等", help="填入你的毕业论文研究方向，AI 会帮你抓 15-20 个长短不一的典型案例。")

    st.markdown("---")
    
    if 'whisper_model' not in st.session_state:
        if st.button("🚀 第一步：激活 AI 引擎"):
            with st.spinner("正在云端初始化引擎..."):
                st.session_state.whisper_model = whisper.load_model(model_size)
                st.success("✅ 引擎就绪！")
    else:
        st.success(f"✅ 引擎 ({model_size}) 运行中")

    st.markdown("---")
    st.write("赶论文前中后期的Spencer真实写照：")
    if os.path.exists("sisu_meme.png"):
        st.image("sisu_meme.png", use_container_width=True)
    else:
        st.info("🖼️ 待上传：sisu_meme.png (没有表情包的工具是没有灵魂的)")

    st.markdown("### 💌 Spencer寄语")
    st.markdown("正能量来了！\n\n**格高志远，学贯中外。**\n\n祝各位同学练习顺利，早日上岸！")

# --- 主界面 ---
st.title("🏫 SISU 口译练习评估系统")
st.markdown("> **“格高志远 学贯中外”** —— 为上外学子定制的 AI 助学实验项目")

# --- 动态时间寄语逻辑 ---
current_hour = time.localtime().tm_hour
if 0 <= current_hour < 6:
    st.warning("🌙 **很晚了！！** 求你，赶紧睡吧！")
elif 6 <= current_hour < 8:
    st.info("☕ **早啊！** 一日之计在于晨，看到这个，说明你起得很早了，再睡会儿吧，你很努力了")

st.info("📱 **手机端用户：** 点击下方上传框后，可直接选择“录音机”现场录制你的翻译。")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📤 源语材料")
    src_file = st.file_uploader("上传原语音频", type=["mp3", "m4a", "wav", "mp4", "docx", "txt"], key="src_up")
    if src_file and not src_file.name.lower().endswith(('.docx', '.txt')): st.audio(src_file)

with col2:
    st.subheader("📥 你的译语")
    interp_file = st.file_uploader("上传口译录音", type=["mp3", "m4a", "wav", "mp4", "docx", "txt"], key="int_up")
    if interp_file and not interp_file.name.lower().endswith(('.docx', '.txt')): st.audio(interp_file)

if st.button("🏁 第二步：开始自动评估反馈", use_container_width=True):
    if not user_api_key or not api_base or not model_id:
        st.error("❌ 请先在左侧边栏完整填写 API 配置！")
    elif not src_file or not interp_file:
        st.error("❌ 请同时上传源语和译语文件！")
    elif 'whisper_model' not in st.session_state:
        st.error("❌ 请先点击左侧按钮激活 AI 引擎！")
    else:
        client = OpenAI(api_key=user_api_key, base_url=api_base)
        with st.spinner("AI 正在深度剖析内容..."):
            try:
                # A. 处理源语
                if src_file.name.lower().endswith('.docx'): p_src_raw = extract_text_from_docx(src_file)
                elif src_file.name.lower().endswith('.txt'): p_src_raw = src_file.read().decode("utf-8")
                else:
                    with open("s.mp3", "wb") as f: f.write(src_file.getbuffer())
                    p_src_raw = st.session_state.whisper_model.transcribe("s.mp3", fp16=False)["text"]

                # B. 处理译语
                if interp_file.name.lower().endswith('.docx'): p_interp_raw = extract_text_from_docx(interp_file)
                elif interp_file.name.lower().endswith('.txt'): p_interp_raw = interp_file.read().decode("utf-8")
                else:
                    with open("i.mp3", "wb") as f: f.write(interp_file.getbuffer())
                    p_interp_raw = st.session_state.whisper_model.transcribe("i.mp3", fp16=False)["text"]

                # C. 校对与评估
                p_src = polish_transcript(client, model_id, p_src_raw, "auto", is_source=True)
                p_interp = polish_transcript(client, model_id, p_interp_raw, "auto", is_source=False, src_context=p_src)
                eval_report = analyze_logic(client, model_id, p_src, p_interp, academic=academic_mode, research_question=research_q)

                st.markdown(f"### 📊 评估报告\n\n{eval_report}")
                st.download_button(label="📥 下载报告 (.md)", data=eval_report, file_name=f"SISU_Analysis_{int(time.time())}.md")
                st.success("✅ 分析完成！")
            except Exception as e: st.error(f"处理过程中出错: {e}")
            finally:
                gc.collect()
                for f in ["s.mp3", "i.mp3"]:
                    if os.path.exists(f): os.remove(f)
