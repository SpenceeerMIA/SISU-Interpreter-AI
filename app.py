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
    """使用 DeepSeek 修正转写文本"""
    lang_name = LANG_NAMES.get(lang, lang)
    instruction = f"你是专业的转录校对助手。请修正 {lang_name} 文本的标点、同音字并分段。"
    if not is_source:
        instruction += "务必保留学生的改口、重复、停顿、犹豫词（嗯、呃），只修正录音转写错误。"

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
    """多维口译评估报告生成"""
    prompt = f"""你是一位资深口译教练。请对以下练习进行评估：
    
    ## 1. 源语原文
    {src_text}
    ## 2. 学生译语
    {interp_text}
    
    请严格按以下模块输出（使用 Markdown 排版）：
    ### 📊 评分摘要表
    使用 Markdown 表格，包含维度（信息传达、策略运用、语言质量、流利度）、得分（满分25）和核心评价。
    ### 📝 详细点评
    分点列出【表现优秀】与【改进空间】。
    ### 💡 针对性建议
    给出 3 条具体的练习建议。"""
    
    if academic:
        prompt += "\n### 🎓 学术指标分析\n统计误译、漏译的具体位置及犹豫词频率。"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "你是一位排版严谨的口译教练，必须使用表格呈现分数。"},
                      {"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"评估分析环节调用失败: {e}"

# ============================================================
# 3. 网页界面 (表情包寄语、彩蛋、布局)
# ============================================================

# --- 侧边栏：配置与个性化展示 ---
with st.sidebar:
    st.title("⚙️ 配置与指引")

    user_api_key = st.text_input("🔑 API Key", type="password", placeholder="sk-...")
    model_size = st.selectbox("🎯 Whisper 精度", ["base", "small"], index=1, help="手机端建议用 Small")
    academic_mode = st.checkbox("🎓 开启学术分析模式")
    
    st.markdown("---")
    
    if 'whisper_model' not in st.session_state:
        if st.button("🚀 第一步：激活 AI 引擎"):
            with st.spinner("正在云端初始化引擎..."):
                st.session_state.whisper_model = whisper.load_model(model_size)
                st.success("✅ 引擎就绪！")
                st.balloons() # ✨ 激活成功彩蛋
    else:
        st.success(f"✅ 引擎 ({model_size}) 运行中")

    st.markdown("---")

    # 🌈 Spencer 寄语与表情包展示区
    
    # 图片上方的文字描述
    st.write("赶论文前中后期的Spencer真实写照：")
    
    # 核心展示逻辑：尝试加载 sisu_meme.png
    if os.path.exists("sisu_meme.png"):
        st.image("sisu_meme.png", use_container_width=True)
    else:
        st.info("🖼️ 待上传表情包：sisu_meme.png")
        st.caption("请确保图片已重命名并上传至 GitHub 仓库根目录。")

    st.markdown("### 💌 Spencer寄语")
    
    # 下方文字分三行独立显示
    st.markdown("""
    正能量来了！  （这里有两个空格）
    **格高志远，学贯中外。**
    祝各位同学练习顺利，早日上岸！
    """)

# --- 主界面 ---
st.title("🏫 SISU 口译练习评估系统")
st.markdown("> **“格高志远 学贯中外”** —— 为上外学子定制的 AI 助学实验项目")

# 🐣 时间彩蛋
current_hour = time.localtime().tm_hour
if 0 <= current_hour < 6:
    st.warning("🌙 **深夜坚持练习的 SISUer：** 辛苦了！但要注意休息，好听力离不开好睡眠。")
elif 6 <= current_hour < 9:
    st.info("☕ **早安！** 又是晨读好时光。")

st.write("---")

# 响应式布局
col1, col2 = st.columns(2)
with col1:
    st.subheader("📤 源语材料 (Source)")
    src_file = st.file_uploader("上传原语音频", type=["mp3", "m4a", "wav", "mp4"], key="src_upload")
    if src_file: st.audio(src_file)

with col2:
    st.subheader("📥 你的译语 (Interpretation)")
    interp_file = st.file_uploader("上传口译录音", type=["mp3", "m4a", "wav", "mp4"], key="int_upload")
    if interp_file: st.audio(interp_file)

st.caption("📱 手机端提示：点击上传框可直接选择本地文件或启动手机录音。")

# 评估执行
if st.button("🏁 第二步：开始自动评估反馈", use_container_width=True):
    if not user_api_key:
        st.error("❌ 请先在左侧边栏填入 API Key！")
    elif not src_file or not interp_file:
        st.error("❌ 请同时上传源语和译语文件！")
    elif 'whisper_model' not in st.session_state:
        st.error("❌ 请先在左侧边栏点击按钮激活 AI 引擎！")
    else:
        client = OpenAI(api_key=user_api_key, base_url="https://api.deepseek.com")
        
        with st.spinner("AI 正在认真听取并分析中（长音频请保持页面开启）..."):
            with open("s.mp3", "wb") as f: f.write(src_file.getbuffer())
            with open("i.mp3", "wb") as f: f.write(interp_file.getbuffer())

            try:
                model = st.session_state.whisper_model
                res_s = model.transcribe("s.mp3", fp16=False)["text"]
                res_i = model.transcribe("i.mp3", fp16=False)["text"]
                gc.collect() 

                st.info("✨ 正在进行文本校对与消歧...")
                p_src = polish_transcript(client, res_s, "auto", is_source=True)
                p_interp = polish_transcript(client, res_i, "auto", is_source=False, src_context=p_src)

                eval_report = analyze_logic(client, p_src, p_interp, "zh", "en", academic=academic_mode)

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
                st.success("✅ 分析完成！")
                st.snow() 
                st.toast("🎉 太棒了，又完成了一次练习！", icon='👏')

                st.markdown(final_output)
                
                st.download_button(
                    label="📥 下载完整练习记录 (.md)",
                    data=final_output,
                    file_name=f"SISU_Practice_{int(time.time())}.md",
                    mime="text/markdown"
                )

            except Exception as e:
                st.error(f"运行出错: {e}")
            finally:
                for f in ["s.mp3", "i.mp3"]:
                    if os.path.exists(f): os.remove(f)
