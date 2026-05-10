import streamlit as st
import os
import whisper
import time
from openai import OpenAI

# ============================================================
# 1. 基础配置与说明
# ============================================================
# 在云端部署时，ffmpeg 会通过 packages.txt 自动安装，不需要手动指定 C 盘路径
# 所以我们把 os.environ["PATH"] 那行去掉了，增加了跨平台兼容性

LANG_NAMES = {
    "zh": "中文", "en": "English", "ja": "日本語",
    "ko": "한국어", "fr": "Français", "de": "Deutsch",
    "es": "Español", "ru": "Русский", "ar": "العربية"
}


# ============================================================
# 2. 核心算法函数 (搬运自你的原代码)
# ============================================================

def polish_transcript(client, raw_text, lang, is_source=True, src_context=""):
    """使用 DeepSeek 修正转写文本"""
    lang_name = LANG_NAMES.get(lang, lang)
    if is_source:
        instruction = f"你是专业的转录校对助手。请对以下 {lang_name} 文本进行修正：添加标点、修正同音字、智能分段并加粗小标题。"
    else:
        # 保留口译过程标记逻辑
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
    """多维口译评估"""
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
        prompt += "\n⚠️ 请额外提供【学术分析表】，包含错误分类统计（漏译、错译等）及流利度指标。"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "你是一位严谨的口译教练，点评需具体且具有学术深度。"},
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
    st.title("⚙️ 助手指引")

    with st.expander("❓ 这是什么？"):
        st.write("""
        本工具由 SISU 学生开发，利用 **Whisper** 进行高精度语音转写，并结合 **DeepSeek** AI 进行口译多维评估。
        - **Whisper**: 你的“数字耳朵”。
        - **DeepSeek**: 你的“AI 考官”。
        """)

    st.markdown("---")

    # 用户输入自己的 API Key
    user_api_key = st