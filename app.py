import os
os.environ["PATH"] += os.pathsep + r'C:\ffmpeg\bin'

import whisper
import time
import re
from openai import OpenAI

# ============================================================
# 配置区 —— 按你的实际路径修改
# ============================================================
WATCH_DIR = r"D:\我的翻译研究\Attachments"      # 音视频存放目录
OUTPUT_DIR = r"D:\我的翻译研究\Transcripts"     # 笔记输出目录
MODEL_SIZE = None  # 启动时由用户选择
DEEPSEEK_API_KEY = "sk-8f589905b1164ec0855811cfe46e2f1e"

# Whisper + ffmpeg 可以处理几乎所有音视频格式，这里列出常见的
# 同时支持直接读入 .txt 文本（跳过转写）
MEDIA_EXTS = (
    # 纯音频
    '.mp3', '.m4a', '.wav', '.ogg', '.flac', '.wma', '.aac', '.opus', '.wav',
    # 视频
    '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.ts', '.mts', '.m2ts',
    '.3gp', '.wmv', '.asf', '.vob', '.rm', '.rmvb', '.mpeg', '.mpg',
    # 文本（跳过 Whisper 转写，直接分析）
    '.txt',
)

AUDIO_EXTS = {'.mp3', '.m4a', '.wav', '.ogg', '.flac', '.wma', '.aac', '.opus'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.ts', '.mts', '.m2ts', '.3gp', '.wmv', '.asf', '.vob', '.rm', '.rmvb', '.mpeg', '.mpg'}

LANG_NAMES = {
    "zh": "中文", "en": "English", "ja": "日本語",
    "ko": "한국어", "fr": "Français", "de": "Deutsch",
    "es": "Español", "ru": "Русский", "ar": "العربية"
}

# ============================================================
# 初始化
# ============================================================
def select_model():
    """让用户选择 Whisper 模型大小"""
    model_map = {
        "1": ("basic", "base"),
        "2": ("small", "small"),
        "3": ("medium", "medium"),
        "4": ("large", "large"),
    }
    print()
    print("请选择 Whisper 模型大小：")
    print("  [1] basic  （最快，准确率较低）")
    print("  [2] small  （较快）")
    print("  [3] medium （推荐，平衡速度与准确率）")
    print("  [4] large  （最准确，速度最慢）")
    while True:
        choice = input("\n请输入 (1-4，回车默认 medium): ").strip()
        if not choice:
            return "medium"
        if choice in model_map:
            display, actual = model_map[choice]
            print(f"  已选择: {display}")
            return actual
        print("  无效选项，请输入 1-4")


client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

MODEL_SIZE = select_model()

print(f"正在加载 Whisper {MODEL_SIZE} 模型，请稍候...")
model = whisper.load_model(MODEL_SIZE)
print("✅ 模型加载完成\n")


# ============================================================
# 工具函数
# ============================================================

def get_audio_files():
    """获取 watch 目录下所有媒体文件"""
    files = []
    if not os.path.exists(WATCH_DIR):
        print(f"❌ 目录不存在: {WATCH_DIR}")
        return files
    for f in os.listdir(WATCH_DIR):
        if os.path.splitext(f)[1].lower() in MEDIA_EXTS:
            files.append(f)
    return sorted(files)


def sanitize_filename(name):
    """去除文件名中的非法字符"""
    return re.sub(r'[<>:"/\\|?*]', '_', name)


def is_audio(filename):
    """检查是否为音频文件（基于扩展名）"""
    return os.path.splitext(filename)[1].lower() in AUDIO_EXTS


def is_video(filename):
    """检查是否为视频文件（基于扩展名）"""
    return os.path.splitext(filename)[1].lower() in VIDEO_EXTS



def transcribe(file_path):
    """转写单个文件。音频/视频走 Whisper，文本文件直接读取。返回 (text, language, is_text)"""
    full_path = os.path.join(WATCH_DIR, file_path)
    ext = os.path.splitext(file_path)[1].lower()

    # 文本文件直接读取
    if ext == '.txt':
        print(f"  📖 读取文本: {file_path} ...")
        with open(full_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        return text, "text", True

    # 音视频走 Whisper
    print(f"  🎤 转写中: {file_path} ...")
    t0 = time.time()
    result = model.transcribe(full_path, fp16=False)
    lang = result.get("language", "unknown")
    elapsed = int(time.time() - t0)
    print(f"  ✅ 转写完成 | 语言: {LANG_NAMES.get(lang, lang)} | 耗时: {elapsed}s")
    return result["text"].strip(), lang, False


def print_box(title, lines, width=56):
    """打印一个视觉明显的框（纯 ASCII，兼容所有终端）"""
    print()
    print("+" + "-" * (width - 2) + "+")
    print("| " + title.ljust(width - 4) + " |")
    print("+" + "-" * (width - 2) + "+")
    for line in lines:
        print("|  " + line.ljust(width - 6) + "  |")
    print("+" + "-" * (width - 2) + "+")
    print()


def select_file(title, subtitle, files, exclude=None):
    """引导式文件选择 —— 显示标题 → 列出文件 → 用户提交"""
    available = [f for f in files if f != exclude]
    if not available:
        print("❌ 没有可选文件")
        return None

    print_box(title, [subtitle] + [""] + [f"📄 {f}" for f in available])

    while True:
        choice = input("  👉 请输入文件编号或文件名: ").strip()

        # 先尝试按编号匹配
        if choice.isdigit() and 0 <= int(choice) < len(available):
            return available[int(choice)]

        # 再尝试按文件名模糊匹配
        matches = [f for f in available if choice.lower() in f.lower()]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            print(f"  ⚠️ 找到多个匹配文件，请用编号选择:")
            for i, f in enumerate(matches):
                print(f"     [{available.index(f)}] {f}")
        else:
            print(f"  ❌ 未找到匹配文件，请重试")


# ============================================================
# DeepSeek 转写修正（标点 + 同音字 + 分段）
# ============================================================

def polish_transcript(raw_text, lang, is_source=True, src_context=""):
    """用 DeepSeek 修正 Whisper 的原始转写文本。
    当修正译语时，传入 src_context（已修正的源语文本）用于对照消歧。"""
    lang_name = LANG_NAMES.get(lang, lang)

    if is_source:
        instruction = f"""你是专业的转录校对助手。请对以下 Whisper 自动转写的 {lang_name} 文本进行修正：

1. **添加标点符号**：根据语义添加正确的标点（句号、逗号、问号、引号、顿号等）。中文必须使用全角标点。
2. **修正同音字/近音字**：根据上下文判断正确的词汇。例如音频说"三分之一"，转写成了"1%"，应改回"三分之一"。英文注意 there/their/they're、to/too/two 等。
3. **修正数字格式**：将数字按口语实际读法修正（如"3-5 年"→"三到五年"，结合语境判断）。
4. **智能分段**：根据语义分成多个段落，为每段添加**加粗小标题**。
5. **保留原意**：不要增删或改写内容，只修正明显的转写错误。"""
    else:
        instruction = f"""你是专业的转录校对助手。请对以下 Whisper 自动转写的口译录音（{lang_name}）进行修正。

★★★ 核心原则：对照源语上下文消歧，保留口译过程标记 ★★★

一、【对照源语消歧】
- 译语中某些词被 Whisper 转写成了更常见的同音词（如"垃圾带"→"垃圾袋"），但结合源语主题判断，这些常见词在语境中不合理。
- 仔细对比源语主题和关键术语，找出此类错误并修正。
  例如：源语谈海洋污染 "garbage patch" → 转写为"垃圾袋" → 修正为"垃圾带"
       源语谈军事 "offensive" → 转写为"公式" → 修正为"攻势"

二、【保留口译过程标记】（非常重要！）
- 务必保留学生的改口、重复、停顿、犹豫词（"嗯"、"呃"、"那个"、"就是"等）
- 不要删除或美化这些表达，它们是口译评估的重要依据，反映了学生的真实输出过程
- 只添加标点，不合并重复句段

三、【标点与分段】
- 添加正确的标点符号。中文使用全角标点。
- 根据语义进行合理分段，每段加**加粗小标题**。
- 修正数字和百分比转写错误。

四、【不得修改学生措辞】
只修正 Whisper 转写层面的错误（同音字、标点），保留学生的语法、用词、句式"""

    if src_context:
        prompt = f"""{instruction}

=== 源语原文（已修正，供上下文对照消歧）===
{src_context}

=== 译语原始转写（待修正）===
{raw_text}

请对照源语上下文，修正译语中的同音字错误后直接输出 Markdown："""
    else:
        prompt = f"""{instruction}

原始转写文本：
{raw_text}

请直接输出修正后的 Markdown 文本："""

    print(f"  🔧 正在 AI 修正转写（标点 / 上下文消歧 / 分段）...")
    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是专业的转录校对助手。你善于结合上下文判断同音字的正确写法，只修正转写错误，绝不改变原文措辞。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            stream=False
        )
        elapsed = int(time.time() - t0)
        print(f"  ✅ 修正完成 | 耗时: {elapsed}s")
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠️ 修正失败，使用原始文本: {e}")
        return raw_text


# ============================================================
# DeepSeek 口译评估 Prompt
# ============================================================

def build_prompt(src_text, interp_text, src_lang, interp_lang, academic=False):
    src_name = LANG_NAMES.get(src_lang, src_lang)
    interp_name = LANG_NAMES.get(interp_lang, interp_lang)

    base = f"""
你是一位资深口译教练，母语水平覆盖中文和英文。请对以下学生的口译练习从**口译视角**进行评估。

**翻译方向**: {src_name} → {interp_name}

## 源语原文
{src_text}

## 学生译语
{interp_text}

---

### 🎯 口译评估理念

口译是**实时口语沟通**，不同于笔译可以反复推敲。核心标准是：**听众能否准确理解源语传达的信息**。

- ✅ 允许学生使用合理的口译策略（意译、概括、重组、解释性增补等）
- ✅ 允许多种不同但等效的表达方式
- ❌ 不要逐字对照苛求"字对字"准确
- ❌ 不要过分纠结不影响理解的细微语法或用词差异
- ✅ 关注信息传达的有效性和译语的自然度

---

### 1. 信息传达效果（最重要）

评估核心：源语的关键信息是否有效传达给了听众？

- 源语的核心论点/主要信息是否完整呈现？
- **严重漏译**：是否有重要信息遗漏导致听众无法理解？（次要细节省略可接受）
- **严重错译**：是否有信息扭曲导致听众误解？
- 数字、专有名词等关键信息是否准确传达？

请逐条举例说明。

### 2. 口译策略运用

学生是否运用了适当的口译策略？

- **意译/释译**：是否用听众易懂的方式解释了源语概念？
- **信息重组**：是否按译语习惯调整了语序和表达方式？
- **概括/省译**：是否合理省略了冗余信息？
- **策略失误**：哪些策略运用不当或导致信息损失？

### 3. 译语表达质量

- 译语是否自然、地道，符合{interp_name}表达习惯？
- 是否存在明显"翻译腔"？（语法正确但母语者不会那样说）
- 表达是否清晰易懂？

### 4. 术语与关键信息

- 专有名词（人名、地名、机构名等）是否处理得当？
- 专业术语是否准确或在语境中可理解？
- 数字、百分比、日期是否准确传达？

### 5. 流利度与稳定性

- 语流是否平稳顺畅？
- 改口、重复、填充词（嗯/呃/那个）是否过多影响听感？
- （注意：口译中适度停顿和填充词属正常，仅指出过度情况）

### 6. 整体评价

- **主要优势**：学生在本次口译中哪些方面做得好？
- **最需改进**：如果只选一个最关键的改进点，是什么？
- **练习建议**：针对发现的问题给出具体可操作的练习方法

---

### 📊 综合评分

| 维度 | 得分(1-10) | 简评 |
|------|-----------|------|
| 信息传达效果 | /10 | |
| 口译策略运用 | /10 | |
| 译语表达质量 | /10 | |
| 术语与信息处理 | /10 | |
| 流利度与稳定性 | /10 | |
| **平均分** | **/10** | |

### 参考译文（高质量{interp_name}版本，体现信息准确传达、表达自然地道）


### 📝 重点词汇与表达积累

| 原文 | 学生译法 | 推荐表达 | 备注 |
|------|----------|---------|------|
| ...  | ...      | ...     | ...  |

---

输出请使用适合 Obsidian 阅读的 Markdown 格式。
"""

    if not academic:
        return base

    # ── 学术分析模式（上海外国语大学口译毕业论文专用）──
    academic_section = f"""

---

## 🎓 学术分析（毕业论文专用）

### 1. 转写规范化说明
- 本次转写是否已按照口译语料库规范处理（标注改口、犹豫、填充词等）？
- 如转写文本未经规范化，此处指出存在哪些不规范之处

### 2. 错误分类统计表

| 错误类型 | 出现次数 | 占比 | 典型例子 |
|----------|----------|------|----------|
| 漏译（信息缺失） | | | |
| 错译（信息错误） | | | |
| 冗余（过度翻译） | | | |
| 表达不当 | | | |
| 语法错误 | | | |
| 术语错误 | | | |
| **合计** | | 100% | |

### 3. 流利度指标
- **总词数/字数**:
- **修正次数**（改口/自我纠正次数）:
- **填充词次数**（嗯/呃/那个/就是等）:
- **平均语速**:
- **流利度综合评价**:

### 4. 详细例证分析
（选取 3-5 个最具代表性的错误，逐一进行原文-译语-分析-修正的详细对比分析）

### 5. 学习策略建议
针对本次练习暴露的问题，以学术论文"建议"部分的风格撰写改进策略

---

> 📌 **注**: 以上学术分析内容可直接用于毕业论文的"口译实践报告"章节。
"""

    return base + academic_section


def analyze(src_text, interp_text, src_lang, interp_lang, academic=False):
    """调用 DeepSeek 进行多维评估"""
    prompt = build_prompt(src_text, interp_text, src_lang, interp_lang, academic=academic)
    print("  🪄 正在调用 AI 进行口译评估（可能需要 30-60 秒）...")
    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位资深口译教练，擅长从多个维度评估口译质量，给出具体、可操作的改进建议。你必须严格按照要求的 Markdown 格式输出，所有点评使用中文。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            stream=False
        )
        elapsed = int(time.time() - t0)
        print(f"  ✅ AI 评估完成 | 耗时: {elapsed}s")
        return response.choices[0].message.content
    except Exception as e:
        print(f"  ❌ AI 调用失败: {e}")
        return f"❌ 分析失败: {e}"


# ============================================================
# Markdown 输出
# ============================================================

def generate_markdown(src_file, interp_file,
                      raw_src, raw_interp,
                      src_text, interp_text,
                      src_lang, interp_lang, analysis,
                      src_is_text=False, interp_is_text=False):
    src_name = LANG_NAMES.get(src_lang, src_lang)
    interp_name = LANG_NAMES.get(interp_lang, interp_lang)

    topic = os.path.splitext(src_file)[0]
    for suffix in ['_source', '_src', '_源语', '_origin', '_original']:
        topic = topic.replace(suffix, '')

    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    date_str = time.strftime('%Y%m%d_%H%M%S')
    safe_topic = sanitize_filename(topic)

    # 构建源语媒体嵌入（文本文件不展示媒体）
    # 音频: 内嵌播放器（紧凑，不占空间，滚动不停止）
    # 视频: 内嵌播放器（限制宽度，右键可选独立窗口播放）
    if not src_is_text:
        if is_audio(src_file):
            src_media = f"![[{src_file}]]\n"
        elif is_video(src_file):
            src_media = f"![[{src_file}|400]]\n\n> 💡 右键视频 → 「在新窗口中打开」可固定播放\n"
        else:
            src_media = f"![[{src_file}]]\n"
    else:
        src_media = ""

    if not interp_is_text:
        if is_audio(interp_file):
            interp_media = f"![[{interp_file}]]\n"
        elif is_video(interp_file):
            interp_media = f"![[{interp_file}|400]]\n\n> 💡 右键视频 → 「在新窗口中打开」可固定播放\n"
        else:
            interp_media = f"![[{interp_file}]]\n"
    else:
        interp_media = ""

    # 是否显示原始转写（仅非文本模式才需要）
    raw_src_note = f"\n> [!note]- 📄 查看 Whisper 原始转写\n> {raw_src}\n" if not src_is_text else ""
    raw_interp_note = f"\n> [!note]- 📄 查看 Whisper 原始转写\n> {raw_interp}\n" if not interp_is_text else ""

    md = f"""# 🎯 口译练习反馈

## 📋 基本信息

| 项目 | 内容 |
|------|------|
| 练习时间 | {timestamp} |
| 翻译方向 | {src_name} → {interp_name} |
| 源语来源 | {"文本文件" if src_is_text else "音视频文件"} |
| 译语来源 | {"文本文件" if interp_is_text else "音视频文件"} |

---

## 🔉 源语

{src_media}

### 📝 源语原文（{src_name}）

{src_text}{raw_src_note}

---

## 🎤 译语

{interp_media}

### 🗣️ 你的译语（{interp_name}）

{interp_text}{raw_interp_note}

---

## 📊 AI 口译评估

{analysis}

---

> 💡 **练习建议**: 对照参考译文和 AI 点评中的具体问题，逐句复盘。同一段材料建议反复练习，直到评分稳定在 8 分以上再换新材料。
"""

    output_file = os.path.join(OUTPUT_DIR, f"{safe_topic}_口译反馈_{date_str}.md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md)
    return output_file


# ============================================================
# 单文件转写模式（保留原功能）
# ============================================================

def single_transcribe_mode():
    """单文件转写 + AI 修正，不进行口译评估"""
    files = get_audio_files()
    if not files:
        print("❌ 没有找到媒体文件")
        return

    target = select_file(
        title="🎤 单文件转写",
        subtitle="选择你要转写的媒体文件",
        files=files,
    )
    if not target:
        return

    raw_text, lang, is_text = transcribe(target)
    if not raw_text:
        print("❌ 转写结果为空")
        return

    name = os.path.splitext(target)[0]
    output_file = os.path.join(OUTPUT_DIR, f"{sanitize_filename(name)}_转写.md")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# 音频转写: {name}\n\n")

        # 文本文件不嵌入媒体链接
        if not is_text:
            if is_audio(target):
                f.write(f"![[{target}]]\n\n")
            elif is_video(target):
                f.write(f"![[{target}|400]]\n\n")
            else:
                f.write(f"![[{target}]]\n\n")
            f.write(f"- **语言**: {LANG_NAMES.get(lang, lang)}\n")
        else:
            f.write(f"- **来源**: 文本文件\n")

        f.write(f"- **时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"---\n\n")

        # AI 修正（仅音视频需要）
        if is_text:
            f.write(raw_text)
        else:
            print()
            polished = polish_transcript(raw_text, lang, is_source=True)
            f.write(polished)
            f.write(f"\n\n> [!note]- 📄 查看 Whisper 原始转写\n> {raw_text}\n")

    print(f"\n✅ 转写完成！已保存至: {output_file}")


# ============================================================
# 主入口
# ============================================================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("=" * 50)
    print("🎯 AI 口译练习教练")
    print("=" * 50)
    print(f"📁 媒体目录: {WATCH_DIR}")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print()

    while True:
        print("请选择模式:")
        print("  [1] 口译练习评估（双音频/视频/文本 → AI 多维评估）")
        print("  [2] 单文件转写（音视频 → 文字）")
        print("  [q] 退出")
        choice = input("\n请输入: ").strip()

        if choice == 'q':
            print("👋 再见！")
            break

        elif choice == '1':
            files = get_audio_files()
            if len(files) < 2:
                print(f"\n❌ 至少需要 2 个媒体文件（源语 + 译语），当前只有 {len(files)} 个\n")
                continue

            # ── 第 1 步：提交源语文件 ──
            src_file = select_file(
                title="📤 第 1 步：提交【源语】",
                subtitle="你听到的原文（音视频自动转写 / 文本文件直接读取）",
                files=files,
            )
            if not src_file:
                continue
            print(f"  ✅ 源语已提交: {src_file}\n")

            # ── 第 2 步：提交译语文件 ──
            interp_file = select_file(
                title="📥 第 2 步：提交【译语】",
                subtitle="你的口译（音视频自动转写 / 文本文件直接读取）",
                files=files,
                exclude=src_file,
            )
            if not interp_file:
                continue
            print(f"  ✅ 译语已提交: {interp_file}\n")

            # ── 是否学术分析模式（毕业论文用）──
            print()
            print("  [0] 常规口译评估（日常练习使用）")
            print("  [1] 🎓 学术分析模式（上海外国语大学口译毕业论文专用）")
            mode_choice = input("  👉 请选择评估模式（回车默认 = 常规模式）: ").strip()
            academic_mode = (mode_choice == '1')

            # ── 提交确认 ──
            print_box(
                title="📋 提交确认",
                lines=[
                    f"源语: {src_file}",
                    f"译语: {interp_file}",
                    f"模式: {'🎓 学术分析（毕业论文）' if academic_mode else '📊 常规评估'}",
                    "",
                    "确认无误后按回车开始评估",
                ]
            )
            confirm = input("  👉 回车确认 / 输入 q 返回: ").strip()
            if confirm.lower() == 'q':
                continue

            # ── 处理中 ──
            print("\n" + "=" * 50)
            print("  🔄 处理中...")
            print("=" * 50)

            # ── 转写 + 修正 ──
            print("\n  ▶ 正在处理源语...")
            raw_src, src_lang, src_is_text = transcribe(src_file)
            if src_is_text:
                src_text = raw_src   # 文本直接使用，不修正
            else:
                src_text = polish_transcript(raw_src, src_lang, is_source=True)

            print("\n  ▶ 正在处理译语...")
            raw_interp, interp_lang, interp_is_text = transcribe(interp_file)
            if interp_is_text:
                interp_text = raw_interp   # 文本直接使用，不修正
            else:
                interp_text = polish_transcript(raw_interp, interp_lang, is_source=False, src_context=src_text)

            if not src_text or not interp_text:
                print("❌ 转写结果为空，请检查媒体文件\n")
                continue

            print(f"\n  🌐 检测到语言方向: {LANG_NAMES.get(src_lang, src_lang)} → {LANG_NAMES.get(interp_lang, interp_lang)}")

            # AI 评估
            print(f"\n  ▶ 正在进行 AI 口译评估（{'🎓 学术' if academic_mode else '常规'}模式）...")
            analysis = analyze(src_text, interp_text, src_lang, interp_lang, academic=academic_mode)

            # 输出
            print("\n  ▶ 正在生成 Obsidian 笔记...")
            output_path = generate_markdown(
                src_file, interp_file,
                raw_src, raw_interp,
                src_text, interp_text,
                src_lang, interp_lang,
                analysis,
                src_is_text=src_is_text,
                interp_is_text=interp_is_text
            )

            print()
            print_box(
                title="✅ 评估完成！",
                lines=[
                    f"📄 笔记已保存至:",
                    f"   {output_path}",
                    "",
                    "可在 Obsidian 中打开查看",
                ]
            )

        elif choice == '2':
            single_transcribe_mode()
            print()

        else:
            print("❌ 无效选项\n")


if __name__ == "__main__":
    main()
