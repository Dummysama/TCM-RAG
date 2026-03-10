from __future__ import annotations

import re
from openai import OpenAI
from deepseek_secrets import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


def clean_answer_text(text: str) -> str:
    """
    对模型输出做轻量清洗：
    1. 去掉 Markdown 痕迹
    2. 去掉模板化标题
    3. 把生硬的“无。”改成更自然的表达
    """
    if not text:
        return ""

    # 去掉 Markdown 粗体、标题、代码符号
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"`{1,3}", "", text)

    # 去掉常见模板化标题
    replacements = {
        "结论：": "",
        "简要结论：": "",
        "依据说明：": "",
        "现代研究：": "",
        "现代研究或靶点信息：": "",
        "现代机制：": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # 孤立的“无。”改得自然一点
    text = re.sub(r"(^|\n)无。?(\n|$)", r"\1现有证据中暂无相关现代研究信息。\2", text)

    # 压缩多余空行
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text


def generate_answer_cn(question: str, evidence_blocks: list[dict]) -> str:
    """
    使用 DeepSeek（OpenAI 兼容）chat.completions
    严格基于 evidence 回答，中文输出
    """
    if not DEEPSEEK_API_KEY or "粘贴" in DEEPSEEK_API_KEY:
        raise RuntimeError("deepseek_secrets.py 中的 DEEPSEEK_API_KEY 还未填写真实 Key")

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )

    evidence_text = "\n\n".join(
        f"[{e['id']}] {e['text']}" for e in evidence_blocks
    )

    system_prompt = (
        "你是一个中医药知识库问答助手。"
        "你必须严格基于给定证据回答，不得引入任何证据以外的事实。"
        "如果证据不足，请明确说明“现有证据不足以支持明确回答”或“现有证据不足以支持明确推荐”。"
        "回答语言为中文，语气专业、自然、简洁。"
        "不要使用 Markdown 标记，不要输出 **、##、-、*、``` 等格式符号。"
        "不要使用模板化小标题，例如“结论”“依据说明”“现代研究”等，除非用户明确要求。"
        "回答要像真实产品的说明文字，而不是大模型生成模板。"
        "若证据中没有现代研究或靶点信息，可自然说明“现有证据中暂无相关现代研究信息”。"
        "回答末尾单独一行列出引用，格式必须为：引用：id1,id2,id3"
    )

    user_prompt = f"""问题：
{question}

证据：
{evidence_text}

请直接给出自然、完整、便于普通用户理解的回答。
要求：
1. 严格基于证据，不得编造
2. 优先回答用户真正关心的结论
3. 如是推荐类问题，说明推荐理由
4. 不要分条，不要模板化标题
5. 最后一行必须保留引用，格式：引用：id1,id2
"""

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        temperature=0.2,
    )

    content = response.choices[0].message.content or ""
    return clean_answer_text(content)