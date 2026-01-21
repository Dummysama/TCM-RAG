# llm_client.py
from __future__ import annotations

from openai import OpenAI
from deepseek_secrets import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


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
        "你必须严格基于给定【证据】回答，不得引入任何证据以外的事实。"
        "如果证据不足，请明确说明“证据不足”。"
        "回答语言为中文，语气专业、简洁。"
        "回答末尾请列出引用的证据ID。"
    )

    user_prompt = f"""【问题】
{question}

【证据】
{evidence_text}

【回答要求】
1. 先给出简要结论（1–3 句）
2. 再给出依据说明（基于证据改写，不得编造）
3. 如有现代研究或靶点信息，单独成段
4. 最后一行格式：引用：id1,id2
"""

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
    )

    return response.choices[0].message.content