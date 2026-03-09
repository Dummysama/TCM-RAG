# llm_preprocess.py
from __future__ import annotations

import json
from typing import Dict, Any

from openai import OpenAI
from deepseek_secrets import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL


def _get_client() -> OpenAI:
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )


def preprocess_query_llm(question: str) -> Dict[str, Any]:
    """
    使用 DeepSeek 对用户问题做结构化解析。
    输出固定 JSON，供本地检索使用。
    """

    client = _get_client()

    system_prompt = """
你是一个中医药问答系统的“问题解析器”。
你的任务不是直接回答问题，而是把用户问题转换成结构化 JSON，供本地知识库检索使用。

你必须只输出一个 JSON 对象，不要输出任何解释、注释或 markdown。

输出格式：
{
  "intent": "HERB_ATTRIBUTE | HERB_MECHANISM | PRESCRIPTION_DEF | RECOMMENDATION | UNKNOWN",
  "entity": "如果用户明确提到中药/方剂名称，则填写；否则为 null",
  "candidate_types": ["herb", "prescription"] 中的一个或两个，
  "symptoms": ["症状1", "症状2"],
  "needs": ["功效/治法关键词1", "关键词2"],
  "query_rewrite": "用于本地检索的简洁查询串"
}

解析规则：
1. 如果用户是在问某味中药或某个方剂本身（功效、作用、主治、定义、出处），应优先识别 entity。
2. 如果用户是在描述症状并寻求推荐、如何用药、怎么办、吃什么药、可参考什么药材，intent = RECOMMENDATION。
3. 如果用户没有明确实体，不要编造 entity，填 null。
4. candidate_types：
   - 症状描述 / 如何用药 / 怎么办 / 推荐：优先 ["prescription", "herb"]
   - 问某味中药本身：["herb"]
   - 问方剂本身：["prescription"]
5. needs 必须尽量抽象成“中医功效 / 治法 / 检索友好关键词”，不要照抄整句口语。

常见映射示例（非常重要）：
- 遗精 -> 固精, 补肾, 涩精, 收敛固涩, 肾虚
- 阳痿 -> 补肾壮阳, 温肾助阳, 补肾, 壮阳, 温阳, 助阳, 益肾, 肾虚, 肾阳虚
- 早泄 -> 固精, 补肾, 涩精, 肾虚
- 肾虚 -> 补肾, 益肾, 温阳, 养阴
- 小便短赤, 尿痛, 尿频, 淋证 -> 利尿通淋, 清热通淋, 清热利湿, 利湿
- 小便不利, 浮肿, 水肿 -> 利水消肿, 利水, 祛湿, 消肿
- 咽喉肿痛, 发热, 口渴 -> 清热解毒, 清热, 利咽, 养阴生津
- 口腔溃疡, 口疮 -> 清热解毒, 清热, 解毒
- 湿疹, 渗出, 瘙痒 -> 清热燥湿, 祛湿, 止痒, 解毒
- 食欲差, 大便溏, 腹泻, 乏力 -> 健脾, 益气, 和中, 止泻, 消食, 化湿
- 胸痛, 呼吸困难 -> 紧急就医（这是高风险，不建议给药）
- 孕期出血, 孕期腹痛 -> 紧急就医（这是高风险，不建议给药）
- 小孩高烧, 抽搐, 惊厥 -> 紧急就医（这是高风险，不建议给药）

补充要求：
6. needs 里尽量不要出现“怎么办、如何用药、推荐、适合、相关、最近经常”等口语噪声。
7. query_rewrite 要尽量简洁，应由：
   - symptoms
   - needs
   - candidate_types 相关词（如 方剂 / 主治 / 中药 / 功效）
   组合而成。
8. 如果问题明显是急症或高风险情况，也不要直接输出“紧急就医”作为最终回答，而应：
   - intent 仍设置为 RECOMMENDATION
   - symptoms 正常提取
   - needs 尽量提炼
   - 但不要编造药物实体
   - query_rewrite 尽量围绕症状/治法构造
   （本地系统会再做安全拦截）
9. 如果实在无法判断，则输出：
   - intent = UNKNOWN
   - entity = null
   - candidate_types = ["herb"]
   - symptoms = []
   - needs = []
   - query_rewrite = 原问题的简洁版
"""

    user_prompt = f"用户问题：{question}"

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        stream=False,
    )

    text = resp.choices[0].message.content.strip()

    # 尽量从输出中截取 JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                data = None
        else:
            data = None

    if not data:
        return {
            "intent": "UNKNOWN",
            "entity": None,
            "candidate_types": ["herb"],
            "symptoms": [],
            "needs": [],
            "query_rewrite": question,
        }

    # ===== 结果清洗与兜底 =====
    intent = data.get("intent", "UNKNOWN")
    entity = data.get("entity", None)
    candidate_types = data.get("candidate_types", ["herb"])
    symptoms = data.get("symptoms", [])
    needs = data.get("needs", [])
    query_rewrite = data.get("query_rewrite", question)

    # 类型安全
    if not isinstance(candidate_types, list) or not candidate_types:
        candidate_types = ["herb"]
    candidate_types = [x for x in candidate_types if x in ("herb", "prescription")]
    if not candidate_types:
        candidate_types = ["herb"]

    if not isinstance(symptoms, list):
        symptoms = []
    if not isinstance(needs, list):
        needs = []

    # 去掉明显口语噪声
    noise_words = {
        "怎么办", "如何用药", "怎么用药", "吃什么药", "推荐", "相关", "适合",
        "最近经常", "最近", "经常", "可参考", "怎么治", "如何治疗"
    }

    cleaned_needs = []
    for x in needs:
        x = str(x).strip()
        if not x or x in noise_words:
            continue
        cleaned_needs.append(x)

    cleaned_symptoms = []
    for x in symptoms:
        x = str(x).strip()
        if not x:
            continue
        cleaned_symptoms.append(x)

    return {
        "intent": intent,
        "entity": entity if entity else None,
        "candidate_types": candidate_types,
        "symptoms": cleaned_symptoms,
        "needs": cleaned_needs,
        "query_rewrite": str(query_rewrite).strip() if query_rewrite else question,
    }