# query_router.py
import re
from typing import Dict, Optional

# 问句“填充词/尾巴”，用于实体裁剪（非常关键）
TAIL_WORDS = [
    "有哪些", "有什么", "是什么", "是啥", "推荐", "介绍", "可以", "能", "是否", "能否", "可否"
]

# 将“意图关键词”分组：用来判断 intent
INTENT_KEYWORDS = {
    "HERB_MECHANISM": ["成分", "靶点", "机制", "药理"],
    "HERB_ATTRIBUTE": ["功效", "作用", "主治", "适应证", "适应症", "性味", "归经", "禁忌", "用法", "用量"],
    "PRESCRIPTION_DEF": ["是什么方", "是什么", "出自", "来源", "组成", "方义", "功效", "主治"],
    "RECOMMENDATION": ["推荐", "有哪些", "什么药", "什么中药", "药材推荐"],
}

# —— 核心：统一的实体规范化（后处理）——
def normalize_entity(ent: Optional[str]) -> Optional[str]:
    if not ent:
        return None

    ent = str(ent).strip()

    # 去掉常见不可见空白 & 全角空格
    ent = ent.replace("\u3000", "").replace("\xa0", "").replace(" ", "")

    # 去掉末尾标点
    ent = ent.strip("？?。.,，!！；;：:")

    # 剪掉问句尾巴（可重复剪，避免“有哪些是什么”这类）
    changed = True
    while changed:
        changed = False
        for tail in TAIL_WORDS:
            if ent.endswith(tail) and len(ent) > len(tail):
                ent = ent[: -len(tail)].strip()
                changed = True

    return ent or None


def extract_entity(q: str) -> Optional[str]:
    """尽量抽出实体；允许抽得“偏脏”，再交给 normalize_entity 清洗。"""
    q = q.strip()

    # 1) X的...
    m = re.search(r"([\u4e00-\u9fff]{2,})的", q)
    if m:
        return m.group(1).strip()

    # 2) 实体 + (可选填充词) + 关键意图词
    # 例如：地耳草有哪些靶点 -> entity=地耳草（但就算变脏，normalize 也能修）
    fillers = r"(?:有哪些|有什么|能|可以|是否|主要)?"
    intent_words = r"(功效|作用|主治|适应证|适应症|成分|靶点|机制|药理|是什么方|是什么|出自|来源|组成)"
    m = re.search(rf"([\u4e00-\u9fff]{{2,}}){fillers}{intent_words}", q)
    if m:
        return m.group(1).strip()

    # 3) 兜底：取开头一段连续中文
    m = re.match(r"^([\u4e00-\u9fff]{2,})", q)
    if m:
        return m.group(1).strip()

    return None


def route_query(q: str) -> Dict:
    q = q.strip()

    # 优先判断是否是“推荐/泛化”问题（通常无明确实体）
    # 例如：解毒消肿的药材推荐
    if ("推荐" in q) and ("的" in q) and not re.search(r"([\u4e00-\u9fff]{2,})的", q):
        return {"intent": "RECOMMENDATION", "entity": None}

    # 识别 intent（按关键词命中）
    # 注意：PRESCRIPTION_DEF 的关键词有可能与 attribute 重叠，但后续会通过实体校验分开
    intent = None
    for k, kws in INTENT_KEYWORDS.items():
        if any(kw in q for kw in kws):
            intent = k
            break

    if intent is None:
        # 没识别到明确 intent，但包含“推荐/有哪些/什么药”等，按推荐处理
        if any(x in q for x in INTENT_KEYWORDS["RECOMMENDATION"]):
            return {"intent": "RECOMMENDATION", "entity": None}
        return {"intent": "UNKNOWN", "entity": None}

    raw = extract_entity(q)
    ent = normalize_entity(raw)
    if intent == "RECOMMENDATION":
        return {"intent": "RECOMMENDATION", "entity": None}
    return {"intent": intent, "entity": ent}