# rag_answer_LLM.py
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from query_router import route_query
from entity_validator import validate_entity
from llm_client import generate_answer_cn
from llm_preprocess import preprocess_query_llm


# ===== 路径配置 =====
INDEX_PATH = Path("outputs/index/faiss.index")
META_PATH = Path("outputs/index/meta.jsonl")
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ===== 参数 =====
TOP_K_ENTITY = 6
REC_VEC_CANDIDATES = 140
REC_FINAL_ITEMS = 10
EVIDENCE_MAX_ITEMS = 8
EVIDENCE_MAX_CHARS_EACH = 1300


# ===== 基础 IO =====
def load_meta() -> List[Dict[str, Any]]:
    items = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


# ===== 检索 =====
def vector_search_with_scores(
    query: str,
    index,
    meta: List[Dict[str, Any]],
    model: SentenceTransformer,
    topk: int
) -> List[Dict[str, Any]]:
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, idxs = index.search(q_emb, topk)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        it = meta[int(idx)]
        results.append({"score": float(score), "item": it})
    return results


def keyword_filter(
    keyword: str,
    meta: List[Dict[str, Any]],
    types: Optional[set] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    hits = []
    for it in meta:
        name = (it.get("metadata") or {}).get("name", "") or ""
        text = it.get("text", "") or ""
        if keyword and (keyword == name or (keyword in name) or (keyword in text)):
            if types is None or it.get("type") in types:
                hits.append(it)
                if len(hits) >= limit:
                    break
    return hits


# ===== 安全兜底 =====
def detect_emergency(q: str) -> Optional[str]:
    s = (q or "").strip()

    if ("怀孕" in s or "孕期" in s) and (("出血" in s) or ("腹痛" in s) or ("阴道流血" in s)):
        return "孕期出现出血或腹痛属于高风险情况，请立即就医（妇产科/急诊）。不建议自行用药。"

    if ("胸痛" in s or "胸口痛" in s) and ("呼吸困难" in s or "喘" in s or "气短" in s):
        return "胸痛伴呼吸困难可能是急症，请立即就医或呼叫急救。不建议自行用药。"

    if ("小孩" in s or "婴儿" in s or "儿童" in s) and (("抽搐" in s) or ("惊厥" in s) or ("高烧" in s) or ("高热" in s)):
        return "儿童高热/抽搐/惊厥属于急症风险，请立即就医（急诊/儿科）。不建议自行用药。"

    return None


# ===== 推荐类辅助 =====
def compress_text(text: str) -> str:
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    keep = []
    key_fields = ("功效", "主治", "适应证", "适应症", "性：", "味：", "归经", "别名", "来源类型", "拉丁名", "拼音", "出处")
    for ln in lines:
        if any(k in ln for k in key_fields):
            keep.append(ln)
    if not keep:
        keep = lines[:12]
    return "\n".join(keep[:14])


def score_source_type(text: str) -> float:
    if not text:
        return 0.0
    if "来源类型：Plant medicine" in text:
        return 0.2
    if "Animal medicine" in text:
        return -0.15
    if "Mineral medicine" in text:
        return -0.15
    return 0.0


def has_key_fields(text: str) -> bool:
    if not text:
        return False
    return any(x in text for x in ["功效", "主治", "适应证", "适应症", "出处"])


def rank_recommendation_candidates(
    needs: List[str],
    candidate_types: List[str],
    vec_results: List[Dict[str, Any]]
) -> List[Tuple[float, Dict[str, Any]]]:
    ranked = []

    allowed_types = set(candidate_types) if candidate_types else {"herb"}

    for r in vec_results:
        base = r["score"]
        it = r["item"]
        typ = it.get("type")

        if typ not in allowed_types:
            continue

        text = it.get("text", "") or ""
        name = ((it.get("metadata") or {}).get("name")) or ""

        bonus = 0.0
        hit = 0

        for t in needs:
            if not t:
                continue
            if t in name:
                hit += 2
            if t in text:
                hit += 1
            if re.search(rf"(功效|主治|适应证|适应症)[:：].*{re.escape(t)}", text):
                hit += 2

        bonus += min(0.4, hit * 0.05)
        bonus += score_source_type(text)

        if not has_key_fields(text):
            bonus -= 0.18

        # 让 prescription 在症状推荐里略有优势
        if typ == "prescription":
            bonus += 0.08

        ranked.append((base + bonus, it))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked


def to_evidence(items: List[Dict[str, Any]], max_items: int, max_chars_each: int) -> List[Dict[str, str]]:
    ev = []
    for it in items[:max_items]:
        ev.append({
            "id": it.get("id"),
            "type": it.get("type"),
            "text": (it.get("text") or "")[:max_chars_each],
        })
    return ev


# ===== 主流程 =====
def main():
    print("Loading index & model...")
    index = faiss.read_index(str(INDEX_PATH))
    meta = load_meta()
    model = SentenceTransformer(EMBED_MODEL)

    while True:
        q = input("\n请输入问题（回车退出）：").strip()
        if not q:
            break

        emergency_msg = detect_emergency(q)
        if emergency_msg:
            print("\n=== 安全提示 ===")
            print(emergency_msg)
            continue

        # ===== Step 1: 先用 LLM 解析问题 =====
        parsed = preprocess_query_llm(q)

        intent = parsed.get("intent", "UNKNOWN")
        entity = parsed.get("entity")
        candidate_types = parsed.get("candidate_types", ["herb"])
        symptoms = parsed.get("symptoms", [])
        needs = parsed.get("needs", [])
        query_rewrite = parsed.get("query_rewrite", q)

        # ===== Step 2: 若 LLM 解析失败，再 fallback 到原规则 =====
        if intent == "UNKNOWN":
            fallback = route_query(q)
            intent = fallback["intent"]
            entity = fallback["entity"]
            candidate_types = ["herb"] if intent != "PRESCRIPTION_DEF" else ["prescription"]
            symptoms = []
            needs = []
            query_rewrite = q

        print(f"[INTENT] {intent}")
        print(f"[ENTITY] {entity}")

        if intent == "RECOMMENDATION":
            print(f"[RECOMMEND] needs={needs}, symptoms={symptoms}, candidate_types={candidate_types}")

        # ===== 实体校验 =====
        if intent in ("HERB_ATTRIBUTE", "HERB_MECHANISM", "PRESCRIPTION_DEF"):
            if not entity or not validate_entity(entity, intent):
                print("⚠️ 当前知识库中未收录该实体，无法给出可靠回答。")
                continue

        evidence_items: List[Dict[str, Any]] = []

        # ===== HERB_ATTRIBUTE =====
        if intent == "HERB_ATTRIBUTE":
            herb_hits = keyword_filter(entity, meta, types={"herb"}, limit=2)
            mech_hits = keyword_filter(entity, meta, types={"mechanism"}, limit=1)

            if not herb_hits:
                vec = vector_search_with_scores(f"{entity} 功效 主治", index, meta, model, topk=TOP_K_ENTITY)
                herb_hits = [x["item"] for x in vec if x["item"].get("type") == "herb"][:1]

            evidence_items = herb_hits + mech_hits

        # ===== HERB_MECHANISM =====
        elif intent == "HERB_MECHANISM":
            mech_hits = keyword_filter(entity, meta, types={"mechanism"}, limit=2)
            if not mech_hits:
                herb_hits = keyword_filter(entity, meta, types={"herb"}, limit=1)
                evidence_items = herb_hits
            else:
                evidence_items = mech_hits

        # ===== PRESCRIPTION_DEF =====
        elif intent == "PRESCRIPTION_DEF":
            pres_hits = keyword_filter(entity, meta, types={"prescription"}, limit=2)
            evidence_items = pres_hits

        # ===== RECOMMENDATION =====
        elif intent == "RECOMMENDATION":
            vec_query = query_rewrite or q
            vec_results = vector_search_with_scores(vec_query, index, meta, model, topk=REC_VEC_CANDIDATES)

            ranked = rank_recommendation_candidates(needs, candidate_types, vec_results)

            final_items = []
            seen_ids = set()
            for _, it in ranked:
                iid = it.get("id")
                if iid in seen_ids:
                    continue
                seen_ids.add(iid)

                it2 = dict(it)
                it2["text"] = compress_text(it.get("text", "") or "")
                final_items.append(it2)

                if len(final_items) >= REC_FINAL_ITEMS:
                    break

            evidence_items = final_items

        else:
            print("⚠️ 无法识别该问题类型。")
            continue

        if not evidence_items:
            print("⚠️ 未检索到可用证据。")
            continue

        evidence_blocks = to_evidence(
            evidence_items,
            max_items=min(EVIDENCE_MAX_ITEMS, len(evidence_items)),
            max_chars_each=EVIDENCE_MAX_CHARS_EACH
        )

        print("\n=== 回答（LLM生成，基于证据） ===")
        answer = generate_answer_cn(q, evidence_blocks)
        print(answer)


if __name__ == "__main__":
    main()