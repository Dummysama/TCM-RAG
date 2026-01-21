# rag_answer.py
import json
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

from query_router import route_query
from entity_validator import validate_entity

# ===== 路径配置 =====
INDEX_PATH = Path("outputs/index/faiss.index")
META_PATH = Path("outputs/index/meta.jsonl")
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 5


# ===== 工具函数 =====
def load_meta():
    items = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def vector_search(query: str, index, meta, model, topk=TOP_K):
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, idxs = index.search(q_emb, topk)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        it = meta[idx]
        results.append({
            "score": float(score),
            "type": it.get("type"),
            "id": it.get("id"),
            "text": it.get("text"),
            "metadata": it.get("metadata"),
        })
    return results


def keyword_filter(entity: str, meta, types=None):
    hits = []
    for it in meta:
        name = (it.get("metadata") or {}).get("name", "")
        text = it.get("text", "")
        if entity and (entity in name or entity in text):
            if types is None or it.get("type") in types:
                hits.append(it)
    return hits


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

        info = route_query(q)
        intent = info["intent"]
        entity = info["entity"]

        print(f"[INTENT] {intent}")
        print(f"[ENTITY] {entity}")

        # ===== 实体校验 =====
        if intent in ("HERB_ATTRIBUTE", "HERB_MECHANISM", "PRESCRIPTION_DEF"):
            if not entity or not validate_entity(entity, intent):
                print("⚠️ 当前知识库中未收录该实体，无法给出可靠回答。")
                continue

        print("\n=== 回答 ===")

        # ===== HERB_ATTRIBUTE =====
        if intent == "HERB_ATTRIBUTE":
            herb_hits = keyword_filter(entity, meta, types={"herb"})
            mech_hits = keyword_filter(entity, meta, types={"mechanism"})

            if herb_hits:
                h = herb_hits[0]
                print(h["text"])
                print(f"\n[引用] {h['id']}")
            if mech_hits:
                m = mech_hits[0]
                print("\n【现代研究补充】")
                print(m["text"][:800] + "...")
                print(f"\n[引用] {m['id']}")

        # ===== HERB_MECHANISM =====
        elif intent == "HERB_MECHANISM":
            mech_hits = keyword_filter(entity, meta, types={"mechanism"})
            if mech_hits:
                m = mech_hits[0]
                print(m["text"])
                print(f"\n[引用] {m['id']}")
            else:
                print("⚠️ 未找到该中药的机制/成分信息。")

        # ===== PRESCRIPTION_DEF =====
        elif intent == "PRESCRIPTION_DEF":
            pres_hits = keyword_filter(entity, meta, types={"prescription"})
            if pres_hits:
                p = pres_hits[0]
                print(p["text"])
                print(f"\n[引用] {p['id']}")
            else:
                print("⚠️ 未找到该方剂的定义信息。")

        # ===== RECOMMENDATION =====
        elif intent == "RECOMMENDATION":
            print("【推荐结果】")
            results = vector_search(q, index, meta, model, topk=5)
            for i, r in enumerate(results, 1):
                if r["type"] != "herb":
                    continue
                name = (r["metadata"] or {}).get("name", "")
                print(f"\n#{i} {name}")
                print(r["text"][:300] + "...")
                print(f"[引用] {r['id']}")

        else:
            print("⚠️ 无法识别该问题类型。")


if __name__ == "__main__":
    main()