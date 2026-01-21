import json
from pathlib import Path
import re

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("outputs/index")
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.jsonl"

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_meta(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


import re

def extract_entity(q: str) -> str:
    q = q.strip()

    # 1) X的...
    m = re.search(r"([\u4e00-\u9fff]{2,})的", q)
    if m:
        return m.group(1).strip()

    # 2) X主治什么 / X功效是什么 / X作用有哪些 / X适应证...
    # 关键：先匹配“完整关键词”，避免只吃到“主”
    keywords = [
        "主治", "功效", "作用", "适应证", "适应症", "禁忌", "用法", "用量", "归经", "性味"
    ]
    # 例如：老鹳草主治什么？ -> entity=老鹳草
    m = re.search(rf"([\u4e00-\u9fff]{{2,}})({'|'.join(keywords)})", q)
    if m:
        return m.group(1).strip()

    # 3) X治什么 / X能治什么 / X可以治什么（口语）
    m = re.search(r"([\u4e00-\u9fff]{2,})(?:能治|可以治|治)(?:什么|啥)", q)
    if m:
        return m.group(1).strip()

    # 4) 兜底：取中文片段，去掉问句词后选最长
    parts = re.findall(r"[\u4e00-\u9fff]{2,}", q)
    stop = {"功效", "作用", "主治", "适应证", "适应症", "是什么", "有哪些", "怎么", "如何", "请问", "介绍", "什么"}
    cleaned = []
    for p in parts:
        if p in stop:
            continue
        # 去掉尾部问句词
        for s in ["功效", "作用", "主治", "适应证", "适应症", "是什么", "有哪些", "什么"]:
            if p.endswith(s) and len(p) > len(s):
                p = p[: -len(s)]
        p = p.strip()
        if len(p) >= 2 and p not in stop:
            cleaned.append(p)

    return max(cleaned, key=len) if cleaned else ""

def keyword_hits(meta, keyword: str, topk: int = 5):
    if not keyword:
        return []
    hits = []
    for i, item in enumerate(meta):
        text = item.get("text", "")
        md = item.get("metadata", {}) or {}
        name = str(md.get("name", "") or "")
        # 命中规则：metadata.name 精确/包含 或 text 包含
        if keyword == name or (keyword and keyword in name) or (keyword and keyword in text):
            hits.append((i, item))
    # 简单排序：优先 name 精确命中，其次 name 包含，其次 text 命中
    def rank_key(pair):
        _, it = pair
        md = it.get("metadata", {}) or {}
        name = str(md.get("name", "") or "")
        text = it.get("text", "")
        if keyword == name:
            return (0, 0)
        if keyword in name:
            return (1, 0)
        if keyword in text:
            return (2, 0)
        return (3, 0)
    hits.sort(key=rank_key)
    return hits[:topk]


def main():
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing index: {FAISS_INDEX_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing meta: {META_PATH}")

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    meta = load_meta(META_PATH)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    while True:
        q = input("\n请输入问题（回车退出）：").strip()
        if not q:
            break
        entity = extract_entity(q)
        print(f"[DEBUG] extracted entity = {repr(entity)}")

        # 1) 先做实体/关键词命中（强约束）
        keyword = extract_entity(q)
        hits = keyword_hits(meta, keyword, topk=3)

        # 2) 再做向量检索（补充召回）
        q_emb = model.encode([q], normalize_embeddings=True).astype(np.float32)
        scores, idxs = index.search(q_emb, 5)

        # 合并（去重）：先输出命中，再输出向量结果
        seen = set()
        results = []

        for i, item in hits:
            seen.add(i)
            results.append(("KW", 1.0, i, item))  # KW命中给个标记分

        for i, s in zip(idxs[0], scores[0]):
            i = int(i)
            if i in seen:
                continue
            seen.add(i)
            item = meta[i]
            results.append(("VEC", float(s), i, item))

        # 只展示 Top-5
        results = results[:5]

        print("\nTop-5 检索结果：")
        for rank, (src, s, i, item) in enumerate(results, start=1):
            print(f"\n#{rank}  src={src}  score={s:.4f}  type={item['type']}  id={item['id']}")
            print(item["text"][:400])


if __name__ == "__main__":
    main()