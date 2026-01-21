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


# ===== 路径配置 =====
INDEX_PATH = Path("outputs/index/faiss.index")
META_PATH = Path("outputs/index/meta.jsonl")
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ===== 参数（优先调这里）=====
TOP_K_ENTITY = 6
REC_VEC_CANDIDATES = 140
REC_FINAL_HERBS = 10
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


# ===== 口语/UNKNOWN 兜底：从全库名字中找实体 =====
def build_entity_catalog(meta: List[Dict[str, Any]]) -> Dict[str, set]:
    herb_names = set()
    pres_names = set()
    for it in meta:
        t = it.get("type")
        name = ((it.get("metadata") or {}).get("name")) or ""
        name = str(name).strip()
        if not name:
            continue
        if t == "herb":
            herb_names.add(name)
        elif t == "prescription":
            pres_names.add(name)
    return {"herb": herb_names, "prescription": pres_names}


def infer_entity_from_query(q: str, name_set: set) -> Optional[str]:
    if not q or not name_set:
        return None
    q = q.strip()
    candidates = [name for name in name_set if name and (name in q)]
    if not candidates:
        return None
    candidates.sort(key=len, reverse=True)
    return candidates[0]


def is_colloquial_attribute_query(q: str) -> bool:
    if not q:
        return False
    triggers = [
        "介绍一下", "介绍", "了解一下", "了解", "科普一下",
        "是什么", "干什么用", "是干什么用的", "有什么用", "有啥用",
        "作用", "功效", "主治", "治什么", "适应证", "适应症"
    ]
    return any(x in q for x in triggers)


# ===== 安全兜底：急症不做推荐（防明显 bug）=====
def detect_emergency(q: str) -> Optional[str]:
    """
    命中这些情形 -> 直接输出就医建议，不给草药推荐
    """
    s = (q or "").strip()

    # 妊娠出血/腹痛：高风险
    if ("怀孕" in s or "孕期" in s) and (("出血" in s) or ("腹痛" in s) or ("阴道流血" in s)):
        return "孕期出现出血或腹痛属于高风险情况，请立即就医（妇产科/急诊）。不建议自行用药。"

    # 严重胸痛 + 呼吸困难：可能急症
    if ("胸痛" in s or "胸口痛" in s) and ("呼吸困难" in s or "喘" in s or "气短" in s):
        return "胸痛伴呼吸困难可能是急症，请立即就医或呼叫急救。不建议自行用药。"

    # 小儿抽搐/惊厥/高热：急症
    if ("小孩" in s or "婴儿" in s or "儿童" in s) and (("抽搐" in s) or ("惊厥" in s) or ("高烧" in s) or ("高热" in s)):
        return "儿童高热/抽搐/惊厥属于急症风险，请立即就医（急诊/儿科）。不建议自行用药。"

    return None


# ===== 推荐类：需求抽取、症状->功效映射、扩展、压缩 =====
STOP_PREFIX = ["想找", "想问", "想要", "想请问", "请问", "能否", "有没有", "可以", "帮我", "给我", "推荐", "相关"]
STOP_SUFFIX = ["相关", "方面", "一点", "一些", "的", "中药", "药材", "有哪些", "推荐", "可参考", "可以参考"]


def clean_need_phrase(need: str) -> str:
    """
    去掉“想找/想问/相关”等口头语噪声。
    """
    if not need:
        return need
    x = need.strip()
    for p in STOP_PREFIX:
        if x.startswith(p):
            x = x[len(p):].strip()
    for s in STOP_SUFFIX:
        if x.endswith(s):
            x = x[: -len(s)].strip()
    # 再去掉可能残留的“想/找/问”
    x = re.sub(r"^(想|找|问)+", "", x).strip()
    return x


def extract_recommend_need(q: str) -> str:
    """
    优先抽取功效短语；失败再回退到症状（供映射层处理）。
    """
    q = q.strip()

    # 1) 形如：想找【X】的中药推荐 / 想【X】的中药有哪些
    m = re.search(r"(?:想找|想问|想要|想请问|请问|有没有|能否|可以|帮我)?\s*([\u4e00-\u9fff]{2,12})\s*的(?:中药|药材)?(?:有哪些|推荐|可参考)?", q)
    if m:
        return clean_need_phrase(m.group(1))

    # 2) 形如：X的中药有哪些 / X药材推荐
    m = re.search(r"([\u4e00-\u9fff]{2,12})\s*(?:的)?\s*(?:中药|药材)\s*(?:有哪些|推荐|可参考)", q)
    if m:
        return clean_need_phrase(m.group(1))

    # 3) 形如：清热解毒的中药有哪些？（没有“想找”）
    m = re.search(r"([\u4e00-\u9fff]{2,12})\s*(?:的)?\s*(?:中药|药材)\s*有哪些", q)
    if m:
        return clean_need_phrase(m.group(1))

    # 兜底：先返回句首连续中文（可能是症状/需求）
    m = re.match(r"^([\u4e00-\u9fff]{2,12})", q)
    if m:
        return clean_need_phrase(m.group(1))

    return q


def derive_need_terms_from_symptoms(q: str) -> List[str]:
    """
    轻量“症状 -> 功效”映射层：
    输入是整句问题 q，输出是用于检索/加权的功效关键词列表。
    """
    s = q

    terms: List[str] = []

    # 泌尿：尿痛/短赤/淋证
    if any(x in s for x in ["尿痛", "尿急", "尿频", "尿涩", "小便短赤", "淋", "淋证"]):
        terms += ["利尿通淋", "清热通淋", "清热利湿", "利湿"]

    # 浮肿/小便不利
    if any(x in s for x in ["浮肿", "水肿", "小便不利", "尿少"]):
        terms += ["利水消肿", "利水", "消肿", "祛湿"]

    # 咽喉肿痛/发热/口渴
    if any(x in s for x in ["咽喉肿痛", "咽痛", "喉咙痛", "发热", "高热", "口渴"]):
        terms += ["清热解毒", "清热", "利咽", "养阴生津"]

    # 口腔溃疡
    if any(x in s for x in ["口腔溃疡", "口疮"]):
        terms += ["清热解毒", "清热", "解毒"]

    # 湿疹/渗出/瘙痒
    if any(x in s for x in ["湿疹", "渗出", "瘙痒", "皮肤瘙痒", "皮疹"]):
        terms += ["清热燥湿", "祛湿", "止痒", "解毒"]

    # 食欲差/便溏/腹泻：偏脾胃
    if any(x in s for x in ["食欲差", "食欲不振", "大便溏", "便溏", "腹泻"]):
        terms += ["健脾", "和中", "消食", "止泻", "化湿"]

    # 头晕/乏力：偏气虚（仅作检索提示，不做诊断）
    if any(x in s for x in ["头晕", "乏力", "倦怠"]):
        terms += ["益气", "补气", "健脾"]

    # 去重
    uniq = []
    seen = set()
    for t in terms:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def expand_need_terms(need: str, extra_terms: Optional[List[str]] = None) -> List[str]:
    need = clean_need_phrase((need or "").strip())

    terms: List[str] = []
    if need:
        terms.append(need)

    # 4字拆分（清热利湿 -> 清热/利湿）
    if len(need) == 4:
        terms.extend([need[:2], need[2:]])

    # 同义扩展
    SYN = {
        "清热": ["清热", "清热解毒", "清热燥湿", "清热凉血", "清热泻火"],
        "利湿": ["利湿", "祛湿", "燥湿", "渗湿", "化湿", "利水", "利水消肿"],
        "解毒": ["解毒", "清热解毒"],
        "消肿": ["消肿", "消肿止痛", "消肿散结", "利水消肿"],
        "利尿": ["利尿", "利水", "通淋", "利尿通淋", "清热通淋"],
        "通淋": ["通淋", "利尿通淋", "清热通淋"],
        "止痒": ["止痒", "收湿止痒"],
        "健脾": ["健脾", "补脾", "健脾益气", "和中"],
        "止泻": ["止泻", "涩肠止泻"],
    }

    for k, vs in SYN.items():
        if k in need:
            terms.extend(vs)

    # 融合症状映射的额外 terms
    if extra_terms:
        terms.extend(extra_terms)

    # 去重保持顺序
    uniq = []
    seen = set()
    for t in terms:
        t = t.strip()
        if not t or t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return uniq


def compress_herb_text_for_need(text: str, terms: List[str]) -> str:
    if not text:
        return ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    keep = []

    key_fields = ("功效", "主治", "适应证", "适应症", "性：", "味：", "归经", "别名", "来源类型", "拉丁名", "拼音")

    # 优先保留命中词行
    for ln in lines:
        if any(t in ln for t in terms if t):
            keep.append(ln)

    # 再保留关键字段行
    for ln in lines:
        if any(k in ln for k in key_fields):
            if ln not in keep:
                keep.append(ln)

    if len(keep) < 8:
        keep = (keep + lines[:10])[:14]
    else:
        keep = keep[:14]

    return "\n".join(keep)


def score_source_type(text: str) -> float:
    if not text:
        return 0.0
    if "来源类型：Plant medicine" in text:
        return 0.25
    if "来源类型：Animal medicine" in text:
        return -0.20
    if "来源类型：Mineral medicine" in text:
        return -0.20
    return 0.0


def has_effect_lines(text: str) -> bool:
    if not text:
        return False
    return ("功效" in text) or ("主治" in text) or ("适应证" in text) or ("适应症" in text)


def rank_recommendation_candidates_v2(
    terms: List[str],
    vec_results: List[Dict[str, Any]]
) -> List[Tuple[float, Dict[str, Any]]]:
    ranked = []
    for r in vec_results:
        base = r["score"]
        it = r["item"]
        if it.get("type") not in ("herb", "prescription"):
            continue
        text = it.get("text", "") or ""
        name = ((it.get("metadata") or {}).get("name")) or ""

        bonus = 0.0
        hit = 0
        for t in terms:
            if not t:
                continue
            if t in name:
                hit += 2
            if t in text:
                hit += 1
            if re.search(rf"功效[:：].*{re.escape(t)}", text):
                hit += 2
        bonus += min(0.35, hit * 0.04)
        bonus += score_source_type(text)

        if not has_effect_lines(text):
            bonus -= 0.18

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

    # 实体词表（UNKNOWN兜底、实体纠偏）
    catalog = build_entity_catalog(meta)
    herb_names = catalog["herb"]
    pres_names = catalog["prescription"]

    while True:
        q_raw = input("\n请输入问题（回车退出）：").strip()
        if not q_raw:
            break

        # 安全兜底：急症直接退出
        emergency_msg = detect_emergency(q_raw)
        if emergency_msg:
            print("\n=== 安全提示 ===")
            print(emergency_msg)
            continue

        q = q_raw
        info = route_query(q)
        intent = info["intent"]
        entity = info["entity"]

        # 兜底：症状口语问法 -> 当作 RECOMMENDATION
        if intent == "UNKNOWN":
            if any(x in q for x in ["怎么办", "怎么治", "如何治疗", "如何用药", "怎么用药", "吃什么药", "用什么药"]):
                intent = "RECOMMENDATION"
                entity = None
        # UNKNOWN + 口语属性问法 -> herb 属性
        if intent == "UNKNOWN" and is_colloquial_attribute_query(q):
            guessed = infer_entity_from_query(q, herb_names)
            if guessed:
                intent = "HERB_ATTRIBUTE"
                entity = guessed

        # 抽到脏实体 -> 纠偏为真实 herb 名
        if intent in ("HERB_ATTRIBUTE", "HERB_MECHANISM"):
            if entity and (not validate_entity(entity, intent)):
                guessed = infer_entity_from_query(q, herb_names)
                if guessed:
                    entity = guessed

        # 方剂兜底（可选）
        if intent == "UNKNOWN" and is_colloquial_attribute_query(q):
            guessed_p = infer_entity_from_query(q, pres_names)
            if guessed_p and any(x in guessed_p for x in ["汤", "散", "丸", "膏", "饮", "丹"]):
                intent = "PRESCRIPTION_DEF"
                entity = guessed_p

        print(f"[INTENT] {intent}")
        print(f"[ENTITY] {entity}")

        # 实体校验
        if intent in ("HERB_ATTRIBUTE", "HERB_MECHANISM", "PRESCRIPTION_DEF"):
            if not entity or not validate_entity(entity, intent):
                print("⚠️ 当前知识库中未收录该实体，无法给出可靠回答。")
                continue

        evidence_items: List[Dict[str, Any]] = []

        # HERB_ATTRIBUTE
        if intent == "HERB_ATTRIBUTE":
            herb_hits = keyword_filter(entity, meta, types={"herb"}, limit=2)
            mech_hits = keyword_filter(entity, meta, types={"mechanism"}, limit=1)

            if not herb_hits:
                vec = vector_search_with_scores(f"{entity} 功效 主治", index, meta, model, topk=TOP_K_ENTITY)
                herb_hits = [x["item"] for x in vec if x["item"].get("type") == "herb"][:1]

            evidence_items = herb_hits + mech_hits

        # HERB_MECHANISM
        elif intent == "HERB_MECHANISM":
            mech_hits = keyword_filter(entity, meta, types={"mechanism"}, limit=2)
            if not mech_hits:
                herb_hits = keyword_filter(entity, meta, types={"herb"}, limit=1)
                evidence_items = herb_hits
            else:
                evidence_items = mech_hits

        # PRESCRIPTION_DEF
        elif intent == "PRESCRIPTION_DEF":
            pres_hits = keyword_filter(entity, meta, types={"prescription"}, limit=2)
            evidence_items = pres_hits

        # RECOMMENDATION：改进后的症状->功效推荐
        elif intent == "RECOMMENDATION":
            need_raw = extract_recommend_need(q)
            extra = derive_need_terms_from_symptoms(q)  # 症状映射
            need = clean_need_phrase(need_raw)

            # 如果 need 太“空/泛”（比如“适合/相关”），就依赖 extra_terms
            GENERIC = {"适合", "相关", "推荐", "中药", "药材", "可参考", "有哪些", "作用"}
            if (not need) or (need in GENERIC) or (len(need) <= 1):
                need = extra[0] if extra else need_raw

            terms = expand_need_terms(need, extra_terms=extra)

            vec_query = f"{need} {' '.join(terms[:8])} 中药 功效 主治 推荐"
            vec_results = vector_search_with_scores(vec_query, index, meta, model, topk=REC_VEC_CANDIDATES)

            ranked = rank_recommendation_candidates_v2(terms, vec_results)

            final_herbs = []
            seen_ids = set()
            for _, it in ranked:
                iid = it.get("id")
                if iid in seen_ids:
                    continue
                seen_ids.add(iid)

                compressed = compress_herb_text_for_need(it.get("text", "") or "", terms)
                it2 = dict(it)
                it2["text"] = compressed

                final_herbs.append(it2)
                if len(final_herbs) >= REC_FINAL_HERBS:
                    break

            evidence_items = final_herbs

            print(f"[RECOMMEND] need='{need}', terms={terms[:10]}, candidates={len(vec_results)}, final={len(evidence_items)}")

            # 给 LLM 的约束提示更清晰：允许“部分匹配”，但必须引用证据
            q = (
                f"{q}\n\n"
                f"【推荐约束】请优先从证据中选择“功效/主治”包含以下关键词（完全或部分命中均可）的药材："
                f"{'、'.join(terms[:12])}。"
                f"若无法找到完全命中“{need}”的药材，可给出部分相关候选，并明确说明是部分匹配；不得脱离证据编造。"
            )

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