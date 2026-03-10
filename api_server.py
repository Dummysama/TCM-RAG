from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

import faiss
from jose import jwt
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from query_router import route_query
from entity_validator import validate_entity
from llm_client import generate_answer_cn
from llm_preprocess import preprocess_query_llm

from fastapi.staticfiles import StaticFiles

from fastapi import Depends, HTTPException, Header
from sqlalchemy.orm import Session


from database import Base, engine, get_db
from models import User
from schemas import RegisterRequest, LoginRequest, TokenResponse, UserInfo
from auth import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
    get_user_by_username, SECRET_KEY, ALGORITHM,
)

import json
from datetime import datetime

from models import User, Conversation, Message
from schemas import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    UserInfo,
    ConversationCreateRequest,
    ConversationItem,
    MessageItem,
    AskInConversationRequest,
)

INDEX_PATH = Path("outputs/index/faiss.index")
META_PATH = Path("outputs/index/meta.jsonl")
EMBED_MODEL = "models/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K_ENTITY = 6
REC_VEC_CANDIDATES = 140
REC_FINAL_ITEMS = 10
EVIDENCE_MAX_ITEMS = 8
EVIDENCE_MAX_CHARS_EACH = 1300

app = FastAPI(title="TCM-RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    theme: Optional[str] = None
    history: Optional[List[Dict[str, Any]]] = None


class AskResponse(BaseModel):
    answer: str
    intent: Optional[str] = None
    entity: Optional[str] = None
    references: List[str] = []
    title: Optional[str] = None
    reference_items: List[Dict[str, Any]] = []

class HerbDetailResponse(BaseModel):
    id: str
    name: Optional[str] = None
    pinyin: Optional[str] = None
    latin_name: Optional[str] = None
    english_name: Optional[str] = None
    nature: Optional[str] = None
    flavor: Optional[str] = None
    meridian: Optional[str] = None
    effects: Optional[str] = None
    indications: Optional[str] = None
    alias: Optional[str] = None
    raw_text: str = ""

meta_cache: List[Dict[str, Any]] = []
index_cache = None
model_cache = None


def load_meta() -> List[Dict[str, Any]]:
    items = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)
    global meta_cache, index_cache, model_cache
    print("Loading FAISS index, metadata, and embedding model...")
    meta_cache = load_meta()
    index_cache = faiss.read_index(str(INDEX_PATH))
    model_cache = SentenceTransformer(EMBED_MODEL)
    print("API server is ready.")


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


def detect_emergency(q: str) -> Optional[str]:
    s = (q or "").strip()

    if ("怀孕" in s or "孕期" in s) and (("出血" in s) or ("腹痛" in s) or ("阴道流血" in s)):
        return "孕期出现出血或腹痛属于高风险情况，请立即就医（妇产科/急诊）。不建议自行用药。"

    if ("胸痛" in s or "胸口痛" in s) and ("呼吸困难" in s or "喘" in s or "气短" in s):
        return "胸痛伴呼吸困难可能是急症，请立即就医或呼叫急救。不建议自行用药。"

    if ("小孩" in s or "婴儿" in s or "儿童" in s) and (("抽搐" in s) or ("惊厥" in s) or ("高烧" in s) or ("高热" in s)):
        return "儿童高热/抽搐/惊厥属于急症风险，请立即就医（急诊/儿科）。不建议自行用药。"

    return None


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


def generate_conversation_title(question: str, answer: str = "") -> str:
    q = (question or "").strip()

    prefixes = [
        "我最近", "最近", "请问", "想问", "我想问", "我想了解",
        "能不能", "可以", "有没有", "请教一下"
    ]
    for p in prefixes:
        if q.startswith(p):
            q = q[len(p):].strip()

    suffixes = [
        "是什么原因", "怎么办", "如何用药", "怎么调理", "怎么治疗",
        "吃什么药", "有什么中药推荐吗", "有什么推荐吗", "怎么回事"
    ]
    for s in suffixes:
        if q.endswith(s):
            q = q[: -len(s)].strip()

    q = q.replace("？", "").replace("?", "").replace("。", "").replace("，", " ").strip()

    if len(q) <= 12 and q:
        return q

    if len(q) > 12:
        return q[:12] + "…"

    return "中医药主题问答"

def build_reference_items(evidence_blocks: List[Dict[str, str]], meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    把 evidence id 转成更适合前端展示的 reference_items
    """
    meta_map = {it.get("id"): it for it in meta}
    result = []

    for e in evidence_blocks:
        iid = e.get("id")
        typ = e.get("type")
        item = meta_map.get(iid, {})
        md = item.get("metadata") or {}

        name = md.get("name")
        if not name:
            # 兜底：从文本里提取“中药名称”
            text = item.get("text", "") or ""
            m = re.search(r"中药名称[:：]\s*(.+)", text)
            if m:
                name = m.group(1).strip()

        result.append({
            "id": iid,
            "type": typ,
            "name": name or iid,
        })

    return result


def parse_herb_detail(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 herb chunk 的 text 中解析详情字段
    """
    text = item.get("text", "") or ""
    md = item.get("metadata") or {}

    def extract(label: str) -> Optional[str]:
        m = re.search(rf"{label}[:：]\s*(.+)", text)
        return m.group(1).strip() if m else None

    return {
        "id": item.get("id", ""),
        "name": md.get("name") or extract("中药名称"),
        "pinyin": extract("拼音"),
        "latin_name": extract("拉丁名"),
        "english_name": extract("英文名"),
        "nature": extract("性"),
        "flavor": extract("味"),
        "meridian": extract("归经"),
        "effects": extract("功效"),
        "indications": extract("主治/适应证") or extract("主治") or extract("适应证"),
        "alias": extract("别名"),
        "raw_text": text,
    }

def run_rag(question: str) -> Dict[str, Any]:
    emergency_msg = detect_emergency(question)
    if emergency_msg:
        return {
            "answer": emergency_msg,
            "intent": "SAFETY_BLOCK",
            "entity": None,
            "references": [],
            "title": generate_conversation_title(question),
            "reference_items": [],
        }

    parsed = preprocess_query_llm(question)

    intent = parsed.get("intent", "UNKNOWN")
    entity = parsed.get("entity")
    candidate_types = parsed.get("candidate_types", ["herb"])
    needs = parsed.get("needs", [])
    query_rewrite = parsed.get("query_rewrite", question)

    if intent == "UNKNOWN":
        fallback = route_query(question)
        intent = fallback["intent"]
        entity = fallback["entity"]
        candidate_types = ["herb"] if intent != "PRESCRIPTION_DEF" else ["prescription"]
        needs = []
        query_rewrite = question

    if intent in ("HERB_ATTRIBUTE", "HERB_MECHANISM", "PRESCRIPTION_DEF"):
        if not entity or not validate_entity(entity, intent):
            return {
                "answer": "当前知识库中未收录该实体，无法给出可靠回答。",
                "intent": intent,
                "entity": entity,
                "references": [],
                "title": generate_conversation_title(question),
                "reference_items": [],
            }

    evidence_items: List[Dict[str, Any]] = []

    if intent == "HERB_ATTRIBUTE":
        herb_hits = keyword_filter(entity, meta_cache, types={"herb"}, limit=2)
        mech_hits = keyword_filter(entity, meta_cache, types={"mechanism"}, limit=1)

        if not herb_hits:
            vec = vector_search_with_scores(f"{entity} 功效 主治", index_cache, meta_cache, model_cache, topk=TOP_K_ENTITY)
            herb_hits = [x["item"] for x in vec if x["item"].get("type") == "herb"][:1]

        evidence_items = herb_hits + mech_hits

    elif intent == "HERB_MECHANISM":
        mech_hits = keyword_filter(entity, meta_cache, types={"mechanism"}, limit=2)
        if not mech_hits:
            herb_hits = keyword_filter(entity, meta_cache, types={"herb"}, limit=1)
            evidence_items = herb_hits
        else:
            evidence_items = mech_hits

    elif intent == "PRESCRIPTION_DEF":
        pres_hits = keyword_filter(entity, meta_cache, types={"prescription"}, limit=2)
        evidence_items = pres_hits

    elif intent == "RECOMMENDATION":
        vec_query = query_rewrite or question
        vec_results = vector_search_with_scores(vec_query, index_cache, meta_cache, model_cache, topk=REC_VEC_CANDIDATES)

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
        return {
            "answer": "无法识别该问题类型。",
            "intent": intent,
            "entity": entity,
            "references": [],
            "title": generate_conversation_title(question),
            "reference_items": [],
        }

    if not evidence_items:
        return {
            "answer": "未检索到可用证据。",
            "intent": intent,
            "entity": entity,
            "references": [],
            "title": generate_conversation_title(question),
            "reference_items": [],
        }

    evidence_blocks = to_evidence(
        evidence_items,
        max_items=min(EVIDENCE_MAX_ITEMS, len(evidence_items)),
        max_chars_each=EVIDENCE_MAX_CHARS_EACH
    )

    answer = generate_answer_cn(question, evidence_blocks)
    refs = [x["id"] for x in evidence_blocks]
    reference_items = build_reference_items(evidence_blocks, meta_cache)

    return {
        "answer": answer,
        "intent": intent,
        "entity": entity,
        "references": refs,
        "title": generate_conversation_title(question, answer),
        "reference_items": reference_items,
    }

def get_current_user(
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    if not authorization:
        raise HTTPException(status_code=401, detail="未登录或令牌缺失")

    token = authorization.replace("Bearer ", "")

    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

    user_id = payload.get("sub")

    user = db.query(User).filter(User.id == int(user_id)).first()

    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")

    return user

@app.post("/api/register", response_model=UserInfo)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    username = payload.username.strip()

    if not username:
        raise HTTPException(status_code=400, detail="用户名不能为空")

    existing = get_user_by_username(db, username)
    if existing:
        raise HTTPException(status_code=400, detail="用户名已存在")

    user = User(
        username=username,
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return user

@app.post("/api/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    username = payload.username.strip()
    user = get_user_by_username(db, username)

    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    access_token = create_access_token(user.id, user.username)
    return TokenResponse(access_token=access_token)

@app.get("/api/me")
def get_me(user=Depends(get_current_user)):
    return {
        "id": user.id,
        "username": user.username
    }

#当前会话列表
@app.get("/api/conversations", response_model=list[ConversationItem])
def list_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(Conversation)
        .filter(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )
    return rows

#创建会话
@app.post("/api/conversations", response_model=ConversationItem)
def create_conversation_api(
    payload: ConversationCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    title = (payload.title or "新会话").strip() or "新会话"

    row = Conversation(
        user_id=current_user.id,
        title=title,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row

#获取某个会话的消息列表
@app.get("/api/conversations/{conversation_id}/messages", response_model=list[MessageItem])
def get_conversation_messages(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conv = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")

    rows = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    return rows

@app.post("/api/conversations/{conversation_id}/ask")
def ask_in_conversation(
    conversation_id: int,
    payload: AskInConversationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conv = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 调用你现有的 RAG 主流程
    result = run_rag(question)

    # 保存用户消息
    user_msg = Message(
        conversation_id=conversation_id,
        role="user",
        content=question,
        created_at=datetime.utcnow(),
    )
    db.add(user_msg)

    # 保存助手消息
    assistant_msg = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=result.get("answer", ""),
        intent=result.get("intent"),
        entity=result.get("entity"),
        references_json=json.dumps(result.get("references", []), ensure_ascii=False),
        created_at=datetime.utcnow(),
    )
    db.add(assistant_msg)

    # 如果当前会话标题还是默认值，可以用结果里的 title 更新
    if conv.title in ("新会话", "默认主题") and result.get("title"):
        conv.title = result["title"]

    conv.updated_at = datetime.utcnow()

    db.commit()

    return result


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    result = run_rag(req.question)
    return AskResponse(**result)

@app.get("/api/herb/{herb_id}", response_model=HerbDetailResponse)
def get_herb_detail(herb_id: str):
    full_id = herb_id if herb_id.startswith("herb::") else f"herb::{herb_id}"

    for it in meta_cache:
        if it.get("id") == full_id and it.get("type") == "herb":
            detail = parse_herb_detail(it)
            return HerbDetailResponse(**detail)

    return HerbDetailResponse(
        id=herb_id,
        name="未找到该中药",
        raw_text=""
    )

app.mount("/", StaticFiles(directory="tcm-rag-frontend/dist", html=True), name="frontend")