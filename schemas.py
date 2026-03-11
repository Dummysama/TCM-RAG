from datetime import datetime
from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserInfo(BaseModel):
    id: int
    username: str
    created_at: datetime

    class Config:
        from_attributes = True

from typing import Optional, List
from datetime import datetime


class ConversationCreateRequest(BaseModel):
    title: Optional[str] = "新会话"


class ConversationItem(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MessageItem(BaseModel):
    id: int
    role: str
    content: str
    intent: Optional[str] = None
    entity: Optional[str] = None
    references_json: Optional[str] = None
    reference_items_json: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class AskInConversationRequest(BaseModel):
    question: str