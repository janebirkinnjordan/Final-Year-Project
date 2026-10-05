from typing import Literal
from pydantic import BaseModel, Field


ExperimentMode = Literal[
    'full',
    'pure_gpt',
    'no_history',
    'no_cross_domain',
]


class CreateSessionRequest(BaseModel):
    title: str = Field(default='New conversation', min_length=1, max_length=120)


class SessionResponse(BaseModel):
    id: str
    title: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    mode: ExperimentMode = 'full'


class MessageResponse(BaseModel):
    role: Literal['user', 'assistant']
    content: str
