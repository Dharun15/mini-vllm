# api/schemas.py

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):

    prompt: str = Field(...)

    max_new_tokens: int = Field(
        default=128,
        ge=1,
        le=2048,
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
    )


class GenerateResponse(BaseModel):

    generated_text: str

    metrics: dict