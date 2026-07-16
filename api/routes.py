# api/routes.py

from fastapi import APIRouter

from api.schemas import (
    GenerateRequest,
    GenerateResponse,
)

from engine.engine import InferenceEngine


router = APIRouter()

engine = InferenceEngine()


@router.post(
    "/generate",
    response_model=GenerateResponse,
)
def generate(request: GenerateRequest):

    result = engine.generate(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
    )

    return result


@router.get("/health")
def health():

    return {
        "status": "healthy"
    }