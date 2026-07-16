# app.py

from fastapi import FastAPI

from api.routes import router


app = FastAPI(
    title="Mini vLLM",
    version="0.1",
)

app.include_router(router)