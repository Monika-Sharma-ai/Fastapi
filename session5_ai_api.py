from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI, AzureOpenAI


load_dotenv()


app = FastAPI(
    title="Session 5 - AI API",
    description="FastAPI + OpenAI/Azure OpenAI example with summarization and chat endpoints",
    version="1.0.0",
)


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=20, max_length=12000)
    max_words: int = Field(default=120, ge=20, le=300)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    system_prompt: str = Field(
        default="You are a helpful assistant for students learning APIs.",
        max_length=500,
    )
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)


class AIResponse(BaseModel):
    provider: Literal["openai", "azure-openai"]
    model: str
    output: str


def _get_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip().strip('"').strip("'")
    return None


def _build_client() -> tuple[OpenAI | AzureOpenAI, str, str]:
    """Return (client, provider, model/deployment)."""

    # Standard OpenAI support
    openai_api_key = _get_env("OPENAI_API_KEY")
    openai_model = _get_env("OPENAI_MODEL", "OPENAI_MODEL_NAME")

    if openai_api_key and openai_model:
        return OpenAI(api_key=openai_api_key), "openai", openai_model

    # Azure OpenAI support (includes your existing variable names)
    azure_endpoint = _get_env(
        "AZURE_OPENAI_ENDPOINT",
        "OPENAI-ENDPOINT-RAAS",
        "NONPROD-EUS2-OPENAI-ENDPOINT",
    )
    azure_api_key = _get_env(
        "AZURE_OPENAI_API_KEY",
        "OPENAI-API-KEY-RAAS",
        "NONPROD-EUS2-OPENAI-API-KEY",
    )
    azure_deployment = _get_env(
        "AZURE_OPENAI_DEPLOYMENT",
        "OPENAI-MODEL-DEPLOYMENT-RAAS",
        "NONPROD-EUS2-OPENAI-MODEL-DEPLOYMENT",
    )
    azure_api_version = _get_env(
        "AZURE_OPENAI_API_VERSION",
        "OPENAI-MODEL-API-VERSION-CHAT-RAAS",
        "NONPROD-EUS2-OPENAI-MODEL-API-VERSION-CHAT",
        "NONPROD-EUS2-OPENAI-MODEL-API-VERSION",
    )

    if azure_endpoint and azure_api_key and azure_deployment and azure_api_version:
        client = AzureOpenAI(
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,
        )
        return client, "azure-openai", azure_deployment

    raise HTTPException(
        status_code=500,
        detail=(
            "Missing AI configuration. Set either OPENAI_API_KEY + OPENAI_MODEL, "
            "or Azure vars: endpoint, api key, deployment, and api version."
        ),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ai/summarize", response_model=AIResponse)
def summarize(req: SummarizeRequest) -> AIResponse:
    client, provider, model = _build_client()

    prompt = (
        f"Summarize the following text in <= {req.max_words} words. "
        "Return concise bullet points.\n\n"
        f"Text:\n{req.text}"
    )

    try:
        result = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a concise summarization assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        output = result.choices[0].message.content or ""
        return AIResponse(provider=provider, model=model, output=output.strip())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI provider error: {exc}")


@app.post("/ai/chat", response_model=AIResponse)
def chat(req: ChatRequest) -> AIResponse:
    client, provider, model = _build_client()

    try:
        result = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": req.system_prompt},
                {"role": "user", "content": req.message},
            ],
            temperature=req.temperature,
        )
        output = result.choices[0].message.content or ""
        return AIResponse(provider=provider, model=model, output=output.strip())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI provider error: {exc}")
