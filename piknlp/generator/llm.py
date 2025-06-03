# piknlp/sentiment/llm.py

import openai
import json
from ollama import chat
from typing import Any

def call_ollama(
        messages: list[dict], 
        model: str = "exaone3.5",
        temperature = 0.9,
        top_k = 50,
        top_p = 0.95,
        repetition_penalty = 1.1,
        num_predict = 256,
        format = None,
        output_format = None
        ) -> Any:
    responce = chat(
        model = model, 
        messages = messages,
        format = format,
        options={
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "num_predict": num_predict,
        }
    )
    content = responce.message.content
    if output_format:
        if output_format == str:
            return content
        return output_format.model_validate_json(content)
    return content

def call_openai(
        messages: list[dict],
        model: str = "gpt-3.5-turbo",
        temperature = 0.9,
        top_p = 0.95,
        max_tokens = 256,
        output_format = None
    ) -> Any:
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    content = response["choices"][0]["message"]["content"]
    if output_format:
        if output_format == str:
            return content
        return output_format.model_validate_json(content)
    return content

def generate_dataset_prompt(review: str, category_set: list[str]) -> list[dict]:
    system_msg = {
        "role": "system",
        "content": (
            "당신은 관광지 리뷰에 대해 주어진 카테고리들 중 해당하는 항목에 대해 감성을 분석하는 어시스턴트입니다.\n"
            "다음 규칙을 따르세요:\n"
            "1. 감성은 'pos', 'neg', 'none' 중 하나로 작성합니다.\n"
            "2. 감성이 없는 카테고리는 'none'으로 작성합니다.\n"
            "3. 출력은 JSON 형식으로 작성하세요.\n"
            "4. 절대 설명 없이 결과만 출력할 것\n\n"
            "출력 형식 예시:\n"
            "{\n"
            "  \"labels\": {\n"
            "    \"경관\": \"pos\",\n"
            "    \"음식\": \"neg\",\n"
            "    \"가격\": \"none\"\n"
            "  }\n"
            "}"
        )
    }

    user_msg = {
        "role": "user",
        "content": (
            "주어진 모든 카테고리에 대해 누락하지 말고 감성을 분석하세요. 절대 다른 설명 금지\n"
            f"리뷰: {review}\n"
            f"카테고리 목록: {', '.join(category_set)}"
        )
    }

    return [system_msg, user_msg]