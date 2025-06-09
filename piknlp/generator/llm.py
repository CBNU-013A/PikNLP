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

def generate_category_prompt(review: str, categories: dict) -> list[dict]:
    system_msg = {
        "role": "system",
        "content": (
            "당신은 관광지 리뷰에 대해 주어진 계층적 카테고리들 중 해당하는 항목을 분류하는 어시스턴트입니다.\n"
            "다음 규칙을 따르세요:\n"
            "1. 먼저 메인 카테고리(장소, 활동, 동반, 계절)를 분류합니다.\n"
            "1.1. \"장소\" 카테고리는 해당 관광지가 리뷰에서 자연경관, 역사문화 등의 어떠한 테마에 속하는지 분류합니다.\n"
            "1.2. \"활동\" 카테고리는 해당 관광지에서 할 수 있는 활동을 분류합니다.\n"
            "1.3. \"동반\" 카테고리는 해당 리뷰에서 작성자가 누구와 함께 다녀왔는지를 분류합니다. 딱히 언급되지 않는 경우 'none'으로 작성합니다.\n"
            "1.4. \"계절\" 카테고리는 해당 리뷰에서 작성자가 어떠한 계절에 다녀왔는지, 어떠한 계절에 가는것을 추천하는지를 분류합니다. 특별한 언급이 없는 경우 'none'으로 작성합니다.\n"
            "2. 카테고리가 딱히 언급되지 않거나, 모호한 경우 반드시 영어로 'none'으로 작성합니다.\n"
            "3. 출력은 반드시 다음 JSON 형식을 정확히 따라야 합니다:\n"
            "{\n"
            "  \"categories\": {\n"
            "    \"장소\": \"자연/경관\",\n"
            "    \"활동\": \"산책/등산/트레킹\",\n"
            "    \"동반\": \"혼자\",\n"
            "    \"계절\": \"none\"\n"
            "  }\n"
            "}\n"
            "4. 절대 부연 설명 없이 결과만 출력할 것\n"
            "5. 반드시 'categories' 키의 내용만을 사용해야 함\n"
        )
    }

    user_msg = {
        "role": "user",
        "content": (
            "주어진 모든 카테고리에 대해 누락하지 말고 분류하세요. 절대 다른 설명 금지\n"
            f"리뷰: {review}\n"
            f"카테고리 구조: {json.dumps(categories, ensure_ascii=False)}"
        )
    }

    return [system_msg, user_msg]