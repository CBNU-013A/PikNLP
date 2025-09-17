# piknlp/sentiment/llm.py

import openai
import json
from ollama import chat
from typing import Any

def call_ollama(
        messages: list[dict], 
        model: str = "exaone3.5",
        temperature = 0.2,
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
            "당신은 관광지 리뷰에 대해 해당 문장이 주어진 카테고리들중 각 항목에 어떠한 감성을 보이는지 분석하는 어시스턴트입니다.\n"
            "다음 규칙을 따르세요:\n"
            "- 출력은 JSON 형식으로 작성할 것\n"
            "- 주어진 카테고리에는 누락된 것이 없어야 함.\n"
            "- 절대 설명 없이 결과만 출력할 것\n\n"
        )
    }

    user_msg = {
        "role": "user",
        "content": (
            f"리뷰: {review}\n"
            f"카테고리 목록: {', '.join(category_set)}\n"
            "해당 문장을 주어진 모든 카테고리들에 대해 누락하지 말고 감성을 분석하세요. 절대 다른 설명 금지\n"
            "다음 규칙을 따를 것\n"
            "1. 분석은 'pos', 'neg', 'none' 중 하나로 작성\n"
            "2. 분석이 없는 카테고리는 'none'으로 작성\n"
            "3. 출력 형식:\n"
            "{\n"
            "  \"sentiments\": {\n"
            "    \"교통편\": \"pos\",\n"
            "    \"주차\": \"neg\",\n"
            "  }\n"
            "}"
        )
    }

    return [system_msg, user_msg]

def generate_dataset_batch_prompt(reviews: list[str], category_set: list[str]) -> list[dict]:
    system_msg = {
        "role": "system",
        "content": (
            "당신은 관광지 리뷰들에 대해 각 문장이 주어진 카테고리들중 각 항목에 어떠한 감성을 보이는지 분석하는 어시스턴트입니다.\n"
            "다음 규칙을 따르세요:\n"
            "- 출력은 JSON 형식으로 작성할 것\n"
            "- 주어진 카테고리에는 누락된 것이 없어야 함.\n"
            "- 각 리뷰에 대해 개별적으로 분석할 것\n"
            "- 절대 설명 없이 결과만 출력할 것\n\n"
        )
    }

    reviews_text = "\n".join([f"{i+1}. {review}" for i, review in enumerate(reviews)])
    
    user_msg = {
        "role": "user",
        "content": (
            f"리뷰들:\n{reviews_text}\n\n"
            f"카테고리 목록: {', '.join(category_set)}\n"
            "각 리뷰를 주어진 모든 카테고리들에 대해 누락하지 말고 감성을 분석하세요. 절대 다른 설명 금지\n"
            "다음 규칙을 따를 것\n"
            "1. 분석은 'pos', 'neg', 'none' 중 하나로 작성\n"
            "2. 분석이 없는 카테고리는 'none'으로 작성\n"
            "3. 출력 형식:\n"
            "{\n"
            "  \"results\": [\n"
            "    {\n"
            "      \"review\": \"리뷰 내용\",\n"
            "      \"sentiments\": {\n"
            "        \"교통편\": \"pos\",\n"
            "        \"주차\": \"neg\"\n"
            "      }\n"
            "    }\n"
            "  ]\n"
            "}"
        )
    }

    return [system_msg, user_msg]

def generate_category_batch_prompt(reviews: list[str], categories: dict) -> list[dict]:
    system_msg = {
        "role": "system",
        "content": (
            "당신은 관광지 리뷰들에 대해 주어진 계층적 카테고리들 중 해당하는 항목을 분류하는 어시스턴트입니다.\n"
            "다음 규칙을 따르세요:\n"
            "- 출력은 JSON 형식으로 작성할 것\n"
            "- 주어진 카테고리에는 누락된 것이 없어야 함.\n"
            "- 반드시 주어진 카테고리, 키워드들에 한하여서만 분류 할 것. 새로운 단어 생성 금지\n"
            "- 각 리뷰에 대해 개별적으로 분류할 것\n"
            "- 절대 설명 없이 결과만 출력할 것\n\n"
        )
    }

    reviews_text = "\n".join([f"{i+1}. {review}" for i, review in enumerate(reviews)])
    
    user_msg = {
        "role": "user",
        "content": (
            f"리뷰들:\n{reviews_text}\n\n"
            f"카테고리 구조: {json.dumps(categories, ensure_ascii=False)}\n"
            "각 리뷰를 주어진 모든 카테고리에 대해 누락하지 말고 분류하세요. 절대 다른 설명 금지\n\n"
            "다음 규칙을 따르세요:\n"
            "- 다음 관광지 리뷰 문장에서 언급된 메인 카테고리(장소, 활동, 동반, 시점)를 분류합니다.\n"
            "- 4개의 메인 카테고리 각각에 대하여 문장에서 주어진 키워드들중 하나로 분류합니다."
            "1.1. \"장소\" 카테고리는 해당 관광지가 리뷰에서 자연경관, 역사문화 등의 어떠한 테마에 속하는지 분류합니다.\n"
            "1.2. \"활동\" 카테고리는 해당 관광지에서 할 수 있는 활동을 분류합니다.\n"
            "1.2.1. 이때 \"탐방\" 카테고리는 산책, 트래킹, 등산 등의 활동을 의미합니다.\n"
            "1.2.2. \"관람\" 카테고리는 축제, 조망, 전시관람, 감상 등의 활동을 의미합니다.\n"
            "1.2.3. \"참여\" 카테고리는 문화체험, 레저활동 등의 참여 형태의 활동을 의미합니다.\n"
            "1.2.4. \"먹거리\" 카테고리는 음식, 카페 등 식음료 관련 활동을 의미합니다.\n"
            "1.2.5. \"쇼핑\" 카테고리는 쇼핑등의 소비 형태의 활동을 의미합니다.\n"
            "1.2.6. \"포토존\" 카테고리는 인생샷 등 특별한 사진촬영과 같은 활동을 의미합니다.\n"
            "1.3. \"동반\" 카테고리는 해당 관광지에 누구와 함께 다녀왔는지 분류합니다. 누구와 다녀왔는지 명확한 언급이 없는 경우 'none'으로 작성합니다.\n"
            "1.4. \"시점\" 카테고리는 해당 관광지에 작성자가 어떠한 시점에 다녀왔는지 분류합니다. 시점에 대한 명확한 언급이 없는 경우 'none'으로 작성합니다.\n"
            "2. 각 카테고리 중 주어진 키워드들에 해당 하는 내용이 언급되지 않거나, 분류가 명확하지 않은 경우 반드시 영어로 'none'으로 작성합니다.\n"
            "3. 출력은 반드시 다음 JSON 형식을 정확히 따라야 합니다:\n"
            "{\n"
            "  \"results\": [\n"
            "    {\n"
            "      \"review\": \"리뷰 내용\",\n"
            "      \"categories\": {\n"
            "        \"장소\": \"자연경관\",\n"
            "        \"활동\": \"탐방\",\n"
            "        \"동반\": \"혼자\",\n"
            "        \"시점\": \"none\"\n"
            "      }\n"
            "    }\n"
            "  ]\n"
            "}"
        )
    }

    return [system_msg, user_msg]

def generate_category_prompt(review: str, categories: dict) -> list[dict]:
    system_msg = {
        "role": "system",
        "content": (
            "당신은 관광지 리뷰에 대해 주어진 계층적 카테고리들 중 해당하는 항목을 분류하는 어시스턴트입니다.\n"
            "다음 규칙을 따르세요:\n"
            "- 출력은 JSON 형식으로 작성할 것\n"
            "- 주어진 카테고리에는 누락된 것이 없어야 함.\n"
            "- 반드시 주어진 카테고리, 키워드들에 한하여서만 분류 할 것. 새로운 단어 생성 금지\n"
            "- 절대 설명 없이 결과만 출력할 것\n\n"
        )
    }

    user_msg = {
        "role": "user",
        "content": (
            f"리뷰: {review}\n"
            f"카테고리 구조: {json.dumps(categories, ensure_ascii=False)}"
            "주어진 모든 카테고리에 대해 누락하지 말고 분류하세요. 절대 다른 설명 금지\n\n"
            "다음 규칙을 따르세요:\n"
            "- 다음 관광지 리뷰 문장에서 언급된 메인 카테고리(장소, 활동, 동반, 시점)를 분류합니다.\n"
            "- 4개의 메인 카테고리 각각에 대하여 문장에서 주어진 키워드들중 하나로 분류합니다."
            "1.1. \"장소\" 카테고리는 해당 관광지가 리뷰에서 자연경관, 역사문화 등의 어떠한 테마에 속하는지 분류합니다.\n"
            "1.2. \"활동\" 카테고리는 해당 관광지에서 할 수 있는 활동을 분류합니다.\n"
            "1.2.1. 이때 \"탐방\" 카테고리는 산책, 트래킹, 등산 등의 활동을 의미합니다.\n"
            "1.2.2. \"관람\" 카테고리는 축제, 조망, 전시관람, 감상 등의 활동을 의미합니다.\n"
            "1.2.3. \"참여\" 카테고리는 문화체험, 레저활동 등의 참여 형태의 활동을 의미합니다.\n"
            "1.2.4. \"먹거리\" 카테고리는 음식, 카페 등 식음료 관련 활동을 의미합니다.\n"
            "1.2.5. \"쇼핑\" 카테고리는 쇼핑등의 소비 형태의 활동을 의미합니다.\n"
            "1.2.6. \"포토존\" 카테고리는 인생샷 등 특별한 사진촬영과 같은 활동을 의미합니다.\n"
            "1.3. \"동반\" 카테고리는 해당 관광지에 누구와 함께 다녀왔는지 분류합니다. 누구와 다녀왔는지 명확한 언급이 없는 경우 'none'으로 작성합니다.\n"
            "1.4. \"시점\" 카테고리는 해당 관광지에 작성자가 어떠한 시점에 다녀왔는지 분류합니다. 시점에 대한 명확한 언급이 없는 경우 'none'으로 작성합니다.\n"
            "2. 각 카테고리 중 주어진 키워드들에 해당 하는 내용이 언급되지 않거나, 분류가 명확하지 않은 경우 반드시 영어로 'none'으로 작성합니다.\n"
            "3. 출력은 반드시 다음 JSON 형식을 정확히 따라야 합니다:\n"
            "{\n"
            "  \"categories\": {\n"
            "    \"장소\": \"자연경관\",\n"
            "    \"활동\": \"탐방\",\n"
            "    \"동반\": \"혼자\",\n"
            "    \"시점\": \"none\"\n"
            "  }\n"
            "}"
        )
    }

    return [system_msg, user_msg]