from pymongo import MongoClient
from dotenv import load_dotenv
import os
import csv
from datetime import datetime
import re
import emoji
from urllib.parse import urlparse
from bs4 import BeautifulSoup
load_dotenv()


client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB")]
reviews_collection = db["reviews"]

def clean_review_text(text):
    """
    리뷰 텍스트를 정제합니다.
    
    Args:
        text (str): 원본 리뷰 텍스트
    
    Returns:
        str: 정제된 리뷰 텍스트
    """
    # HTML 태그 제거
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    
    # URL 제거
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # 이모지 제거
    text = emoji.replace_emoji(text, '')
    
    # 따옴표 제거
    text = re.sub(r"[\"'`‘’“”]", "", text)
    
    # 특수문자 중 일부만 유지 (., !, ?, ,, :, ;, -, _)
    # 나머지 특수문자는 공백으로 대체
    text = re.sub(r'[^\w\s.,!?:;-~]', ' ', text)
    
    # 연속된 공백을 하나로 통일
    text = re.sub(r'\s+', ' ', text)
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text

def get_review_texts(min_length=10, max_reviews=None):
    """
    MongoDB의 reviews 컬렉션에서 지정된 길이 이상인 리뷰의 텍스트만 추출하여 리스트로 반환합니다.
    
    Args:
        min_length (int): 최소 리뷰 길이 (기본값: 10)
        max_reviews (int, optional): 가져올 최대 리뷰 개수 (기본값: None, 모든 리뷰)
    """
    review_texts = []
    for review in reviews_collection.find({}, {"content": 1}):
        if "content" in review:
            cleaned_text = clean_review_text(review["content"])
            if len(cleaned_text) >= min_length:
                review_texts.append(cleaned_text)
                if max_reviews and len(review_texts) >= max_reviews:
                    break
    return review_texts

def ensure_directory_exists(file_path):
    """
    파일 경로의 디렉토리가 존재하는지 확인하고, 없으면 생성합니다.
    
    Args:
        file_path (str): 파일 경로
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def export_reviews_to_csv(output_path=None, min_length=10, max_reviews=None):
    """
    리뷰 텍스트를 CSV 파일로 내보냅니다.
    
    Args:
        output_path (str, optional): CSV 파일 저장 경로. 기본값은 'reviews_{timestamp}.csv'
        min_length (int): 최소 리뷰 길이 (기본값: 10)
        max_reviews (int, optional): 가져올 최대 리뷰 개수 (기본값: None, 모든 리뷰)
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f'reviews_{timestamp}.csv'
    
    # 디렉토리가 없으면 생성
    ensure_directory_exists(output_path)
    
    reviews = get_review_texts(min_length=min_length, max_reviews=max_reviews)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(['Review'])
        for review in reviews:
            writer.writerow([review])
    
    print(f"리뷰가 {output_path}에 저장되었습니다.")
    print(f"총 {len(reviews)}개의 리뷰가 저장되었습니다. (최소 길이: {min_length}자, 최대 개수: {max_reviews if max_reviews else '제한 없음'})")

if __name__ == "__main__":
    # 예시: 20자 이상인 리뷰 중 1000개만 저장
    export_reviews_to_csv(output_path="data/raw/reviews.csv", min_length=20, max_reviews=1000) 

