# Pik-NLP

관광지 리뷰 데이터를 기반으로 관광객의 관심사(카테고리)와 그 감성을 분석
1. Sentiment: 리뷰에서 지정된 속성마다 그 감성을 분석하는 모델
2. Category: 리뷰에서 지정된 대분류 마다, 그 속성들을 분류하는 모델

- ollama(LG의 exaone3.5)을 활용하여 학습 데이터를 생성
- BERT 기반 모델로 ACSC Task 수행

## Requirement
> [!warning]
> uv 설치가 필요합니다.
```bash
pip install uv
```

## 실행 방법

```bash
uv run main.py --help
```

## 학습 결과
|model|F1:pos|F1:neg|F1:none|accuracy|
|--|--|--|--|--|
|[KoBERT](https://github.com/SKTBrain/KoBERT)|0.83|0.53|0.94|0.90|
|[KoELECTRA-base-v3](https://github.com/monologg/KoELECTRA)|0.85|0.59|0.94|0.90|
|[KoELECTRA-small-v3](https://github.com/monologg/KoELECTRA)|0.81|0.53|0.92|0.88|
|[KcELECTRA-base-v2022](https://github.com/Beomi/KcELECTRA)|0.85|0.59|0.94|0.91|

> [!NOTE]
> 학습데이터가 적어서 클래스 불균형 존재
> 모든 결과는 동일 config에서 진행