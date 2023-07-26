# 벡터데이터베이스
벡터를 고차원 포인트로 저장하는 데이터 베이스
일반적으로 k-NN(k-Nearest Neighbor) 인덱스로 구동되며 계층적 탐색 가능한 소규모 세계(HNSW) 및 반전된 파일 인덱스(IVF) 알고리즘과 같은 알고리즘으로 구축
데이터 관리, 내결함성, 인증 및 액세스 제어, 쿼리 엔진과 같은 추가 기능 제공

일반적으로 시각적, 의미 체계, 다중 모달 검색과 같은 벡터 검색 사용 사례를 강화하는 데 사용

# weaviate
벡터데이터베이스의 오픈소스 중 하나로 벡터 검색, 키워드와 벡터 검색을 결합한 하이브리드 검색 등의 기능이 있으며 weaviate에서 지원하는 임베딩모델을 사용하기에는 편리하지만 custom 모델을 사용하려면 go 언어를 사용하여 weaviate의 구성요소의 수정이 필요하지만 복잡함

## vecTextSearch
각각의 도커 환경에 weaviate와 distilbert 적재 후 distilbert를 이용해 벡터화 한 텍스트를 weaviate에 삽입

## 환경
![image](https://github.com/moon09I980616I/vecTextSearch/assets/95466895/a5d17daa-0edb-4920-82fe-cdb6465f18bd)
