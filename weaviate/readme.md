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

## 0)BERT 도커 환경 구축
CUDA 버전 확인

```bash
nvidia-smi
```

![image](https://github.com/moon09I980616I/vecTextSearch/assets/95466895/3378026a-4d94-4faa-b8a9-2c82e1d6a1c2)


cuda12.0 버전에 맞춰서 도커 이미지파일 다운로드 

[cuda12.0 docker image file 링크](https://hub.docker.com/r/abishekdatabricks/cuda12custom/tags)

```graphql
$ docker run --gpus all -it -v /[경로] androsovm/cuda12:latest /bin/bash
```

`$ docker run --gpus all -it -v /home/moon0/bert2:/home/bert -d -p 44343:4434/tcp --name bert_container2 embedding:latest`

![image](https://github.com/moon09I980616I/vecTextSearch/assets/95466895/a065959c-13f3-4c9e-99ed-7719d6d404e5)
[https://velog.io/@johyonghoon/docker-Error-response-from-daemon-could-not-select-device-driver-with-capabilities-gpu-해결](https://velog.io/@johyonghoon/docker-Error-response-from-daemon-could-not-select-device-driver-with-capabilities-gpu-%ED%95%B4%EA%B2%B0)

💡이러한 에러 메시지는 NVIDIA GPU를 사용하는 도커 컨테이너를 실행하기 위해 필요한 GPU 드라이버가 호스트 시스템에서 올바르게 설정되지 않았거나 사용 가능한 드라이버가 없을 때 발생할 수 있습니다.

- toolkit 재설치

https://wolfzone.tistory.com/31
https://jjeongil.tistory.com/1274
https://discuss.pytorch.org/t/install-pytorch-with-cuda-12-1/174294/17

```bash
apt install nvidia-cuda-toolkit
apt install python3-pip
pip3 install tensorflow
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip3 install transformers
pip3 install pandas
```

## 1) Weaviate 도커 환경 구축
https://weaviate.io/developers/weaviate/installation

1. Docker Compose 설치
    1. Docker-compose 설치 확인
        
        ```bash
        docker-compose --version
        ```
        
    2. Docker-compose 바이너리 다운로드
        
        ```bash
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        ```
        
    3. 실행 권한 부여
        
        ```bash
        sudo chmod +x /usr/local/bin/docker-compose
        ```
        
2. PATH에 Docker Compose 추가

```bash
nano ~/.bashrc
```

.bashrc에 다음 줄 추가

```bash
export PATH="/usr/local/bin:$PATH"
```

1. Docker 이미지 파일 다운로드

```bash
docker pull semitechnologies/weaviate
```

1. docker-compose.yml 생성
- basic
    
    ```bash
    version: '3.4'
    services:
      weaviate:
        image: semitechnologies/weaviate:latest
        ports:
          - 8080:8080
        restart: on-failure:0
        environment:
          QUERY_DEFAULTS_LIMIT: 25
          AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
          PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
          DEFAULT_VECTORIZER_MODULE: 'none'
          CLUSTER_HOSTNAME: 'node1'
          WEAVIATE_AUTHENTICATION_ENABLED: 'false'
      contextionary:
        image: semitechnologies/weaviate:latest
        environment:
          ENABLE_CUDA: 0 #gpu 사용
          OCCURRENCE_WEIGHT_LINEAR_FACTOR: 0.75
          EXTENSIONS_STORAGE_MODE: weaviate
          EXTENSIONS_STORAGE_ORIGIN: 'http://localhost:8080'
          NEIGHBOR_OCCURRENCE_IGNORE_PERCENTILE: 5
    ```
    

```graphql
version: '3.4'
services:
  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - 8080:8080
    restart: on-failure:0
    volumes:
      - /var/weaviate:/home/moon0/weaviate
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      CLUSTER_HOSTNAME: 'node1'
      WEAVIATE_AUTHENTICATION_ENABLED: 'false'
      DEFAULT_VECTORIZER_MODULE: 'none'
      TRANSFORMERS_INFERENCE_API: 'http://localhost:44343'

contextionary:
    image: embedding:latest
    environment:
      OCCURRENCE_WEIGHT_LINEAR_FACTOR: 0.75
      EXTENSIONS_STORAGE_MODE: weaviate
      EXTENSIONS_STORAGE_ORIGIN: 'http://localhost:8080'
      NEIGHBOR_OCCURRENCE_IGNORE_PERCENTILE: 5
```

```bash
docker-compose up -d
```
