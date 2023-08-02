# milvus

### **Docker Compose로 Milvus 독립 실행형 설치 [참고문서](https://milvus.io/docs/install_standalone-docker.md)**

밀버스 컨테이너를 따로 띄우지 않고 python에서 종속되어 작업할 수 있는 Milvus Lite가 있다. 프로덕션 환경이나 고성능이 필요한 경우 독립 실행형을 사용하는 게 좋다 [참고 문서](https://milvus.io/docs/milvus_lite.md)

1. docker-compose yml 파일 다운로드
    
    `$ wget https://github.com/milvus-io/milvus/releases/download/v2.2.12/milvus-standalone-docker-compose.yml -O docker-compose.yml`
    
    
    💡 `$ docker compose version`
    
      ![image](https://github.com/moon09I980616I/vecTextSearch/assets/95466895/ba642266-f2aa-4dc4-abc6-ceed452bc1d3)
     
      ![image](https://github.com/moon09I980616I/vecTextSearch/assets/95466895/083f2f1a-9275-452d-a98b-a44e49257f26)
      https://milvus.io/docs/v2.1.x
      
      docker compose 버전 확인 후 milvus 튜토리얼에서 버전 맞춰서 가이드 문서 참고
    
    
    
3. start milvus
    
    `$ docker-compose up -d`
   
    ![image](https://github.com/moon09I980616I/vecTextSearch/assets/95466895/08d9e2d5-9b28-4522-b641-335c309eef98)

    
5. connect milvus
Milvus 서버가 수신 대기 중인 로컬 포트를 확인
`$ docker port milvus-standalone 19530/tcp`
    
    ![image](https://github.com/moon09I980616I/vecTextSearch/assets/95466895/755f2169-c5cf-4702-baed-65c0738dcab2)

    
    이 명령으로 반환된 로컬 ip 주소와 포트 번호를 사용하여 milvus 클러스터에 연결
    

`$ docker port milvus-proxy 19530/tcp`



### milvus.bert용 도커 컨테이너 [참고문서](https://milvus.io/docs/manage_databases.md)

`$ docker run --gpus all -it -v /home/moon0/milvus.bert/:/home/milvus.bert -d --name milvus.bert moon0:1.0`

1. 데이터베이스 생성
    
    ```jsx
    from pymilvus import connections, db
    conn = connections.connect(host="192.168.2.104", port=19530)
    database = db.create_database("book")
    ```
    
2. 데이터베이스 사용
    
    ```jsx
    db.using_database("book")
    ```
    
3. 클러스터에 연결할 때 사용할 데이터베이스 설정
    
    ```jsx
    conn = connections.connect(
        host="192.168.2.104",
        port="19530",
        db_name="default"
    )
    ```
    
4. 데이터베이스 출력
   
    ```jsx
    print(db.list_database())
    ```
