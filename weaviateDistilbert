from logging.config import dictConfig
import logging

dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(message)s',
        }
    },
    'handlers': {
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': 'debug.log',
            'formatter': 'default',
        },
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['file']
    }
})

import torch
import json
from transformers import BertTokenizer, BertModel
from nltk.tokenize import sent_tokenize
import weaviate
from weaviate.util import generate_uuid5
from weaviate import Client
import time

# initialize weaviate client for importing and searching
client = Client(url="http://localhost:8080")

# udpate to use different model if desired
MODEL_NAME = 'bert-base-uncased'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME).to(device)

# json data
with open('data/thesis.json', 'r', encoding='UTF-8') as file:
    data = json.load(file)

def text2vec(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        padding='longest',
        truncation=True,
        return_tensors='pt'
    ).to('cuda')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    sentence_vector = outputs.pooler_output.detach().cpu().numpy()[0]
    return sentence_vector.tolist()

# Array with only the data to vectorize in json
def get_data2vec(data):
    posts=[]
    for d in data:
        posts+=[d.get('paper_title')]
    
    return posts

# Array of vectorized data
def vectorize_data(posts=[]):
    post_vectors=[]
    before=time.time()
    for i, post in enumerate(posts):
        if i==156785:
            print()
        try:
            if post:
                vec=text2vec(sent_tokenize(post))
            else:
                vec=None
        except:
            pass
        post_vectors += [vec]
        if i % 100 == 0 and i != 0:
            logging.debug("So far {} objects vectorized in {}s".format(i, time.time()-before))
            print("So far {} objects vectorized in {}s".format(i, time.time()-before))
    after=time.time()
    
    logging.debug("Vectorized {} items in {}s".format(len(posts), after-before))
    print("Vectorized {} items in {}s".format(len(posts), after-before))
    
    return post_vectors

def init_weaviateschema():
    CLASSNAME = "VectorDatabase2"
    schema = {
        "class": CLASSNAME,
        "vectorizer": "none",
        "description": "A written text, VectorDatabase for news data to vector data",
        "properties": [
            {
                "name": "data_id",
                "dataType": ["text"],
                "description": "id of the vectordatabase"
            },
            {
                "name": "year",
                "dataType": ["int"],
                "description": "year of the vectordatabase"
            },
            {
                "name": "symposium_title",
                "dataType": ["text"],
                "description": "symposium_title of the vectordatabase"
            },
            {
                "name": "paper_type",
                "dataType": ["text"],
                "description": "paper_type of the vectordatabase"
            },
            {
                "name": "paper_title",
                "dataType": ["text"],
                "description": "paper_title of the vectordatabase"
            },
            {
                "name": "paper_abs",
                "dataType": ["text"],
                "description": "paper_abstract of the vectordatabase"
            }
        ]
        
    }
    client.schema.delete_class(CLASSNAME)
    client.schema.create_class(schema)

# Weaviate에 배치하는 함수
def insert2weaviate(data):
    for i, entry in enumerate(data):
        id = entry.get('id')
        symposium_title = entry.get('year')
        paper_type = entry.get('symposium_title')
        paper_title = entry.get('paper_type')
        paper_abstract = entry.get('paper_title')
        paper_abs = vectorize_data(get_data2vec(data))
        
        properties = {
            "data_id" : id,
            "year" : 0,
            "symposium_title" : symposium_title,
            "paper_type" : paper_type,
            "paper_title" : paper_title,
            "paper_abs" : paper_abstract
        }

        client.data_object.create(
            properties,
            "VectorDatabase2",
            vector=paper_abs[i],
            consistency_level=weaviate.data.replication.ConsistencyLevel.ALL,
        )

def search(query="", limit=1):
    before = time.time()
    vec = text2vec(query)
    vec_took=time.time() - before

    before=time.time()
    near_vec={"vector":vec.tolist}
    res = client \
        .query.get("Post", ["paper_abs2vec", "_additional {certainty}"]) \
        .with_near_vector(near_vec) \
        .with_limit(limit) \
        .do()
    search_took = time.time() - before

    print("\nQuery \"{}\" with {} results took {:.3f}s ({:.3f}s to vectorize and {:.3f}s to search)" \
          .format(query, limit, vec_took+search_took, vec_took, search_took))
    for post in res["data"]["Get"]["Post"]:
        print("{:.4f}: {}".format(post["_additional"]["certainty"], post["content"]))
        print('---')

def near(param):
    # 기본 값 설정
    class_name = 'VectorDatabase'
    ## txt 값 설정
    property=['paper_abs','paper_type']

    ## obj 값 설정
    obj_id = "56b9449e-65db-5df4-887b-0a4773f52aa7"
    ## vec 값 설정
    test_txt = "The radioactivity of SUP14/SUPC and SUP3/SUPH in graphite samples from the dismantled Korea Research Reactor-2 (the KRR-2) site was analyzed by high-temperature oxidation and liquid scintillation counting and the graphite waste was suggested to be disposed of as a low-level radioactive waste. The graphite samples were oxidized at a high temperature of 800SUPo/SUPC and their counting rates were measured by using a liquid scintillation counter (LSC). The combustion ratio of the graphite was about 99% on the sample with a maximum weight of 1g. The recoveries from the combustion furnace were around 100% and 90% in SUP14/SUPC and SUP3/SUPH respectively. The minimum detectable activity was 0.04-0.05Bq/g for the SUP14/SUPC and 0.13-0.15Bq/g for the SUP3/SUPH at the same background counting time. The activity of SUP14/SUPC was higher than that of SUP3/SUPH over all samples with the activity ratios of the SUP14/SUPC to SUP3/SUPH SUP14/SUPC/SUP3/SUPH being between 2.8 and 25. The dose calculation was carried out from its radioactivity analysis results. The dose estimation gave a higher annual dose than the domestic legal limit for a clearance. It was thought that the sampled graphite waste from the dismantled research reactor was not available for reuse or recycling and should be monitored as low-level radioactive waste."
    
    if param == 'txt':
        response = (
            client.query
            .get(class_name, property)
            .with_near_text({
                "concepts": ["animals in movies"]
            })
            .with_limit(2)
            .with_additional(["distance"])
            .do()
        )
        
    if param == 'obj':
        response = (
            client.query
            .get("JeopardyQuestion", ["question", "answer"])
            .with_near_object({
                "id": obj_id
            })
            .with_limit(2)
            .with_additional(["distance"])
            .do()
        )

    if param == 'vec':
        response = (
            client.query
            .get(class_name, property)
            .with_near_vector({
                "vector": text2vec(test_txt)
            })
            .with_limit(5)
            .with_additional(["distance"])
            .do()
        )

    return response
        
# txt : Finds objects closest to an input medium
# obj : Finds objects closest to another Weaviate object
# vec : Find objects closest to an input vector
print(json.dumps(near('vec'), indent=2))
search(data)

init_weaviateschema()
insert2weaviate(data)
