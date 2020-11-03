# _*_ coding: utf-8 _*_

import numpy as np
import requests
import bilsm_crf_model
import process_data

_, (vocab, chunk_tags) = bilsm_crf_model.create_model(train=False)
predict_text = '中华人民共和国国务院总理刘勇在陈宏毅的陪同下，连续访问了埃塞俄比亚等非洲10国以及阿尔巴尼亚'
_input, length = process_data.process_data(predict_text, vocab)

url = "http://localhost:8080/v1/models/ner:predict"
# url = "http://localhost:8080/v1/models/nlp:predict"

response = requests.post(url, json={"instances": _input.tolist()})
data = response.json()
print(data)

raw = data.get("predictions")[0][-length:]

result = [np.argmax(row) for row in raw]
result_tags = [chunk_tags[i] for i in result]

per, loc, org = '', '', ''

for s, t in zip(predict_text, result_tags):
    if t in ('B-PER', 'I-PER'):
        per += ' ' + s if (t == 'B-PER') else s
    if t in ('B-ORG', 'I-ORG'):
        org += ' ' + s if (t == 'B-ORG') else s
    if t in ('B-LOC', 'I-LOC'):
        loc += ' ' + s if (t == 'B-LOC') else s

print(['person:' + per, 'location:' + loc, 'organzation:' + org])
