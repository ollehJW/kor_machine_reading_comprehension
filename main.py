import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import json

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def predict_answer(qa_text_pair):
    # Encoding
    encodings = tokenizer(qa_text_pair['context'], qa_text_pair['question'], 
                      max_length=512, 
                      truncation=True,
                      padding="max_length", 
                      return_token_type_ids=False,
                      return_offsets_mapping=True
                      )
    encodings = {key: torch.tensor([val]).to(device) for key, val in encodings.items()}             

    # Predict
    pred = model(encodings["input_ids"], attention_mask=encodings["attention_mask"])
    start_logits, end_logits = pred.start_logits, pred.end_logits
    token_start_index, token_end_index = start_logits.argmax(dim=-1), end_logits.argmax(dim=-1)
    pred_ids = encodings["input_ids"][0][token_start_index: token_end_index + 1]
    answer_text = tokenizer.decode(pred_ids)

    # Offset
    answer_start_offset = int(encodings['offset_mapping'][0][token_start_index][0][0])
    answer_end_offset = int(encodings['offset_mapping'][0][token_end_index][0][1])
    answer_offset = (answer_start_offset, answer_end_offset)
 
    return {'answer_text':answer_text, 'answer_offset':answer_offset}


## Load fine-tuned MRC model by HuggingFace Model Hub ##
HUGGINGFACE_MODEL_PATH = "bespin-global/klue-bert-base-aihub-mrc"

tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(HUGGINGFACE_MODEL_PATH).to(device)


## Predict ## 
import json
import pandas as pd
with open("./data/test.json", "r", encoding='utf-8') as json_file:
    test = json.load(json_file)

answer_list = []
guid_list = []

for i in range(len(test['data'])):
    question = test['data'][i]['paragraphs'][0]['qas'][0]['question']
    context = test['data'][i]['paragraphs'][0]['context']
    guid = test['data'][i]['paragraphs'][0]['qas'][0]['guid']
    qa_text_pair = {'context':context, 'question':question}
    result = predict_answer(qa_text_pair)
    answer_list.append(result['answer_text'])
    guid_list.append(guid)
    if i % 100 == 0:
        print(i)
        print(result['answer_text'])

    
result = pd.DataFrame({'guid': guid_list, 'answer': answer_list})
result.to_csv('test_result1.csv', index = False, encoding = 'utf-8')
