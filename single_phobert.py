import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model():
    checkpoint = "mr4/phobert-base-vi-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.config.id2label = {0: 'Negative', 1:'Positive', 2:'Neutral'}
    return model,tokenizer

def sentiment(comment,model,tokenizer)-> dict:
    result_dict = {}
    result,score,max_score = None,None,None
    raw_inputs = comment
    try:
        inputs = tokenizer(raw_inputs, padding=True,
                        truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        idx = torch.argmax(predictions)
        result = model.config.id2label[int(idx)]
        score = predictions.tolist()[0]
        max_score = max(predictions.tolist()[0])
    except Exception as E:
        print(E)

    result_dict = {
        'comment' : raw_inputs,
        'result' : result,
        'score' : score,
        'max_score' : max_score
    }
    return result_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='PhoBERT',
                    description='Sentiment Analysis')
    parser.add_argument('comment', help='comment you need to get the sentiment')
    args = parser.parse_args()
    phobert,tokenizer = load_model()
    output = sentiment(args.comment,phobert,tokenizer)
    print(output)
 