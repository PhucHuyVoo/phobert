import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model():
    checkpoint = "mr4/phobert-base-vi-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    return model,tokenizer

def sentiment(df,model,tokenizer)-> pd.DataFrame:
    df_sentiment = df.copy()
    results,scores = [],[]
    for i in tqdm(range(len(df_sentiment))):
        comment = df_sentiment["comment"][i]
        raw_inputs = comment
        try:
            inputs = tokenizer(raw_inputs, padding=True,
                            truncation=True, return_tensors="pt")
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            idx = torch.argmax(predictions)
            results.append(model.config.id2label[int(idx)])
            scores.append(predictions.tolist()[0])
        except Exception as E:
            print(E)
            results.append("")
            scores.append("")
            continue
    df_sentiment["phobert"] = results
    df_sentiment["scores"] = scores
    return df_sentiment

    