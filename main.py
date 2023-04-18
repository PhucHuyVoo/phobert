import argparse
import os

import pandas as pd

from phobert import load_model, sentiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='PhoBERT',
                    description='Sentiment Analysis')
    parser.add_argument('filename', help='csv file destination')
    args = parser.parse_args()
    phobert,tokenizer = load_model()
    df = pd.read_csv(args.filename)
    df_sentiment = sentiment(df,phobert,tokenizer)
    df_sentiment.to_csv(f"/src/phobert_{os.path.basename(args.filename)}")