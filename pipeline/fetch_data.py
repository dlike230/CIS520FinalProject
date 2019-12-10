from typing import List

import pandas as pd
from bs4 import BeautifulSoup as Soup
from pandas import DataFrame


def get_df() -> DataFrame:
    return pd.read_csv("../data/Reviews.csv", sep=',', quotechar='"')


def extract_text(df) -> List[str]:
    return [Soup(text, features="html.parser").get_text() for text in df["Text"]]


def fetch_data(sample_size=10000):
    raw_df = get_df()
    if sample_size is not None:
        df = raw_df.sample(n=sample_size)
    else:
        df = raw_df
    texts = extract_text(df)
    return texts, df
