import pandas as pd
from bs4 import BeautifulSoup as Soup


def fetch_data():
    raw_df = pd.read_csv("../data/Reviews.csv", sep=',', quotechar='"')
    df = raw_df.sample(n=10000)
    texts = df["Text"]
    texts = [Soup(text, features="html.parser").get_text() for text in texts]
    return texts, df
