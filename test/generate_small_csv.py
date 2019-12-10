from pipeline.fetch_data import fetch_data, get_df

get_df().sample(n=20000).to_csv("../data/small_data.csv", sep=',', quotechar='"')