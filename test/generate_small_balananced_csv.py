from pipeline.fetch_data import get_df, balance_dataset

df = get_df()
df["Score"] = df["Score"].apply(lambda x: 1 if x > 3 else 0)
df = balance_dataset(df, "Score", 20000)
df.to_csv("../data/small_data_balanced.csv", sep=',', quotechar='"')