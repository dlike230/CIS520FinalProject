from models.LstmAutoEncoder import LstmAutoEncoder
from pipeline.fetch_data import extract_text, get_df

autoencoder = LstmAutoEncoder()
text_data = extract_text(get_df().sample(n=10000))
autoencoder.fit_transform(text_data)