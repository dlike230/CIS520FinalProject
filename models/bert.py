# from bert_sent_encoding import bert_sent_encoding 


# bse = bert_sent_encoding(model_path='C:/Users/Owen/.pytorch_pretrained_bert/9c41111e2de84547a463fd39217199738d1e3deb72d4fec4399e6e241983c6f0.ae3cef932725ca7a30cdcb93fc6e09150a55e2a130ec7af63975a16c153ae2ba', seq_length=64, batch_size=8) # 2nd line


# vector = bse.get_vector('test1', word_vector=False, layer=-1)   # 3rd line 1. get vector of string
# vectors = bse.get_vector(['test2', 'test3'], word_vector=False, layer=-1)  # 4th line 2. get vector list of strings
# print(vector)
# print(vectors)
# bse.write_txt2vector(input_file, output_file, word_vector=False, layer=-1)   # 5th line 3. get and write vectors of strings

# import torch
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForSequenceClassification

# # OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
# import logging
# logging.basicConfig(level=logging.INFO)

# # Load pre-trained model tokenizer (vocabulary)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# # Tokenized input
# text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
# tokenized_text = tokenizer.tokenize(text)

# print(tokenized_text)

# # Mask a token that we will try to predict back with `BertForMaskedLM`
# masked_index = 8
# tokenized_text[masked_index] = '[MASK]'
# assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']

# # Convert token to vocabulary indices
# indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
# segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]


# # Convert inputs to PyTorch tensors
# tokens_tensor = torch.tensor([indexed_tokens])
# segments_tensors = torch.tensor([segments_ids])

# print(tokens_tensor.shape)

# # model = BertModel.from_pretrained('bert-base-uncased')
# # model.eval()
# # b = model.embeddings(tokens_tensor)
# # print(b)
# # print(b.shape)
# # a = model.encoder(text)

# # print(model)
# # print(a)

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 3)

# model.

# test = model.encoder(tokenizer.do_basic_tokenize(text))
# print(test)

# # If you have a GPU, put everything on cuda
# # tokens_tensor = tokens_tensor.to('cuda')
# # segments_tensors = segments_tensors.to('cuda')
# # model.to('cuda')

# # Predict hidden states features for each layer
# with torch.no_grad():
#     encoded_layers, _ = model(tokens_tensor, segments_tensors)
# # We have a hidden states for each of the 12 layers in model bert-base-uncased
# assert len(encoded_layers) == 12

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForSequenceClassification
# from pytorch_transformers import *

model_class = BertModel
tokenizer_class = BertTokenizer
pretrained_weights = 'bert-base-uncased'

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

# Encode text
ttext = tokenizer.tokenize("Here is some text to encode")
print(ttext)

input_ids = torch.tensor(ttext)
last_hidden_states = model(input_ids)[0][0]

print(last_hidden_states)