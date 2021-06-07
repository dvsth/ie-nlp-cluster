import torch
print(torch.cuda.is_available())

device = "cuda:0" if torch.cuda.is_available() else "cpu"
sentence = 'Hello World!'
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained('bert-large-uncased')

inputs = tokenizer(sentence, return_tensors="pt").to(device)
model = model.to(device)
outputs = model(**inputs)