# import torch
# from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
# configuration = RobertaConfig()  # Инициализация конфигурации RoBERTa
# model = RobertaModel(configuration)  # Инициализация модели из конфигурации
# configuration = model.config  # Доступ к конфигурации модели
# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")  # Этот токенизатор был обучен обрабатывать пробелы как
# # части токенов (немного как часть предложения), поэтому слово будет кодироваться по-разному, независимо от того,
# # находится оно в начале предложения (без пробела) или нет

from transformers import RobertaTokenizer, RobertaModel
import torch

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
print(inputs, '\n')

outputs = model(**inputs)
print(outputs, '\n')

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)