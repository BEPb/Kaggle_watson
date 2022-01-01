'''
Выявление естественного языка (NLI) - это классическая проблема NLP (обработка естественного языка), которая включает
в себя выбор двух предложений (посылки и гипотезы) и определение того, как они связаны - если посылка влечет за собой
гипотезу, противоречит ей или нет.

В этом руководстве мы рассмотрим набор данных о соревнованиях Contradictory, My Dear Watson, построим предварительную
модель с использованием Tensorflow 2, Keras и BERT и подготовим файл для отправки.
'''
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf

train = pd.read_csv("../input/contradictory-my-dear-watson/train.csv")
test = pd.read_csv("../input/contradictory-my-dear-watson/test.csv")

train.head()

#FC : только французские тексты
train[train.lang_abv=='fr'].head()

# Давайте посмотрим на одну из пар предложений.
print("Пример заключения","\n")
print(train.premise.values[2])
print(train.hypothesis.values[2])
print(train.label.values[2])


print("Нейтральный пример","\n")
print(train.premise.values[69])
print(train.hypothesis.values[69])
print(train.label.values[69])

print("Противоречивый пример","\n")
print(train.premise.values[41])
print(train.hypothesis.values[41])
print(train.label.values[41])


