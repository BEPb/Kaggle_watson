'''
Выявление естественного языка (NLI) - это классическая проблема NLP (обработка естественного языка), которая включает
в себя выбор двух предложений (посылки и гипотезы) и определение того, как они связаны - если посылка влечет за собой
гипотезу, противоречит ей или нет.

В этом руководстве мы рассмотрим набор данных о соревнованиях Contradictory, My Dear Watson, построим предварительную
модель с использованием Tensorflow 2, Keras и BERT и подготовим файл для отправки.
'''
import numpy as np  # линейная алгебра
import pandas as pd  # обработка данных, CSV file I/O (e.g. pd.read_csv)

# подлючим библиотеки для отображения графиков
import matplotlib.pyplot as plt  #

import matplotlib
matplotlib.use('tkagg')
import tkinter  # библиотека графических окон
import seaborn as sns  # для построения графика распределения классов внутри каждого языка

from nlp import load_dataset


train = pd.read_csv("/home/user/PycharmProjects/Kaggle_watson/train.csv")
test = pd.read_csv("/home/user/PycharmProjects/Kaggle_watson/test.csv")

print(train.head())

#FC : только французские тексты
print(train[train.lang_abv=='fr'].head())

# Давайте посмотрим на одну из пар предложений.
print("\nПример заключения:","\n")
print(train.premise.values[2])
print(train.hypothesis.values[2])
print(train.label.values[2])


print("\nНейтральный пример:","\n")
print(train.premise.values[69])
print(train.hypothesis.values[69])
print(train.label.values[69])

print("\nПротиворечивый пример:","\n")
print(train.premise.values[41])
print(train.hypothesis.values[41])
print(train.label.values[41])


# Давайте посмотрим на размер набора данных и распределение языков в обучающем наборе.
def load_mnli(use_validation=True):
    result = []
    dataset = load_dataset('multi_nli')
    print(dataset['train'])
    keys = ['train', 'validation_matched','validation_mismatched'] if use_validation else ['train']
    for k in keys:
        for record in dataset[k]:
            c1, c2, c3 = record['premise'], record['hypothesis'], record['label']
            if c1 and c2 and c3 in {0,1,2}:
                result.append((c1,c2,c3,'en'))
    result = pd.DataFrame(result, columns=['premise','hypothesis', 'label','lang_abv'])
    return result

mnli = load_mnli()

total_train = train[['id', 'premise', 'hypothesis','lang_abv', 'language', 'label']]
print(total_train.head())  # оценим содержимое тренировочной таблицы

mnli = mnli[['premise', 'hypothesis', 'lang_abv', 'label']]
mnli.insert(0, 'language', 'English')
mnli = mnli[['premise', 'hypothesis', 'lang_abv', 'language', 'label']]
mnli.insert(0, 'id', 'xxx')
print(mnli.head())  # оценим выборку на английском

total_train = pd.concat([total_train, mnli], axis = 0)

print('\n train table \n', train.describe(include='all'))

print('\n test table \n', test.describe(include='all'))


# построим круговую диаграмму распределения тренировочного датасета по языкам
labels, frequencies = np.unique(train.language.values, return_counts = True)
plt.figure(figsize = (10,10))
plt.pie(frequencies,labels = labels, autopct = '%1.1f%%')
plt.show()


# изучим распределение классов внутри языков
fig, ax = plt.subplots(figsize = (12,5))
# для максимальной эстетики
palette = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True)
graph1 = sns.countplot(train['language'], hue = train['label'])
graph1.set_title('Distribution of Languages and Labels')
plt.tight_layout()
plt.show()





