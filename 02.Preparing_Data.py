'''
Подготовка данных и тренировка модели
'''
import pandas as pd  # обработка данных, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  #
from nlp import load_dataset
from transformers import BertTokenizer, TFBertModel, TFAutoModel, AutoTokenizer


train = pd.read_csv("/home/user/PycharmProjects/Kaggle_watson/train.csv")
test = pd.read_csv("/home/user/PycharmProjects/Kaggle_watson/test.csv")

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
mnli = mnli[['premise', 'hypothesis', 'lang_abv', 'label']]
mnli.insert(0, 'language', 'English')
mnli = mnli[['premise', 'hypothesis', 'lang_abv', 'language', 'label']]
mnli.insert(0, 'id', 'xxx')
total_train = pd.concat([total_train, mnli], axis = 0)


#model_name = "bert-base-multilingual-cased"
#tokenizer = BertTokenizer.from_pretrained(model_name) # это токенизатор, который мы будем использовать для наших текстовых данных, чтобы токенизировать их

# model_name = "joeddav/xlm-roberta-large-xnli"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model_name = TFBertModel.from_pretrained('bert-base-cased')



# Токенизаторы превращают последовательности слов в массивы чисел. Давайте посмотрим на пример:
print(list(tokenizer.tokenize("I love machine learning")))  # tokenize только создать список слов

# мы делаем функцию, чтобы иметь список идентификаторов для каждого слова и разделителя
def encode_sentence(s):
   tokens = list(tokenizer.tokenize(s))  # разделить предложение на токены, которые являются словами или подсловами
   tokens.append('[SEP]') # для обозначения конца каждого предложения добавляется маркер [SEP] (= разделитель).
   return tokenizer.convert_tokens_to_ids(tokens)  # вместо того, чтобы возвращать список токенов, возвращается
   # список каждого идентификатора токена

encode_sentence("I love machine learning")  # вывод представляет собой число для каждого слова плюс идентификатор
# токена [SEP].



def bert_encode(hypotheses, premises,
                tokenizer):  # FC: for RoBERTa we remove the input_type_ids from the inputs of the model

    num_examples = len(hypotheses)

    sentence1 = tf.ragged.constant([  # FC: constructs a constant ragged tensor. every entry has a different length
        encode_sentence(s) for s in np.array(hypotheses)])

    sentence2 = tf.ragged.constant([
        encode_sentence(s) for s in np.array(premises)])

    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * sentence1.shape[
        0]  # FC: list of IDs for the token '[CLS]' to denote each beginning

    input_word_ids = tf.concat([cls, sentence1, sentence2],
                               axis=-1)  # FC: put everything together. every row still has a different length.

    # input_word_ids2 = tf.concat([cls, sentence2, sentence1], axis=-1)

    # input_word_ids = tf.concat([input_word_ids1, input_word_ids2], axis=0) # we duplicate the dataset inverting sentence 1 and 2

    input_mask = tf.ones_like(
        input_word_ids).to_tensor()  # FC: first, a tensor with just ones in it is constructed in the same size as input_word_ids. Then, by applying to_tensor the ends of each row are padded with zeros to give every row the same length

    # type is not need for the RoBERTa model it will not be include in the output of this function
    type_cls = tf.zeros_like(cls)  # FC: creates a tensor same shape as cls with only zeros in it

    type_s1 = tf.zeros_like(sentence1)

    type_s2 = tf.ones_like(
        sentence2)  # FC: creates a tensor same shape as sentence2 with only ones in it to mark the 2nd sentence

    input_type_ids = tf.concat(
        [type_cls, type_s1, type_s2], axis=-1).to_tensor()  # FC: concatenates everything and again adds padding

    inputs = {
        'input_word_ids': input_word_ids.to_tensor(),  # FC: input_word_ids hasn't been padded yet - do it here now
        'input_mask': input_mask

        # ,'input_type_ids': input_type_ids
    }

    return inputs

train_input = bert_encode(train.premise.values, train.hypothesis.values, tokenizer)
print(train_input)

total_train_input = bert_encode(total_train.premise.values, total_train.hypothesis.values, tokenizer)

train.label.values.shape

total_train.label.values.shape

print(np.count_nonzero(train_input['input_word_ids'], axis=1))


# Fixing random state for reproducibility
np.random.seed(19680801)


x = np.count_nonzero(train_input['input_word_ids'], axis=1)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, density=True, facecolor='b', alpha=0.75)


plt.xlabel('input word lenght')
plt.ylabel('Probability')
plt.title('Distribution of word length on the train set')
plt.text(60, .021, r'max_length=245')
plt.xlim(0, 250)
#plt.ylim(0, 0.03)
plt.grid(True)
plt.show()


test_input = bert_encode(test.premise.values, test.hypothesis.values, tokenizer)



# Fixing random state for reproducibility
np.random.seed(19680801)


x = np.count_nonzero(test_input['input_word_ids'], axis=1)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, density=True, facecolor='b', alpha=0.75)


plt.xlabel('input word lenght')
plt.ylabel('Probability')
plt.title('Distribution of word length on the test set')
plt.text(60, .021, r'max_length=236')
plt.xlim(0, 250)
#plt.ylim(0, 0.03)
plt.grid(True)
plt.show()

# Fixing random state for reproducibility
np.random.seed(19680801)


x = np.count_nonzero(total_train_input['input_word_ids'], axis=1)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, density=True, facecolor='b', alpha=0.75)


plt.xlabel('input word lenght')
plt.ylabel('Probability')
plt.title('Distribution of word length on the test set')
plt.text(60, .021, r'max_length=236')
plt.xlim(0, 250)
#plt.ylim(0, 0.03)
plt.grid(True)
plt.show()

'''
Создание и обучение модели
Теперь мы можем включить преобразователь BERT в функциональную модель Кераса. Для получения дополнительной информации 
о функциональном API Keras см .: https://www.tensorflow.org/guide/keras/functional.

Эта модель была вдохновлена моделью из этого блокнота: 
https://www.kaggle.com/tanulsingh077/deep-learning-for-nlp-zero-to-transformers-bert#BERT-and-Its-Implementation-on
-this -Соревнования, которые являются прекрасным знакомством с НЛП!  

- Теперь мы готовы построить реальную модель. Как упоминалось выше, окончательная модель будет состоять из большой 
модели RoBERTa, которая выполняет контекстное встраивание идентификаторов входных токенов, которые затем передаются 
классификатору, который будет возвращать вероятности для каждой из трех возможных меток «влечет за собой» (0), 
«нейтральный». «(1), или« противоречие »(2). Классификатор состоит из регулярной плотносвязной нейронной сети.   
'''

max_len = 236  #: FC 50 in the initial tutorial


def build_model():
    # encoder = TFBertModel.from_pretrained(model_name)
    # FC: constructs a RoBERTa model pre-trained on the above described language model 'xlm-roberta-large-xnli'
    encoder = TFAutoModel.from_pretrained('joeddav/xlm-roberta-large-xnli')
    # FC: now we adjust the model so that it can accept our input by telling the model what the input looks like:
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32,
                                    name="input_word_ids")  # FC: tf.keras.Input constructs a symbolic tensor object whith certain attributes: "shape" tells it that the expected input will be in batches of max_len-dimensional vectors; "dtype" tells it that the data type will be int32; "name" will be the name string for the input layer
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32,
                                name="input_mask")  # FC: repeat the same for the other two input variables
    # FC: the input type is only needed for the BERT model
    # input_type_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_type_ids")

    # FC: now follows, what we want to happen with our input:
    # FC: first, our input goes into the BERT model bert_encoder. It will return a tuple and the contextualized embeddings that we need are stored in the first element of that tuple
    embedding = encoder([input_word_ids, input_mask])[0]  # FC: add_input_type_ids for the BERT model
    # FC: we only need the output corresponding to the first token [CLS], which is a 2D-tensor with size (#sentence pairs, 768) and is accessd with embedding[:,0,:]. This will be input for our classifier, which is a regular densely-connected neural network constructed through tf.keras.layers.Dense. The inputs mean: "3" is the dimensionality of the output space, which means that the output has shape (#sentence pairs,3). More practically speaking, for each sentence pair that we input, the output will have 3 probability values for each of the 3 possible labels (entailment, neutral, contradiction). They will be in range(0,1) and add up to 1; "activation" denotes the activation function, in this case 'softmax', which connects a real vector to a vector of categorical possibilities.

    # I tried to put another layer put it doesn't help in performance
    # output = tf.keras.layers.Dense(10, activation='softmax')(embedding[:,0,:]) #FC: no need of a GlobalAveragePooling for BERT

    output = tf.keras.layers.Dense(3, activation='softmax')(embedding[:, 0, :])
    # FC: we also have the posibility of making a globalAveragepooling of all the embeddings, but the resuls are not better
    # output = tf.keras.layers.GlobalAveragePooling1D()(embedding)
    # output = tf.keras.layers.Dense(3, activation='softmax')(output)

    model = tf.keras.Model(inputs=[input_word_ids, input_mask],
                           outputs=output)  # FC: based on the code in the lines above, a model is now constructed and passed into the variable model
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=[
        'accuracy'])  # FC: we tell the model how we want it to train and evaluate: "tf.keras.optimizers.Adam": use an optimizer that implements the Adam algorithm. "lr" denotes the learning rate; "loss" denotes the loss function to use; "metrics" specifies which kind of metrics to use for training and testing

    return model

# Почему нам нужно только встраивать [0] [:, 0 ,:] из вывода BERT? **
#
# Вот замечательная статья в блоге (включая блокнот), которая объясняет это очень подробно (см., В частности,
# раздел «Распаковка выходного тензора BERT» для наглядной визуализации). Я постараюсь дать здесь краткое резюме:
# встраивание [0] состоит из трехмерного тензора с номером (встраивание) для каждого токена в паре предложений (
# столбцах) для каждой пары предложений (строк) для каждой скрытой единицы в BERT (768 ). Поскольку его вывод,
# соответствующий первому токену [CLS] (он же embedding [0] [:, 0 ,:], это двумерный тензор с размером (# пары
# предложений, 768)), можно рассматривать как вложение для каждого отдельного человека. пары предложений, достаточно
# просто использовать ее в качестве входных данных для нашей модели классификации.

#
# Настроим наш ТПУ.

try:
    # detect and init the TPU
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # FC: detect and init the TPU: TPUClusterResolver() locates the TPUs on the network
    # instantiate a distribution strategy
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(
        tpu)  # FC: "strategy" contains the necessary distributed training code that will work on the TPUs
except ValueError:  # FC: in case Accelerator is not set to TPU in the Notebook Settings
    strategy = tf.distribute.get_strategy()  # for CPU and single GPU
    print('Number of replicas:', strategy.num_replicas_in_sync)  # FC: returns the number of cores



try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
except ValueError:
  tpu = None
  gpus = tf.config.experimental.list_logical_devices("GPU")

if tpu:
  tf.tpu.experimental.initialize_tpu_system(tpu)
  strategy = tf.distribute.experimental.TPUStrategy(tpu,) # Going back and forth between TPU and host is expensive. Better to run 128 batches on the TPU before reporting back.
  print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
elif len(gpus) > 1:
  strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
  print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
elif len(gpus) == 1:
  strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
  print('Running on single GPU ', gpus[0].name)
else:
  strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
  print('Running on CPU')
print("Number of accelerators: ", strategy.num_replicas_in_sync)

# instantiating the model in the strategy scope creates the model on the TPU

with strategy.scope(): # FC: defines the compute distribution policy for building the model. or in other words: makes sure that the model is created on the TPU/GPU/CPU, depending on to what the Accelerator is set in the Notebook Settings
    model = build_model() # FC: our model is being built
    model.summary()       # FC: let's look at some of its properties

tf.keras.utils.plot_model(model, "my_model.png", show_shapes=True) # FC: I added this line because it gives a nice visualization showing the individual components of our model


# - На приведенном выше графике очень подробно показано, как выглядит наша модель и ее входные данные:
# input_word_ids, input_mask и input_type_ids - это 3 входные переменные для модели BERT, которая, в свою очередь,
# возвращает кортеж. Вложения слов, которые хранятся в первой записи кортежа, затем передаются классификатору,
# который затем возвращает 3 категориальные вероятности. Знаки вопроса обозначают количество строк во входных данных,
# которые, конечно, неизвестны.

# We can freeze the RoBERTa weights in order to save some time
print(model.layers[2])
model.layers[2].trainable=True

# We need to put the train set with the same size of the model
for key in train_input.keys():
    train_input[key] = train_input[key][:,:max_len]

# We need to put the train set with the same size of the model
for key in total_train_input.keys():
    total_train_input[key] = total_train_input[key][:,:max_len]

early_stop = tf.keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)
# FC: make sure that TPU in Accelerator under Notebook Settings is turned on so that model trains on the TPU. Otherwise this line will crash
model.fit(total_train_input, total_train.label.values, epochs = 30, verbose = 1, validation_split = 0.01,
         batch_size=16*strategy.num_replicas_in_sync
          ,callbacks=[early_stop]
         ) # FC: now we fit the model to our training data that we prepared before. The number of training epochs is 2, verbose = 1 shows progress bar, # of rows in each batch is 64, and 20% of the data is used for validation


test = pd.read_csv("../input/contradictory-my-dear-watson/test.csv")
test_input = bert_encode(test.premise.values, test.hypothesis.values, tokenizer) # FC: finally we prepare our competition data for the model

# same for the test set we need to put it in the same size of the model
for key in test_input.keys():
    test_input[key] = test_input[key][:,:max_len]

print(test.head())

predictions = [np.argmax(i) for i in model.predict(test_input)] # FC; ve the model predict three categorical
# probabilities, choose the highest probability, and save the respective label ID (0,1, or 2)

# Файл отправки будет состоять из столбца идентификатора и столбца прогноза. Мы можем просто скопировать столбец ID
# из тестового файла, сделать его фреймом данных, а затем добавить столбец прогноза.

submission = test.id.copy().to_frame()
submission['prediction'] = predictions

print(submission.head())

submission.to_csv("submission.csv", index = False)






