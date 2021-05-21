import json
import json
import os
import pickle
import random

import nltk
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer

# Dimensionality (number of features)
dimensionality = 256
# The batch size
batch_size = 10
# The number of training epochs
epochs = 60

lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']
intents_file = open('./data/intents.json', encoding='UTF8').read()
intents = json.loads(intents_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # токенизация всех слов
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        # добавление документа в корпус
        documents.append((word, intent['tag']))
        # добавление к списку классов
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# лемматизация, удаление дубликатов
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
# сортировка классов
classes = sorted(list(set(classes)))
# documents = связка patterns + intents
print(len(documents), "documents")
# classes = intents
print(len(classes), "classes", classes)
# words = полный словарь
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open('./data/dataset-lightweight-dialogues.pkl', 'wb'))
pickle.dump(classes, open('./data/classes.pkl', 'wb'))

# подготовка тренировочных данных
training = []
# создание пустого массива для вывода
output_empty = [0] * len(classes)
# тренировочный сет: мешок слов для каждого предложения
for doc in documents:
    # инициализация мешка слов
    bag = []
    # список токенов
    pattern_words = doc[0]
    # лемматизация каждого слова, подбор связанных с ним слов
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # если слово найдено в паттерне, добавляем его в мешок
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    # вывод '0' для всех тегов и '1' для текущего тега (для всех паттернов)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# перемешивание признаков и добавление их в np.array
random.shuffle(training)
training = np.array(training)
# создание списков для тренировки и тестирования: X - паттерны, Y - интенты
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

if not os.path.isfile('retrieval.h5'):
    # Создание модели - 3 слоя, в первом 128 нейронов, во втором 64 нейрона,
    # в третьем количество нейронов равно количеству интентов для составления прогноза (функция softmax)
    model = Sequential()
    model.add(Dense(dimensionality, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # Компиляция модели. Для этой модели выбран стохастический градиентный спуск с ускоренным градиентом Нестерова
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Обучение и сохранение модели
    hist = model.fit(np.array(train_x), np.array(train_y), epochs=epochs, batch_size=batch_size, verbose=1)
    model.save('retrieval.h5', hist)
    print("model created")

model = load_model('retrieval.h5')


intents = json.loads(open('./data/intents.json', encoding='UTF8').read())
words = pickle.load(open('./data/dataset-lightweight-dialogues.pkl', 'rb'))
classes = pickle.load(open('./data/classes.pkl', 'rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - splitting words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stemming every word - reducing to base form
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for words that exist in sentence
def bag_of_words(sentence, words, show_details=True):
    # tokenizing patterns
    sentence_words = clean_up_sentence(sentence)
    # bag of words - vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % word)
    return (np.array(bag))


def predict_class(sentence):
    # filter below  threshold predictions
    p = bag_of_words(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sorting strength probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getAnswer(ints):
    result = ''
    tag = ints[0]['intent']
    list_of_intents = intents['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result

