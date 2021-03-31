import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy as np
import tensorflow as tf
import random
import json
import pickle
import os


class Chatbot(object):
    def __init__(
        self, intents="intents.json", pickled_data="data.pickle", model="chatbot-model"
    ):
        if os.path.exists(intents):
            self.intents = intents
            with open("intents.json") as f:
                self.intents_data = json.load(f)
        else:
            raise Exception(f"Cannot find intents.json, abort.")

        if os.path.exists(pickled_data):
            with open(pickled_data, "rb") as f:
                self.words, self.labels, self.training, self.output = pickle.load(f)
        else:
            self.preprocess(pickled_data)

        if not os.path.exists(model):
            self.train_model(model)

    def preprocess(self, pickled_data):
        self.words = []
        self.labels = []
        docs_x = []
        docs_y = []

        for intent in self.intents_data["intents"]:
            for pattern in intent["patterns"]:
                wrds = nltk.word_tokenize(pattern)
                self.words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent["tag"])

            if intent["tag"] not in self.labels:
                self.labels.append(intent["tag"])

        self.words = [
            stemmer.stem(w.lower()) for w in self.words if w not in ["?", ".", ",", "!"]
        ]
        self.words = sorted(list(set(self.words)))

        self.labels = sorted(self.labels)

        training = []
        output = []

        out_empty = [0 for _ in range(len(self.labels))]

        for x, doc in enumerate(docs_x):
            bag = []
            wrds = [stemmer.stem(w.lower()) for w in doc]

            for w in self.words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = out_empty[:]
            output_row[self.labels.index(docs_y[x])] = 1

            training.append(bag)
            output.append(output_row)

        self.training = np.array(training)
        self.output = np.array(output)

        with open(pickled_data, "wb") as f:
            pickle.dump((self.words, self.labels, self.training, self.output), f)

    def train_model(self, model_name):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    8, activation="relu", input_shape=(len(self.training[0]),)
                ),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(len(self.output[0]), activation="softmax"),
            ]
        )
        model.compile(
            # YOUR CODE HERE
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

        model.fit(self.training, self.output, epochs=10, batch_size=8, verbose=2)
        model.save(model_name)


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict(np.array([bag_of_words(inp, words)]))[0]

        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg["tag"] == tag:
                responses = tg["responses"]

        print(random.choice(responses))


if __name__ == "__main__":
    chat()
