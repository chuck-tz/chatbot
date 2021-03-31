import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import pickle
import os


class IntentionsModel(object):
    def __init__(
        self,
        intents="intents.json",
        pickled_data="data.pickle",
        model_file="chatbot-model",
        retrain=False,
    ):
        if os.path.exists(intents):
            with open(intents) as f:
                self.intents = json.load(f)
        else:
            raise Exception(f"Cannot find intents.json, abort.")

        if os.path.exists(pickled_data):
            with open(pickled_data, "rb") as f:
                self.words, self.labels, self.training, self.output = pickle.load(f)
        else:
            self.preprocess(pickled_data)

        self.model = self.train_model(model_file, retrain)

    def preprocess(self, pickled_data):
        self.words = []
        self.labels = []
        docs_x = []
        docs_y = []

        for intent in self.intents["intents"]:
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

    def train_model(self, model_name, retrain):
        if retrain or not os.path.exists(model_name):
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

            model.fit(self.training, self.output, epochs=1000, batch_size=8, verbose=2)
            model.save(model_name)
        else:
            model = keras.models.load_model(model_name)

        return model

    def get_model(self):
        return self.model

    def get_words(self):
        return self.words

    def get_labels(self):
        return self.labels

    def get_intents(self):
        return self.intents