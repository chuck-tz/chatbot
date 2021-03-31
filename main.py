import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
import random
import numpy as np
from IntentionsModel import IntentionsModel


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
    intentions = IntentionsModel()
    model = intentions.get_model()
    words = intentions.get_words()
    labels = intentions.get_labels()
    intents = intentions.get_intents()

    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict(np.array([bag_of_words(inp, words)]))[0]

        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in intents["intents"]:
            if tg["tag"] == tag:
                responses = tg["responses"]

        print(random.choice(responses))


if __name__ == "__main__":
    chat()