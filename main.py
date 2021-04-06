import random
import numpy as np
from intents_model import IntentsModel


class chatbot(IntentsModel):
    def __init__(self, lang="en"):
        super().__init__(lang=lang)

    def bag_of_words(self, s, words):
        bag = [0 for _ in range(len(words))]

        s_words = self.lemmatize(s)

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1

        return np.array(bag)

    def chat(self):

        print("Start talking with the bot (type quit to stop)!")
        while True:
            input_words = input("You: ")
            if input_words.lower() == "quit":
                break

            results = self.get_model().predict(
                np.array([self.bag_of_words(input_words, self.get_words())])
            )[0]

            results_index = np.argmax(results)
            tag = self.get_labels()[results_index]

            for tg in self.get_intents()["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]

            print(random.choice(responses))


if __name__ == "__main__":
    bot = chatbot("sv")
    bot.chat()
