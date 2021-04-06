import stanza


class Lemmatizer(object):
    def __init__(self, lang="en"):
        self.lang = lang
        try:
            nlp = stanza.Pipeline(lang=self.lang)
        except:
            stanza.download(lang)
            nlp = stanza.Pipeline(lang=self.lang)
        self.nlp = nlp

    def lemmatize(self, document, remove_punct=True):
        doc = self.nlp(document)
        ret = []
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.pos != "PUNCT" or (not remove_punct and word.pos == "PUNCT"):
                    ret.append(word.lemma)
        return ret


if __name__ == "__main__":
    lemma = Lemmatizer("sv")
    print(
        lemma.lemmatize(
            "Kalle drömmer om självförsörjning, och då behöver man ett automatiskt bevattningssystem."
        )
    )
