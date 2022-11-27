import pandas as pd

# dataset imported from
twitters = pd.read_csv(r'C:\Users\Yann\Desktop\gotTwitter.csv', usecols=['text'])

import re


def preprocess(texts, vocabSize=2000):
    ppt = []

    for text in texts:
        # removing everything after @
        filteredText = re.sub(r'@\S+', "", text);
        # removing URLS
        filteredText = re.sub(r'http\S+', '', filteredText)
        filteredText = re.sub(r'http\S+', '', filteredText)
        # removing everything that is not a letter or space
        filteredText = re.sub(r'[^\w ]', u'', filteredText)
        filteredText = re.sub(r'[^a-zA-Z\s]', u'', filteredText, flags=re.UNICODE)
        # removing trailing whitespaces and \n
        filteredText = re.sub(' +', ' ', filteredText)
        filteredText = filteredText.strip()
        filteredText = filteredText.replace('\n', '')
        # lower casing
        ppt.append(filteredText.lower())

    # keeping only the first vocabSize words of the text
    # first finding the vocabSize first elements
    t = ' '.join(ppt)
    t = t.split(' ')
    d = dict()
    for word in t:
        d[word] = 0
    for word in t:
        d[word] += 1
    d = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))
    # making a dictionary of words to keep or to discard (discarded words will be 0 in the dic)
    wordsKept = dict()
    i = 0
    for word in d:
        if i < vocabSize:
            wordsKept[word] = word
        else:
            wordsKept[word] = '0'
        i += 1
    vocabText = []
    for sentence in ppt:
        t = sentence.split(' ')
        s = []
        for word in t:
            s.append(wordsKept[word])
        vocabText.append(' '.join(s))
    return vocabText

def words2int(texts):
    t = ' '.join(texts)
    t = t.split(' ')
    d = dict([(y,x+1) for x,y in enumerate(sorted(set(t)))])
    intTexts = []
    for sentence in texts:
        t = []
        for word in sentence:
            if word == ' ':
                continue
            t.append(d[word])
        intTexts.append(t)
    dS = {v: k for k, v in d.items()}
    return intTexts, d, dS

ppt = preprocess(twitters['text'])