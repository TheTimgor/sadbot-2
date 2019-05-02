import pickle
import nltk
from nltk.corpus import nps_chat
from nltk.corpus import stopwords
from statistics import mean
from random import choice
import pandas as pd
from scipy.spatial.distance import cosine
import spell

nltk.data.path = ['nltk_data']

try:
    f = open('word_weights.pickle', 'rb')
    word_weights = pickle.load(f)
    f.close()
except FileNotFoundError:
    word_weights = {}

try:
    f = open('categorized_sentences.pickle', 'rb')
    categorized_sentences = pickle.load(f)
    f.close()
except FileNotFoundError:
    categorized_sentences = []


all_words = nltk.FreqDist(w.lower() for w in nps_chat.words())

word_features = [a[0] for a in all_words.most_common()[:2000]]
sentences = [(nltk.word_tokenize(a.text.lower()), a.attrib['class']) for a in nps_chat.xml_posts()]
response_types = {
    'Accept':       ['Statement', 'Emotion', 'Emphasis'],
    'Bye':          ['Bye'],
    'Clarify':      ['Accept', 'Reject', 'Statement', 'Emphasis'],
    'Emotion':      ['Accept', 'Reject', 'Statement', 'Emotion', 'Emphasis'],
    'Continuer':    ['Accept', 'Reject', 'Statement', 'Emphasis'],
    'Emphasis':     ['Accept', 'Reject', 'Statement', 'Emotion', 'Emphasis'],
    'Greet':        ['Greet'],
    'Other':        ['Statement'],
    'Reject':       ['Statement', 'Emotion', 'Emphasis'],
    'Statement':    ['Accept', 'Reject', 'Statement', 'Emotion', 'Emphasis'],
    'System':       ['Statement'],
    'nAnswer':      ['Statement', 'Emotion', 'Emphasis'],
    'whQuestion':   ['Statement'],
    'yAnswer':      ['Accept', 'Reject', 'Statement', 'Emotion', 'Emphasis'],
    'ynQuestion':   ['nAnswer', 'yAnswer', 'Accept', 'Reject']
}


def printr(p):
    print(p)
    return p


def unpack_weights(d):
    unpacked = {}
    for ws in d:
        for w in d[ws]:
            if w not in unpacked:
                unpacked[w] = [d[ws][w]]
            else:
                unpacked[w].append(d[ws][w])
    return {word: mean(unpacked[word]) for word in unpacked}


def sent_similarity(sent1, sent2, weights):
    s1weights = {w: weights[w] for w in weights if w in sent1}
    s2weights = {w: weights[w] for w in weights if w in sent2}
    s1topics = unpack_weights(s1weights)
    s2topics = unpack_weights(s2weights)
    # print(s1topics)
    # print(s2topics)
    s1vector = pd.Series({w: 0 if w not in s1topics else s1topics[w] for w in weights})
    s2vector = pd.Series({w: 0 if w not in s2topics else s2topics[w] for w in weights})
    return 1 - cosine(s1vector, s2vector)


def sentence_features(sentence):
    sentence_words = set(sentence)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in sentence_words)
    return features


def train_classifier():
    featuresets = [(sentence_features(s), c) for (s, c) in sentences]
    train_set, test_set = featuresets[100:], featuresets[:100]
    clas = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(clas, test_set))
    clas.show_most_informative_features(5)
    return clas


try:
    f = open('classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
except FileNotFoundError:
    classifier = train_classifier()
    f = open('classifier.pickle', 'wb')
    f.close()
    pickle.dump(classifier, f)


def get_responses(message, weights=word_weights, sents=categorized_sentences, clas=classifier, threshold=0.4, n=10):
    tkn_message = nltk.word_tokenize(message)
    tkn_message = [spell.correction(w) for w in tkn_message]
    print(tkn_message)
    has_non_stop = False
    for word in tkn_message:
        if word in word_weights:
            has_non_stop = True
    typs = response_types[clas.classify(sentence_features(tkn_message))]
    relevant = []
    i = 0
    if has_non_stop:
        for s in sents:
            if i > 500 and threshold >= 0.1:
                threshold -= 0.05
                i = 0
            i += 1
            sm = sent_similarity(tkn_message, nltk.word_tokenize(s[0]), weights)
            if sm >= threshold and s[1] in typs:
                relevant.append((s, sm))
            if len(relevant) >= n:
                break
    else:
        for s in sents:
            if s[1] in typs:
                relevant.append((s, 0))
            if len(relevant) >= n:
                break

    relevant.sort(key=lambda x: x[1])
    if relevant:
        return relevant[:10]
    else:
        return []


def get_response(message, weights=word_weights, sents=categorized_sentences, clas=classifier, threshold=0.4, n=3):
    responses = get_responses(message, weights, sents, clas, threshold, n)
    return choice(responses)[0][0]


def generate_word_weights(hist):
    stop_words = stopwords.words('english')
    filtered_sentences = [[b for b in nltk.word_tokenize(a.lower()) if b not in stop_words] for a in hist]
    non_stop_words = list(set([b for a in filtered_sentences for b in a]))
    # print(non_stop_words)
    word_counts = {a: {b: [0, 0] for b in non_stop_words} for a in non_stop_words}
    count = 0
    for sent in filtered_sentences:
        # print(count/len(filtered_sentences))
        count += 1
        for word in sent:
            for wword in non_stop_words:
                word_counts[word][wword][0] += 1
                if wword in sent:
                    word_counts[word][wword][1] += 1

    weights = {a: {b: word_counts[a][b][1] / word_counts[a][b][0] for b in word_counts[a] if word_counts[a][b][1] > 0} for a in word_counts}
    return weights


def generate_model(hist):
    global word_weights
    global categorized_sentences
    spell.setWORDS(nltk.word_tokenize('\n'.join(hist)))
    word_weights = generate_word_weights(hist)
    f = open('word_weights.pickle', 'wb')
    pickle.dump(word_weights, f)
    f.close()
    categorized_sentences = [(a, classifier.classify(sentence_features(nltk.word_tokenize(a)))) for a in hist]
    f = open('categorized_sentences.pickle', 'wb')
    pickle.dump(categorized_sentences, f)
    f.close()
    print(word_weights)
    print(categorized_sentences)


if __name__ == '__main__':
    h = [a.text for a in nps_chat.xml_posts()]
    if not word_weights:
        print('weighting words')
        word_weights = generate_word_weights(h)
        f = open('word_weights.pickle', 'wb')
        pickle.dump(word_weights, f)
        f.close()
    if not categorized_sentences:
        print('categorizing sentences')
        categorized_sentences = [(a.text, a.attrib['class']) for a in nps_chat.xml_posts()]
        f = open('categorized_sentences.pickle', 'wb')
        pickle.dump(categorized_sentences, f)
        f.close()
    # categorized_sentences = [(a, classifier.classify(sentence_features(nltk.word_tokenize(a)))) for a in h]
    # print(get_responses('hello'))
    spell.setWORDS(nltk.word_tokenize('\n'.join(h)))
    while True:
        m = input('>>> ')
        print(get_response(m))
