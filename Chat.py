import pickle
import nltk
from nltk.corpus import nps_chat
from nltk.corpus import stopwords
from statistics import mean
from random import choice, choices
import pandas as pd
from scipy.spatial.distance import cosine
from spell import correct
from augmented_markov import AugmentedChain
import matplotlib.pyplot as plt

# change nltk data path
nltk.data.path = ['nltk_data']

# load up markov model if found
try:
    f = open('markov_model.pickle', 'rb')
    m_model = pickle.load(f)
    f.close()
except FileNotFoundError:
    m_model = None

# load up word weights if found
try:
    f = open('word_weights.pickle', 'rb')
    word_weights = pickle.load(f)
    f.close()
except FileNotFoundError:
    word_weights = {}

# load up categorized sentences if found
try:
    f = open('categorized_sentences.pickle', 'rb')
    categorized_sentences = pickle.load(f)
    f.close()
except FileNotFoundError:
    categorized_sentences = []

# load up categorized sentences if found
try:
    f = open('sentence_clusters.pickle', 'rb')
    sentence_clusters= pickle.load(f)
    f.close()
except FileNotFoundError:
    sentence_clusters = []


# preprocessing nps chat corpus for sentence classification
all_words = nltk.FreqDist(w.lower() for w in nps_chat.words())
word_features = [a[0] for a in all_words.most_common()[:2000]]
sentences = [(nltk.word_tokenize(a.text.lower()), a.attrib['class']) for a in nps_chat.xml_posts()]

# logical response types for each input sentence type
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
    'ynQuestion':   ['nAnswer', 'yAnswer']
}


# print and return, handy for use in list comps
def printr(p):
    print(p)
    return p


# average out dict of weights
def unpack_weights(d):
    unpacked = {}
    for ws in d:
        for w in d[ws]:
            if w not in unpacked:
                unpacked[w] = [d[ws][w]]
            else:
                unpacked[w].append(d[ws][w])
    return {word: mean(unpacked[word]) for word in unpacked}


def topic_vectorize_sent(sent, weights):
    # gets a dict of word weights
    sweights = {w: weights[w] for w in weights if w in sent}
    # averages out word
    stopics = unpack_weights(sweights)
    # adds unrelated words back in
    return {w: 0 if w not in stopics else stopics[w] for w in weights}


# returns the similarity of two sentences
def sent_similarity(sent1, sent2, weights):
    # vectorize if not already
    s1vector = topic_vectorize_sent(sent1, weights) if not type(sent1) == dict else sent1
    s2vector = topic_vectorize_sent(sent2, weights) if not type(sent2) == dict else sent2
    # return similarity
    return 1 - cosine(pd.Series(s1vector), pd.Series(s2vector))


# vectorizes sentences for classifier
def sentence_features(sentence):
    sentence_words = set(sentence)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in sentence_words)
    return features


# trains sentence classifier
def train_classifier():
    featuresets = [(sentence_features(s), c) for (s, c) in sentences]
    train_set, test_set = featuresets[100:], featuresets[:100]
    clas = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(clas, test_set))
    clas.show_most_informative_features(5)
    return clas


# loads classifier if it exists
try:
    f = open('classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
# generates classifier if it does not exist, and dumps it to file
except FileNotFoundError:
    classifier = train_classifier()
    f = open('classifier.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()


def show_clusters(sents):
    l = [x[0] for x in choices(sents, k=1000)]
    s = nltk.word_tokenize(choice(sents)[0])
    d = []
    for a in l:
        sim = sent_similarity(nltk.word_tokenize(a), s, word_weights)
        d.append(sim)
    plt.hist(d, bins=20)
    plt.show()


def generate_clusters(sents=categorized_sentences, threshold=0.2, weights=word_weights):
    clusters = []
    while sents:
        s = topic_vectorize_sent(nltk.word_tokenize(choice(sents)[0]), weights)
        clusters.append((s, {}))
        for sent, clas in sents:
            svec = topic_vectorize_sent(nltk.word_tokenize(sent), weights)
            sim = sent_similarity(s, svec, weights)
            if sim >= threshold:
                clusters[-1][1][sent] = (svec, clas)
                sents.remove((sent, clas))
        print('s', len(sents))
        print('c', len(clusters))
    return clusters


# gets a list of valid responses
# TODO: make this faster
def get_responses(message, weights=word_weights, sents=categorized_sentences, clas=classifier, threshold=0.4, n=10):
    # pre tokenize and spelling correct message
    tkn_message = nltk.word_tokenize(correct(message))
    print(tkn_message)
    has_non_stop = False
    # check if message has any words that are in the dictionary
    for word in tkn_message:
        if word in word_weights:
            has_non_stop = True
    # get response types
    typs = response_types[clas.classify(sentence_features(tkn_message))]
    print(typs)
    relevant = []
    i = 0
    # if some words are in the dictionary, look for a sentence with sufficient similarity
    if has_non_stop:
        for s in sents:
            # decrement threshold every 500 words to speed up search
            if i > 500 and threshold >= 0.1:
                threshold -= 0.05
                i = 0
            i += 1
            sm = sent_similarity(tkn_message, nltk.word_tokenize(s[0]), weights)
            if sm >= threshold and s[1] in typs:
                relevant.append((s, sm))
            if len(relevant) >= n:
                break
    # if no dictionary words are found, give up and just look for a sentence with the right type
    else:
        for s in sents:
            if s[1] in typs:
                relevant.append((s, 0))
            if len(relevant) >= n:
                break

    # sort by relevance
    relevant.sort(key=lambda x: x[1])
    if relevant:
        return relevant[:10]
    else:
        return []


# gets a single response
def get_response(message, weights=word_weights, model=m_model):
    v = topic_vectorize_sent(message, weights)
    print({a:v[a] for a in v if v[a]>0})
    return model.make_sentece(v)


# generates a dict of probabilities that any given word will appear in a sentence with another word
def generate_word_weights(hist):
    # get all words that aren't stopwords
    stop_words = stopwords.words('english')
    filtered_sentences = [[b for b in nltk.word_tokenize(a.lower()) if b not in stop_words] for a in hist]
    non_stop_words = list(set([b for a in filtered_sentences for b in a]))
    word_counts = {a: {b: [0, 0] for b in non_stop_words} for a in non_stop_words}
    # get counts for each word pair
    count = 0
    for sent in filtered_sentences:
        count += 1  # I don't know what this does but I refuse to touch it
        for word in sent:
            for wword in non_stop_words:
                word_counts[word][wword][0] += 1
                if wword in sent:
                    word_counts[word][wword][1] += 1

    weights = {a: {b: word_counts[a][b][1] / word_counts[a][b][0] for b in word_counts[a] if word_counts[a][b][1] > 0} for a in word_counts}
    return weights


# generates word probability weights and categorized sentences
# dumps word weights and categorized sentences to file
def generate_model(hist):
    global word_weights
    global categorized_sentences
    global m_model
    m_model = AugmentedChain(hist)
    f = open('markov_model.pickle', 'wb')
    pickle.dump(m_model, f)
    f.close()
    word_weights = generate_word_weights(hist)
    f = open('word_weights.pickle', 'wb')
    pickle.dump(word_weights, f)
    f.close()
    categorized_sentences = [(a, classifier.classify(sentence_features(nltk.word_tokenize(a)))) for a in hist]
    f = open('categorized_sentences.pickle', 'wb')
    pickle.dump(categorized_sentences, f)
    f.close()


# simple main function for testing
if __name__ == '__main__':
    # using nps chat for testing
    h = [a.text for a in nps_chat.xml_posts()]
    # generate markov model if not loaded
    if not m_model:
        print('generating markov model')
        m_model = AugmentedChain(h[:10000])
        f = open('markov_model.pickle', 'wb')
        pickle.dump(m_model, f)
        f.close()
    # get word weights if they weren't loaded
    if not word_weights:
        print('weighting words')
        word_weights = generate_word_weights(h)
        f = open('word_weights.pickle', 'wb')
        pickle.dump(word_weights, f)
        f.close()
    # get categorized sentences if they weren't loaded
    if not categorized_sentences:
        print('categorizing sentences')
        categorized_sentences = [(a.text, a.attrib['class']) for a in nps_chat.xml_posts()]
        f = open('categorized_sentences.pickle', 'wb')
        pickle.dump(categorized_sentences, f)
        f.close()
    # converse!
    while True:
        m = input('>>> ')
        v = topic_vectorize_sent(nltk.word_tokenize(m), word_weights)
        print(m_model.make_sentece(v))
