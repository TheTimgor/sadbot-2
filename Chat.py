import asyncio
import pickle
import statistics

import markovify
import nltk
from nltk.corpus import nps_chat

nltk.data.path = ['nltk_data']

all_words = nltk.FreqDist(w.lower() for w in nps_chat.words())
word_features = [a[0] for a in all_words.most_common()[:2000]]
sentences = [(nltk.word_tokenize(a.text.lower()), a.attrib['class']) for a in nps_chat.xml_posts()]

text = '\n\n'.join([a.text for a in nps_chat.xml_posts()[:10000] if 'System' != a.attrib['class']])
text_model = markovify.Text(text)

response_types = {
    'Accept':       (100, ['Statement', 'Emotion', 'Emphasis']),
    'Bye':          (20, ['Bye']),
    'Clarify':      (75, ['Accept', 'Reject', 'Statement', 'Emphasis']),
    'Emotion':      (75, ['Accept', 'Reject', 'Statement', 'Emotion', 'Emphasis']),
    'Continuer':    (100, ['Accept', 'Reject', 'Statement', 'Emphasis']),
    'Emphasis':     (100, ['Accept', 'Reject', 'Statement', 'Emotion', 'Emphasis']),
    'Greet':        (20, ['Greet']),
    'Other':        (100, ['Statement']),
    'Reject':       (100, ['Statement', 'Emotion', 'Emphasis']),
    'Statement':    (200, ['Accept', 'Reject', 'Statement', 'Emotion', 'Emphasis', 'whQuestion', 'ynQuestion']),
    'System':       (200, ['Statement']),
    'nAnswer':      (100, ['Statement', 'Emotion', 'Emphasis']),
    'whQuestion':   (200, ['Statement']),
    'yAnswer':      (100, ['Accept', 'Reject', 'Statement', 'Emotion', 'Emphasis']),
    'ynQuestion':   (20, ['nAnswer', 'yAnswer', 'Accept', 'Reject'])
}


def sentence_features(sentence):
    sentence_words = set(sentence)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in sentence_words)
    return features


def generate_markov_model(hist):
    global text_model
    text_model = markovify.Text('\n'.join(hist))


def train():
    global classifier
    featuresets = [(sentence_features(s), c) for (s, c) in sentences]
    train_set, test_set = featuresets[100:], featuresets[:100]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)

    f = open('my_classifier.pickle', 'wb')
    f.close()
    pickle.dump(classifier, f)


try:
    f = open('my_classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
except FileNotFoundError:
    train()


def get_response(message):
    m_type = classifier.classify(sentence_features(nltk.word_tokenize(message)))
    r_length = response_types[m_type][0]
    r_types = response_types[m_type][1]
    while True:
        response = text_model.make_sentence(tries=100)
        print(response)
        if response:
            r_type = classifier.classify(sentence_features(nltk.word_tokenize(response)))
            if r_type in r_types:
                break
    return response


if __name__ == '__main__':
    s = get_response('hello')
    print(s)
    print(classifier.classify(sentence_features(nltk.word_tokenize(s))))



