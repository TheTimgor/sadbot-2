import nltk
from nltk.corpus import nps_chat
from sacremoses import MosesDetokenizer
import numpy


class AugmentedChain:

    def __init__(self, hist, state_size=2):
        self.model = []
        self.state_size = state_size
        if type(hist) == str:
            hist = hist.split('\n')
        for s in hist:
            sent = nltk.word_tokenize(s)
            sent.insert(0, '__BEGIN__')
            sent.insert(0, '__BEGIN__')
            sent.append('__END__')

            s_model = []
            for i in range(state_size-1, len(sent)-1):
                state = sent[i-state_size+1:i+1]
                s_model.append([state,sent[i+1]])

            for p in s_model:
                new = True
                for m in self.model:
                    if m[0] == p[0]:
                        if p[1] in m[1]:
                            m[1][p[1]] += 1
                        else:
                            m[1][p[1]] = 1
                        new = False
                if new:
                    self.model.append([p[0], {p[1]:1}])

    def make_sentece(self, augment={}, threshold=0.4):
        sent = ['__BEGIN__', '__BEGIN__']
        while sent[-1] != '__END__':
            state = sent[-self.state_size:]
            counts = [a[1] for a in self.model if a[0] == state][0]
            if not counts:
                counts = [a[1] for a in self.model if a[0] == state][0]
            augments = {a:augment[a] if augment[a] >= threshold else 0 for a in augment}
            weights = {a:counts[a]*(augments[a]) if a in augments else counts[a] for a in counts}
            total = sum(weights.values())
            if total == 0:
                weights = {a:1 for a in weights}
                total = sum(weights.values())
            probs = [a/total for a in weights.values()]
            draw = numpy.random.choice(list(counts.keys()), 1, p=probs)[0]
            sent.append(draw)
        detokenizer = MosesDetokenizer()
        return detokenizer.detokenize(sent[2:-1])


if __name__ == '__main__':
    chain = AugmentedChain([a.text for a in nps_chat.xml_posts()][:5000])
    print(chain.model)
    print(chain.make_sentece())

