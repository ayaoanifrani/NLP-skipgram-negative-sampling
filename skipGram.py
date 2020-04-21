from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize
from time import time
from collections import Counter
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en.lex_attrs import _num_words

__authors__ = ['Ayao Anifrani', 'Thomas Fraisse']
__emails__ = ['ayao.anifrani@student.ecp.fr', 'thomas.fraisse@student.ecp.fr']


def text2sentences(path):
    sentences = []
    with open(path, encoding='latin1') as f:
        for l in f:
            sentence = l.lower().split()
            sentence_clean = []
            for word in sentence:
                precedent_was_num = False  # idea is to avoid successions of <NUMBER> token
                # (for example from "twenty one" --> "<NUMBER>", "<NUMBER>" --> "<NUMBER>")

                # Split composite words like "35-year-old"
                if ('-' in word) and (len(word) > 1):
                    split = word.split('-')
                    for item in split:
                        if (item.isalpha()) and (item not in STOP_WORDS):
                            sentence_clean.append(item)
                            precedent_was_num = False
                        elif (item.isnumeric()) and (not precedent_was_num):
                            sentence_clean.append('<NUMBER>')  # Using a unique token to replace all numbers
                            precedent_was_num = True

                elif (word.isalpha()) and (word not in STOP_WORDS):
                    sentence_clean.append(word)
                    precedent_was_num = False
                elif word in ["'t", "'nt", "nt"]:
                    sentence_clean.append('<NEGATIVE>')  # Using a unique token for negative form
                    precedent_was_num = False
                elif ((word.isnumeric()) or (word in _num_words)) and (not precedent_was_num):
                    sentence_clean.append('<NUMBER>')  # Using a unique token to replace all numbers
                    precedent_was_num = True
            if (len(sentence_clean) > 0) and (np.unique(sentence_clean).tolist() != ['<NUMBER>']):
                sentences.append(sentence_clean)
    return sentences


def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return pairs


class SkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize=5, minCount=5):
        self.w2id = {}  # word to ID mapping
        self.trainset = sentences  # set of sentences
        words = [word for sentence in sentences for word in sentence]  # words in vocab with their multiplicity
        self.vocab = sorted(set(words))  # list of valid words
        vocab_size = len(self.vocab)
        word_count = dict(
            Counter(words))  # we used Counter from collections package to get the counts/frequency of words in the set
        self.freq = np.array(
            [word_count[word] / vocab_size for word in self.vocab])  # useful for the computation of the unigram table
        self.nEmbed = nEmbed
        self.negativeRate = negativeRate
        self.winSize = winSize
        self.minCount = minCount  # minimum required number of words in a sentence
        self.accLoss = 0
        self.trainWords = 0
        self.loss = []  # list of loss function evolution. Could be usefull to plot

        self.centerVecs = (np.random.rand(vocab_size, nEmbed) - 0.5) / nEmbed
        self.contxtVecs = (np.random.rand(vocab_size, nEmbed) - 0.5) / nEmbed

        # Updating w2id
        for i, word in enumerate(self.vocab):
            self.w2id[word] = i

        # Computing unigram table
        self.unigram_table = self.compute_unigram_table()

    def compute_unigram_table(self, power=3 / 4, table_length=int(1e8)):
        """computing the unigram table"""
        vocab_size = len(self.vocab)
        denominator = sum(
            [np.power(self.freq[i], power) for i in range(vocab_size)])  # normalization constant in probability
        table = np.array(np.zeros(table_length), dtype='int')
        p = 0  # Cumulative probability
        counter = 0  # counter for list filling
        for i in range(vocab_size):
            p += np.power(self.freq[i], power) / denominator
            while (counter < table_length) and (counter / table_length < p):
                table[counter] = i
                counter += 1
        np.random.shuffle(table)
        return table

    def sample(self, omit):
        """samples negative words, ommitting those in set omit"""
        counter = 0
        negWordsId = []
        while counter < self.negativeRate:
            negWordId = np.random.choice(self.unigram_table)
            if negWordId not in omit:  # omitting words in omit
                negWordsId.append(negWordId)
                counter += 1
        return negWordsId

    def train(self, n_epoch=5, lr=0.05):
        # lr is learning rate
        t0 = time()
        for epoch in range(n_epoch):
            print("\n epoch: %d of %d" % (epoch + 1, n_epoch))
            for counter, sentence in enumerate(self.trainset):
                if len(sentence) < self.minCount:
                    if counter % 1000 == 0:
                        t1 = time()
                        self.loss.append(self.accLoss / self.trainWords)
                        hours, rem = divmod(t1 - t0, 3600)
                        minutes, seconds = divmod(rem, 60)
                        timer = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes),
                                                                seconds)  # print progression time in hh:mm:ss format
                        print(' > training %d of %d ----- time = ' % (
                            counter, len(self.trainset)) + timer + ' ----- loss: %.3f' % self.loss[-1])
                        self.trainWords = 0
                        self.accLoss = 0
                        continue

                for wpos, word in enumerate(sentence):
                    wIdx = self.w2id[word]
                    winsize = np.random.randint(self.winSize) + 1
                    start = max(0, wpos - winsize)
                    end = min(wpos + winsize + 1, len(sentence))

                    for context_word in sentence[start:end]:
                        ctxtId = self.w2id[context_word]
                        if ctxtId == wIdx: continue
                        negativeIds = self.sample({wIdx, ctxtId})
                        self.trainWord(wIdx, ctxtId, negativeIds, lr)
                        self.trainWords += 1

                if counter % 1000 == 0:
                    t1 = time()
                    self.loss.append(self.accLoss / self.trainWords)
                    hours, rem = divmod(t1 - t0, 3600)
                    minutes, seconds = divmod(rem, 60)
                    timer = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes),
                                                            seconds)  # print progression time in hh:mm:ss format
                    print(' > training %d of %d ----- time = ' % (
                        counter, len(self.trainset)) + timer + ' ----- loss: %.3f' % self.loss[-1])
                    self.trainWords = 0
                    self.accLoss = 0

    def trainWord(self, wordId, contextId, negativeIds, lr):
        vec = self.centerVecs[wordId]
        ctxtvec = self.contxtVecs[contextId]
        negvecs = self.contxtVecs[negativeIds]
        z = expit(-np.dot(ctxtvec, vec))
        zNeg = - expit(np.dot(negvecs, vec))

        # Computing gradients
        contextGrad = z * vec  # to be multiplied by -1 because of the definition of z
        centerGrad = z * self.contxtVecs[contextId] + np.dot(zNeg,
                                                             negvecs)  # to be multiplied by -1 because of the definition of z and zNeg
        negativesGrad = np.outer(zNeg, vec)  # to be multiplied by -1 because of the definition of zNeg

        # Gradient descent step
        np.add(vec, centerGrad * lr, out=vec);
        np.add(ctxtvec, contextGrad * lr, out=ctxtvec);
        np.add(negvecs, negativesGrad * lr, out=negvecs);

        # Computing loss
        z = expit(np.dot(ctxtvec, vec))
        zNeg = expit(-np.dot(negvecs, vec))
        self.accLoss -= np.log(z) + np.sum(np.log(zNeg))

        # Update of embeddings
        self.centerVecs[wordId] = vec
        self.contxtVecs[contextId] = ctxtvec
        self.contxtVecs[negativeIds] = negvecs

    def save(self, path):
        """Path must have a .zip extension"""
        import os
        from zipfile import ZipFile, ZIP_DEFLATED
        from json import dumps

        # If path is not a zip file we are going to change it
        if "." in path:
            filename_split = path.split('.')
            if filename_split[-1] != "zip":
                filename_split[-1] = "zip"
                path = ".".join(filename_split)
        else:
            path += "/skipgram.zip"

        zf = ZipFile(path, mode="w", compression=ZIP_DEFLATED)
        # Parameters of the model
        model_info = dumps(
            {
                "nEmbed": self.nEmbed,
                "negativeRate": self.negativeRate,
                "winSize": self.winSize,
                "minCount": self.minCount,
                "w2id": self.w2id
            }, indent=4)
        trainset = dumps(self.trainset, indent=4)
        vocab = dumps(self.vocab, indent=4)

        zf.writestr("model_info.json", model_info)
        zf.writestr("trainset.json", trainset)
        zf.writestr("vocab.json", vocab)

        # Saving embeddings
        np.save("centerVecs.npy", self.centerVecs)
        zf.write("centerVecs.npy")
        os.remove("centerVecs.npy")
        np.save("contxtVecs.npy", self.contxtVecs)
        zf.write("contxtVecs.npy")
        os.remove("contxtVecs.npy")
        np.save("freq.npy", self.freq)
        zf.write("freq.npy")
        os.remove("freq.npy")
        np.save("loss.npy", self.loss)
        zf.write("loss.npy")
        os.remove("loss.npy")

        zf.close()

    def similarity(self, word1, word2):
        """
          computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        word1, word2 = word1.lower(), word2.lower()
        if (word1 in self.vocab) and (word2 in self.vocab):
            vec1 = self.centerVecs[self.w2id[word1]]
            vec2 = self.centerVecs[self.w2id[word2]]
            # cosine similarity
            cos_similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            score = np.clip(cos_similarity, 0, 1)
        else:
            score = np.random.rand()  # random score if the word is not in vocabulary (uniform in [0,1])
        if score <= 1e-4:
            score = 0
        return score

    @staticmethod
    def load(path):
        from json import loads
        from io import BytesIO
        from zipfile import ZipFile

        try:
            zf = ZipFile(path, "r")
        except FileNotFoundError:
            path = path.split('.')
            path[-1] = 'zip'
            path = '.'.join(path)
            zf = ZipFile(path, "r")

        model_info = loads(zf.read("model_info.json"))
        trainset = loads(zf.read('trainset.json'))
        vocab = loads(zf.read('vocab.json'))

        skipgram = SkipGram(trainset, model_info['nEmbed'], model_info['negativeRate'],
                            model_info['winSize'], model_info['minCount'])
        skipgram.vocab = vocab
        skipgram.centerVecs = np.load(BytesIO(zf.read('centerVecs.npy')))
        skipgram.contxtVecs = np.load(BytesIO(zf.read('contxtVecs.npy')))
        skipgram.loss = np.load(BytesIO(zf.read('loss.npy'))).tolist()
        skipgram.freq = np.load(BytesIO(zf.read('freq.npy')))
        # No need to cumpute the unigram table, it is computed at the initialization.
        # This is important because the table takes up to 700MB when saved. (lenght is 1e8)

        zf.close()
        return skipgram


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences)
        sg.train(n_epoch=5, lr=0.05)  # change n_epoch to use the training set more
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = SkipGram.load(opts.model)
        for a, b, _ in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
            print(sg.similarity(a, b))
