# -*- coding: utf-8 -*-

import codecs

from wpe import WPE


class Corpus(object):

    def __init__(self, filepath):
        self.source = codecs.open(filepath, 'r', encoding='utf-8')

    def __iter__(self):
        for line in self.source:
            yield line.strip()


if __name__ == '__main__':
    corpus = Corpus('data/corpus.txt')
    wpe = WPE(corpus, verbose=True, min_freq=2)
    wpe.preprocess()
    wpe.encode()
