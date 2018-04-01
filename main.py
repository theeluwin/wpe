# -*- coding: utf-8 -*-

from utils import Corpus
from wpe import WPE


if __name__ == '__main__':
    corpus = Corpus('data/corpus.txt')
    wpe = WPE(corpus, min_freq=2, verbose=True)
    wpe.preprocess()
    wpe.encode()
