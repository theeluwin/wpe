# -*- coding: utf-8 -*-

from utils import Corpus
from wpe import WPE


if __name__ == '__main__':
    corpus = Corpus('data/corpus.txt')
    wpe = WPE(corpus, verbose=True)
    wpe.preprocess()
    wpe.encode()
