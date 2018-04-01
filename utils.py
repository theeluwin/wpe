# -*- coding: utf-8 -*-

import codecs


class Corpus(object):

    def __init__(self, filepath):
        self.source = codecs.open(filepath, 'r', encoding='utf-8')

    def __iter__(self):
        for line in self.source:
            yield line.strip()
