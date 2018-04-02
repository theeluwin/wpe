# -*- coding: utf-8 -*-

import os
import copy
import codecs
import numpy as np

from collections import defaultdict


class WPE(object):  # Word Pair Encoding. Most ideas were borrowed from https://github.com/rsennrich/subword-nmt/blob/master/learn_bpe.py

    PRUNE_EVERY = 100
    PAD = 10000

    def __init__(self, corpus, result='./result/', delimiter=' ', merger='$$', min_iter=100, max_iter=500, min_freq=10, window=3, convergence=1e-4, cut=None, verbose=False):
        self.corpus = [line.strip() for line in corpus]
        self.result = result
        self.delimiter = delimiter
        self.merger = merger
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.min_freq = min_freq
        self.window = window
        self.convergence = convergence
        self.verbose = verbose
        self.cut = cut
        if not os.path.isdir(self.result):
            os.mkdir(self.result)

    def preprocess(self):
        self.stats = defaultdict(int)
        self.indices = defaultdict(lambda: defaultdict(int))
        for i, line in enumerate(self.corpus):
            tokens = line.split(self.delimiter)
            for j in range(len(tokens) - 1):
                pair = (tokens[j], tokens[j + 1])
                self.stats[pair] += 1
                self.indices[pair][i] += 1
        self.threshold = max(self.stats.values()) / 10
        self.full = copy.deepcopy(self.stats)

    def prune(self):
        for item, freq in list(self.stats.items()):
            if freq < self.threshold:
                del self.stats[item]
                if freq < 0:
                    self.full[item] += freq
                else:
                    self.full[item] = freq

    def replace_paired(self, pair):
        source = self.delimiter.join(pair)
        target = self.merger.join(pair)
        changes = []
        for i, freq in self.indices[pair].items():
            if freq < 1:
                continue
            oldline = self.corpus[i]
            newline = oldline.replace(source, target)
            self.corpus[i] = newline
            changes.append((i, oldline, newline, freq))
        return changes

    def update(self, pair, changed):
        self.stats[pair] = 0
        self.indices[pair] = defaultdict(int)
        first, second = pair
        flat = self.merger.join([first, second])
        for i, oldline, newline, freq in changed:
            tokens = oldline.split(self.delimiter)
            len_tokens = len(tokens)
            j = 0
            while True:
                if j == len_tokens:
                    break
                if tokens[j] != first:
                    j += 1
                    continue
                if j < len_tokens - 1 and tokens[j + 1] == second:
                    if j:
                        prev = tuple(tokens[j - 1: j + 1])
                        self.stats[prev] -= freq
                        self.indices[prev][i] -= 1
                    if j < len_tokens - 2:
                        if tokens[j + 2] != first or j >= len_tokens - 3 or tokens[j + 3] != second:
                            next = tuple(tokens[j + 1: j + 3])
                            self.stats[next] -= freq
                            self.indices[next][i] -= 1
                    j += 2
                else:
                    j += 1
            tokens = newline.split(self.delimiter)
            len_tokens = len(tokens)
            j = 0
            while True:
                if j == len_tokens:
                    break
                if tokens[j] != flat:
                    j += 1
                    continue
                if j:
                    prev = tuple(tokens[j - 1: j + 1])
                    self.stats[prev] += freq
                    self.indices[prev][i] += 1
                if j < len_tokens - 1 and tokens[j + 1] != flat:
                    next = tuple(tokens[j: j + 2])
                    self.stats[next] += freq
                    self.indices[next][i] += 1
                j += 1

    def flatten(self, word):
        return word.replace(self.merger, self.delimiter)

    def curvature(self, diffs):
        return abs(diffs[-1] + diffs[-(2 * self.window + 1)] - 2 * diffs[-(self.window + 1)])

    def encode(self):
        stds = []
        diffs = []
        merge = []
        for i in range(self.max_iter):
            if self.stats:
                pair = max(self.stats, key=self.stats.get)
            if not self.stats or (i and self.stats[pair] < self.threshold):
                self.prune()
                self.stats = copy.deepcopy(self.full)
                pair = max(self.stats, key=self.stats.get)
                self.threshold = self.stats[pair] * i / (i + self.PAD)
                self.prune()
            if self.stats[pair] < self.min_freq:
                print("no pair with freq >= {}. early stopping...".format(self.min_freq))
                break
            if self.verbose:
                print("[iter {:3d}] freq {:5d}\t{}".format(i, self.stats[pair], ' + '.join([self.flatten(word) for word in pair])))
            merge.append(pair)
            changes = self.replace_paired(pair)
            self.update(pair, changes)
            self.stats[pair] = 0
            if not i % self.PRUNE_EVERY:
                self.prune()
            stds.append(np.std(list([freq for freq in self.stats.values() if freq])))
            if len(stds) > 2:
                diffs.append(stds[-1] - stds[2])
            if len(diffs) > (2 * self.window + 1):
                if i > self.min_iter and self.curvature(diffs) < self.convergence:
                    print("std reached elbow point. early stopping...")
                    break
        with codecs.open(os.path.join(self.result, 'merge.txt'), 'w', encoding='utf-8') as out:
            out.write('\n'.join([' '.join(pair) for pair in merge]))
        dc = {}
        for line in self.corpus:
            tokens = line.split(self.delimiter)
            for token in tokens:
                token = self.flatten(token)
                dc[token] = dc.get(token, 0) + 1
        keywords = sorted(dc, key=dc.get, reverse=True)
        if self.cut:
            keywords = keywords[:self.cut]
        with codecs.open(os.path.join(self.result, 'keywords.txt'), 'w', encoding='utf-8') as out:
            out.write('\n'.join(keywords))
        if self.verbose and len(stds) > (2 * self.window + 1):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            x = np.array([i for i in range(len(stds))])
            y = np.array(stds)
            plt.clf()
            plt.grid(True)
            plt.plot(x, y)
            plt.savefig(os.path.join(self.result, 'std.png'))
