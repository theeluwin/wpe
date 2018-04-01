# WPE

Word Pair Encoding (WPE) for semi-automatic meaningful-keywords generation.

Of course, just word-level version of Byte Pair Encoding (BPE) but quite optimized for word-level.

Most of implementation ideas were borrowed from [subword-nmt](https://github.com/rsennrich/subword-nmt). Which means fast.

```python
from wpe import WPE
wpe = WPE(["a a b", "a a c"], min_freq=2)
wpe.preprocess()
wpe.encode()  # we get "a a", "b", "c" as keywords!
```
