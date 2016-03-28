from itertools import islice
import json
import sys
import time
try:
    import cPickle as pickle
    import cStringIO as StringIO
except ImportError:
    import pickle
    import StringIO

import numpy as np

# pass the path of the JSON file as first argument
FULL_PATH = sys.argv[1]

def doc_iter(limit=None):
    with open(FULL_PATH) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break

            yield json.loads(line)['text']

def transformer_iter(cv, lsa, batch_size=1000):
    iterable = doc_iter()
    data = list(islice(iterable, batch_size))

    while len(data) != 0:
        yield lsa.transform(cv.transform(data))
        data = list(islice(iterable, batch_size))

def printer(iterable):
    for batch in iterable:
        f = StringIO.StringIO()
        np.savetxt(f, batch, delimiter=',', fmt='%.6e')
        print f.getvalue()[:-1]

if __name__ == '__main__':
    with open('review_bow.pkl', 'rb') as f:
        cv = pickle.load(f)

    with open('lsa.pkl', 'rb') as f:
        lsa = pickle.load(f)

    printer(transformer_iter(cv, lsa))
