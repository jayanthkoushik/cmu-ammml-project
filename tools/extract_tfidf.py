from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import dok_matrix
import numpy as np

VOCAB_SIZE = 7987
TRANSC_SIZE = 277

with open('data/feats/transc.txt', 'r') as file:
    counts = dok_matrix((TRANSC_SIZE, VOCAB_SIZE), dtype=np.float32)
    doc = 0
    for line in file:
        line = line.strip()
        items = line.split()
        filename = items[0]
        label = items[2]
        words = items[3:]
        for word in words:
            counts[doc,int(word)] += 1
        doc += 1
    transformer = TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=True)
    print "Transforming"
    tfidf_counts = transformer.fit_transform(counts)
    with open('data/feats/tfidf.txt', 'w') as write_file:
        for transc in range(TRANSC_SIZE):
            sparse_tfidf_counts = [str(tfidf_counts[transc, i]) for i in range(VOCAB_SIZE)]
            write_file.write(label + " " + " ".join(sparse_tfidf_counts))
            write_file.write("\n")

        
        


