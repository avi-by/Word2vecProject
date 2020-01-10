import gensim
import numpy
from gensim.models import Word2Vec
model = Word2Vec.load("w2v_56_heb.model")
model.save_word2vec_format("w2v_56_format")
