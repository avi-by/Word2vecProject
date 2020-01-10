import gensim
import codecs
from datetime import datetime


#class to pass to gensim large file that need encode (most case - non english text)
class SentenceIterator:
    def __init__(self, filepath,encodeType):
        self.filepath = filepath
        self.encode = encodeType

    def __iter__(self):
        for line in codecs.open(self.filepath,encoding=self.encode):
            yield line.split()

#creat the object, replace example.txt withe your file path, if its in the script folder the file name is enough
#enter your file codec, in my example its utf 8, but it can be utf-16-le or something else
print("strat")
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

sentences = SentenceIterator("all.txt","utf-16-le")

#train your model with CBOW algorithm - make yourself cup of coffee or whats you like and wait
model = gensim.models.Word2Vec(sentences,workers=4)
print("end")
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

#train the model with skip-gram algorithm
#modelSg = gensim.models.Word2Vec(sentences,sg=1)

#save your file in the current dir
#model.wv.save_word2vec_format("w2v_of_heb_format_utf8")
model.save("w2v_1_heb")
model.wv.save_word2vec_format("w2v_1_format")

