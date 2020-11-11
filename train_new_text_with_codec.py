import gensim
import codecs
from datetime import datetime


# class to pass to gensim large file that need encode (most case - not english text)
class SentenceIterator:
    def __init__(self, filepath,encodeType):
        self.filepath = filepath
        self.encode = encodeType

    def __iter__(self):
        for line in codecs.open(self.filepath,encoding=self.encode):
            yield line.split()


def train(filepath,codec,savepath,sg=0,epochs=56,size=300,min_count=1,window=5,worker=4, hs=0, negative=5, ns_exponent=0.75,sample=1e-3):
    print("strat")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    sentences = SentenceIterator(filepath, codec)
    # train your model  - make yourself cup of coffee or whats you like and wait
    model = gensim.models.Word2Vec(sentences, sg=sg,epochs=epochs,size=size,min_count=min_count,window=window,workers=worker,hs=hs,negative=negative,ns_exponent=ns_exponent,sample=sample)
    print("end")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    # model.wv.save_word2vec_format("w2v_of_heb_format_utf8")
    model.save(savepath)
    #model.wv.save_word2vec_format(savepath)



