import gensim
import utils

model = gensim.models.KeyedVectors.load_word2vec_format("w2v_of_heb_format_utf8")
lst=['אבי','הי"ו','הצעיר','הראשונים','הרה"ג','ואביו','והנ"י','ז"ל','זי"ע','זיע"א','זלה"ה','זללה"ה','זצ"ל','זצוק"ל','זקני','יצ"ו','כמה"ר','כמוה"ר','מהר"ר','מו"ר','נ"י','נ"ר','נר"ו','סילט"א','ע"ה','שליט"א','תלמידי']
print(utils.one_mean(lst,model))
dist = []
for word in lst:
    temp=model.wv.distances(word,lst)
    temp.sort()
    dist.append((word,temp[-1]))
dist.sort(key=lambda k:k[1])
del dist[-1]
ldist=[e[0] for e in dist]
print(utils.one_mean(ldist,model))
