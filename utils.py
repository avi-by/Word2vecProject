import numpy
import gensim


def avg_vec(a):
    avg = numpy.average(a, axis=0)
    arr = numpy.asarray(avg)
    return arr


def avg_vec_model(lst, model, norm=False):
    lst_vec = []
    for e in lst:
        lst_vec.append(model.wv.word_vec(e, use_norm=norm))
    avg = numpy.average(lst_vec, axis=0)
    arr = numpy.asarray(avg)
    return arr



def one_mean(lst, model, norm=False):
    avg = avg_vec_model(lst, model, norm)
    dist=my_distances(avg,lst,model)
    dist.sort()
    dist = dist[0]
    i = 100
    while model.similar_by_vector(avg, topn=i)[-1][1] > dist:
        i *= 2
    return [(word, distance) for (word, distance) in model.similar_by_vector(avg, topn=i) if distance >= dist]


def normal(vec):
    return vec / numpy.linalg.norm(vec)


def normal_met(metrix):
    res = []
    for element in metrix:
        res.append(normal(element))
    return res


def name_list(vec):
    res = []
    for element in vec:
        res.append(element[0])
    return res


def print_differents(lst1, lst2):
    print("---- in list1 and not in list2 vec ----")
    counter = 0
    for a in name_list(lst1):
        counter += 1
        if a not in name_list(lst2):
            print(a, counter)
    print("----- in list2 and not in list1 vec ----")
    counter = 0
    for a in name_list(lst2):
        counter += 1
        if a not in name_list(lst1):
            print(a, counter)
    return


def my_similarity(v1,v2):
    """
    the method that gensim use when calculate most similaruty
    found at the github of gesim
    """
    return numpy.dot(gensim.matutils.unitvec(v1), gensim.matutils.unitvec(v2))


def my_distances(vec,lst,model):
    res=[]
    for i in lst:
        res.append(my_similarity(vec, model.wv.get_vector(i)))
    return res


def my_var(lst,model):
    a = numpy.array([])
    for i in lst:
        a = numpy.append(a, model.wv.get_vector(i))
    a = a.reshape(len(lst), model.vector_size)
    res = numpy.var(a, axis=0)
    nres=[]
    for e in range(len(res)):
        nres.append([e,res[e]])
    return nres


def remove_dim(lst,model,dim):
    arr=[]
    var_lst=(my_var(lst,model))
    var_lst.sort(key=lambda k:k[1])
    for i in var_lst[:dim]:
        arr.append(i[0])
    arr.sort()
    vectors=model.wv.vectors[:,arr]
    model.wv.vectors=vectors
    model.wv.vectors_norm=None
    model.wv.init_sims()


"""

templst=lst
print(templst)
for i in range(len(lst)):
    l1=utils.one_mean(templst,m56)
    ln1=utils.one_mean(templst,n1)
    print("round number: "+str(i)+"\nnumber of element in the list: "+str(len(templst)))
    print("\n=====\n\nin the 300dim vectors:")
    print("number of word in all the vocab: "+str(len(m56.wv.vocab)))
    print("number of word in the result: "+str(len(l1)))
    print("element in the list/word in the vocab: "+str(len(l1)/len(m56.wv.vocab)))
    print("first 20 similar:")
    res=m56.wv.similar_by_vector(utils.avg_vec_model(templst,m56),topn=20)
    for i in range(20):
        print(str(i+1)+". \t"+res[i][0]+" \tsimilarity: "+str(res[i][1]))
    print("\n+++++\n\nin the 100dim vectors:")
    print("number of word in all the vocab: "+str(len(n1.wv.vocab)))
    print("number of word in the result: "+str(len(ln1)))
    print("element in the list/word in the vocab: "+str(len(ln1)/len(n1.wv.vocab)))
    print("first 20 similar:")
    res=n1.wv.similar_by_vector(utils.avg_vec_model(templst,n1),topn=20)
    for i in range(20):
        print(str(i+1)+". \t"+res[i][0]+" \tsimilarity: "+str(res[i][1]))
    dist = []
    for word in templst:
        temp=utils.my_similarity(m56.wv.get_vector(word),utils.avg_vec_model(templst,m56))
        dist.append((word,temp))
    dist.sort(key=lambda k:k[1])
    print("\n\n\ncosim of the words and the avg vec")
    print(dist)
    print("\n\ndelete the word:\n"+dist[0][0]+"\ncosim from avg is: "+str(dist[0][1])+"\n\n------------------------------------------------------------------------------------------\n\n------------------------------------------------------------------------------------------\n\n")
    del dist[0]
    ldist=[e[0] for e in dist]
    templst=ldist

templst=lst
for i in range(len(lst)):
    l1=utils.one_mean(templst,m56)
    ln1=utils.one_mean(templst,n1)
    print("round number: "+str(i)+"\nnumber of element in the list: "+str(len(templst)))
    print("\n=====\n\nin the 300dim vectors:")
    print("number of word in all the vocab: "+str(len(m56.wv.vocab)))
    print("number of word in the result: "+str(len(l1)))
    print("element in the list/word in the vocab: "+str(len(l1)/len(m56.wv.vocab)))
    print("\n+++++\n\nin the 100dim vectors:")
    print("number of word in all the vocab: "+str(len(n1.wv.vocab)))
    print("number of word in the result: "+str(len(ln1)))
    print("element in the list/word in the vocab: "+str(len(ln1)/len(n1.wv.vocab)))
    dist = []
    for word in templst:
        temp=utils.my_similarity(m56.wv.get_vector(word),utils.avg_vec_model(templst,m56))
        dist.append((word,temp))
    dist.sort(key=lambda k:k[1])
    print("\n\n\ncosim of the words and the avg vec")
    print(dist)
    print("\n\ndelete the word:\n"+dist[0][0]+"\ncosim from avg is: "+str(dist[0][1])+"\n\n------------------------------------------------------------------------------------------\n\n------------------------------------------------------------------------------------------\n\n")
    del dist[0]
    ldist=[e[0] for e in dist]
    templst=ldist

"""

