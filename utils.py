import numpy
import gensim
from gensim.models import KeyedVectors
import pandas as ps
import xlwt
from xlwt import Workbook


def avg_vec(a):
    """
    make 1D vector with the avg value of each dimension from matrix
    :param a: matrix of vectors
    :return: 1-D vector of the average
    """
    avg = numpy.average(a, axis=0)
    arr = numpy.asarray(avg)
    return arr


def avg_vec_model(lst, model, norm=False):
    """
    make average 1D vector from list of words and word2vec model
    :param lst: list of word to make from them the avg vector
    :param model: word2vec model to get the original vectors from it
    :param norm: optional - get the normals vectors
    :return:1D vectors of the avg
    """
    lst_vec = []
    for e in lst:
        lst_vec.append(model.wv.word_vec(e, use_norm=norm))
    avg = numpy.average(lst_vec, axis=0)
    arr = numpy.asarray(avg)
    return arr



def one_mean(lst, model, norm=False):
    """
    calculate the avg vector from list of words by word2vec model and calculate the distance from the avg vector to
    every vector of word in the list and use the largest distance to define accepted distance,
    then find all the vectors in the model that in the accepted distance
    :param lst: list of word to make from them the avg vector
    :param model:word2vec model
    :param norm:optional - use normalize vectors
    :return:list of tuple of word and similarity to the avg vector sorted by similarity
    """
    # calculate the avg vector
    avg = avg_vec_model(lst, model, norm)
    # calculate the similarity(distance by cosine similarity metrica) of every word in the list and the avg vector
    dist=my_distances(avg,lst,model,norm)
    # sort the distances from the farthest to the closest and take the first, the farthest
    dist.sort()
    dist = dist[0]
    # find i bigger then the num of the word that we want to include to use in 'topn' parameter
    i = 100
    while model.wv.similar_by_vector(avg, topn=i)[-1][1] > dist:
        i *= 2
    return [(word, distance) for (word, distance) in model.wv.similar_by_vector(avg, topn=i) if distance >= dist]


def normal(vec):
    """
    return normalize vector by Euclidean norm ( L2 norm)
    :param vec: 1D vector
    :return: normalize vector
    """
    return vec / numpy.linalg.norm(vec)


def normal_met(matrix):
    res = []
    for element in matrix:
        res.append(normal(element))
    return res


def name_list(vec):
    """
    separate the words from list of tuples in the form (word,similarity) - gensim most_similarity output
    :param vec: list of tuples in the form (word,similarity)
    :return: list of words
    """
    res = []
    for element in vec:
        res.append(element[0])
    return res


"""
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
"""


def my_similarity(v1,v2):
    """
    the method that gensim use when calculate most similarity
    found at the github of gensim
    """
    return numpy.dot(gensim.matutils.unitvec(v1), gensim.matutils.unitvec(v2))


def my_distances(vec,lst,model,norn=False):
    """
    calculate the similarity (distance by cosine similarity metrica) from the vector and every word in the lst list
    by using the word vector of the word from the model
    :param vec: 1D vector
    :param lst: list of words
    :param model: Word2Vec model that include the words
    :return: list of distances, order by the original words index in the list
    """
    res=[]
    for i in lst:
        res.append(my_similarity(vec, model.wv.word_vec(i,use_norm=norn)))
    return res


def my_var(lst,model,norm=False):
    """
    calculate the variance of each dimension in the word vectors of the words in lst
    and return list of the dimension and its variance
    if norm is true it will use normalized vectors
    for example in the vectors [1,2,3], [4,5,6] , [1,5,9]
    the result will be like this [(variance of (1,4,1),0),(variance of (2,5,5),1),(variance of (3,6,9),2)]
    :param lst: list of words
    :param model: Word2Vec model with the words
    :param norm: optional - if true use normalized vectors
    :return: list of tuple of dimension and its variance (dim,variance) sorted by dim index
    """
    a = numpy.array([])
    for i in lst:
        a = numpy.append(a, model.wv.word_vec(i,use_norm=norm))
    a = a.reshape(len(lst), len(model.wv.word_vec(lst[0])))
    res = numpy.var(a, axis=0)
    var_array=[]
    for e in range(len(res)):
        var_array.append([e,res[e]])
    return var_array


def remove_dim(lst, model, dim_num,norm=False):
    """
    make from the model and the word list (lst) new model with new vectors with only dim_num dimension
    the vector calculate by calculate the variance of the dimension of the word vectors of "lst"
    then take only the dimension with the smaller variance
    the number of the dimension in the result is:
    if dim num is between -1 to 1 it calculate vector_size * dim_num else it the original dim num
    then if dim num > 0  dim num is the number of the dimension in the result
    if  dim num <0 dim num is the number of dimension to subtract
    and drop the (model.wv.vectors_size - dim_num ) dimension with the biggest variance
    :param lst: list of words to calculate the variance
    :param model: Word2Vec model
    :param dim_num: vector size of the new model
    :return: KeyedVectors object with the new vectors
    """
    if dim_num == 0:
        new_model = KeyedVectors(model.wv.vectors.shape[1])
        new_model.add(model.wv.index2word, model.wv.vectors)
        return new_model
    #  if dim num is percent of the vector size
    if -1 < dim_num < 1:
        dim_num =  int(model.wv.vector_size * dim_num)
    if dim_num < 0:
        dim = model.wv.vector_size + dim_num
    else:
        dim = dim_num
    arr=[]
    var_lst=(my_var(lst,model,norm))
    # sort by variance from smaller to bigger and that is also from the similar to the different
    var_lst.sort(key=lambda k:k[1])
    # list of dimension, then list size is dim_num, the most similar dimension
    for i in var_lst[:dim]:
        arr.append(i[0])
    # sort by dimension index
    arr.sort()
    # take all the vector and for each vector take only the elements in the arr list
    # its in fact remove the not necessary dimension from the vectors
    vectors=model.wv.vectors[:,arr]
    new_model = KeyedVectors(vectors.shape[1])
    new_model.add(model.wv.index2word,vectors)
    return new_model


def remove_words_from_lst(lst,model,num=1,norm=False):
    """
    return new word list in size of len(lst)-num of the most similar words in lst
    calculate by the cosine similarity (distance in cosine similarity metrica) to the avg vector
    and remove the "farthest" words
    then calculate the avg vector again (by recursion) and remove the new farthest word and so on
    until the length of the res is right
    :param lst:words array
    :param model:Word2Vec model that include the words
    :param num:number of words to remove
    :return:array of words
    """
    if num <= 0:
        return lst
    dist = []
    for word in lst:
        temp = my_similarity(model.wv.word_vec(word,use_norm=norm), avg_vec_model(lst, model,norm))
        dist.append((word, temp))
    dist.sort(key=lambda k: k[1])
    del dist[0]
    res = [e[0] for e in dist]
    if num <= 1:
        return res
    return remove_words_from_lst(res,model,num-1,norm)


def remove_dim_and_words(lst, model, dim_num=-0.3, num_of_words_to_subtract=0,norm=False):
    """
    remove from the words list "lst" "num_of_words_to_subtract" words
    then remove dimension by dim_num with the new list of words
    and return the new model and the new words
    :param lst: list of words
    :param model: Word2Vec model that include lst
    :param dim_num: number of dimension of the result
                    if dim num is between -1 to 1 it calculate vector_size * dim_num else it is the original dim num
                    then if dim num > 0  dim num is the number of the dimension in the result
                    if  dim num <0 dim num is the number of dimension to subtract
    :param num_of_words_to_subtract: number of words to remove
    :return: KeyedVectors object with the new vectors , list of the new words
    """
    new_lst = remove_words_from_lst(lst, model, num_of_words_to_subtract,norm)
    return remove_dim(new_lst,model,dim_num,norm) , new_lst


def __check_words(lst, model, norm=False):
    data = ps.ExcelFile('word_list.xlsx')
    data = data.parse('words list')
    first=0
    not_count = 0
    count = 0
    counter=0
    for word, index in one_mean(lst, model,norm):
        counter+=1
        if word in (data[data['type'] == 1]['name']).values:
            count += 1
        if word not in data['name'].values:
            not_count += 1
        if counter == 100:
            first = count
    return count, not_count , first


def output_res(lst,model,savepath ="output.xls",norm=False):
    wb = Workbook()
    sheet1 = wb.add_sheet('result')
    sheet1.write(0, 0, "the words:")
    for i in range(1, len(lst) + 1):
        sheet1.write(0, i, lst[i - 1])
    sheet1.write(3, 0, "vector dim")
    sheet1.write(3, 1, "words number")
    sheet1.write(3, 2, "num of vec")
    sheet1.write(3, 3, "num of result")
    sheet1.write(3, 4, "good results")
    sheet1.write(3, 5, "good results in the first 100")
    sheet1.write(3, 6, "good res / all res")
    sheet1.write(3, 7, "num of not classified words")
    line = 4
    temp = gensim.models.Word2Vec.load("w2v_56_300dim_heb")

    for dim_num in numpy.arange(0, -0.4, -0.1):
        for word_num in range(10):
            new_model, new_lst = remove_dim_and_words(lst, model, dim_num, word_num,norm)
            temp.wv.vectors=new_model.vectors
            temp.wv.vectors_norm = None
            temp.wv.init_sims()
            res = one_mean(new_lst, temp,norm)
            count, not_count, first100 = __check_words(new_lst, temp, norm)
            sheet1.write(line, 0, len(new_model.vectors[0]))
            sheet1.write(line, 1, len(new_lst))
            sheet1.write(line, 2, len(new_model.vectors))
            sheet1.write(line, 3, len(res))
            sheet1.write(line, 4, count)
            sheet1.write(line, 5, first100)
            sheet1.write(line, 6, count / len(res))
            sheet1.write(line, 7, not_count)
            line += 1
    wb.save(savepath)



"""
def old_remove_dim_and_words(lst,model,dim_num=-0.3,words_num=0):
    orgvec=model.wv.vectors
    model = remove_dim(lst,model,len(model.wv.get_vector(lst[0]))-dim_num)
    nlst=lst
    while words_num>0:
        nlst=remove_words_from_lst(nlst,model)
        words_num-=1
        # return the model to its orginal state and remove another dim by the new list
        model.wv.vectors = orgvec
        model.wv.vectors_norm = None
        model.wv.init_sims()
        remove_dim(nlst, model, len(model.wv.get_vector(nlst[0])) - dim_num)
    return nlst , model

"""

def model_details(model):
    """
    print info on the word2vec model by its build in attribute
    :param model: word2vec model ( not KeyedVectors!)
    :return:
    """
    print(model.window, " window")
    print(model.vocabulary.min_count, "min count")
    print(model.vector_size, " vector size")
    print(model.epochs, "epochs")
    print(len(model.wv.vectors),"number of vectors")
    if model.sg == 1:
        print("Training algorithm: skip-gram")
    else:
        print("Training algorithm: CBOW")
    if model.hs == 0 and model.negative > 0:
        print("negative sampling with " + str(model.negative) + " negative words")
    else:
        print("hierarchical softmax")
    print(model.ns_exponent, "negative sampling distribution with exponnet of")
    print(model.vocabulary.sample, "sample")
    return
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

