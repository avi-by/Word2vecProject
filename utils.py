import gensim
import matplotlib.pyplot as plt
import numpy
import pandas as ps
from gensim.models import KeyedVectors
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

    wv = model.wv if type(model) is not gensim.models.keyedvectors.Word2VecKeyedVectors else model
    lst_vec = []
    for e in lst:
        lst_vec.append(wv.word_vec(e, use_norm=norm))
    return avg_vec(lst_vec)


def one_mean(lst, model, norm=False):
    """
    calculate the avg vector from list of words vectors from word2vec model, then calculate the distance from the avg
    vector to every vector of word in the list, and use the largest distance to define accepted distance,
    then find all the vectors in the model that in the accepted distance.
    :param lst: list of word to make from them the avg vector
    :param model: word2vec model
    :param norm: use normalized vectors
    :return:list of tuple of word and similarity to the avg vector sorted by the similarity
    """
    wv = model.wv if type(model) is not gensim.models.keyedvectors.Word2VecKeyedVectors else model
    # calculate the avg vector
    avg = avg_vec_model(lst, model, norm)
    # calculate the similarity(distance by cosine similarity metric) of every word in the list and the avg vector
    dist = distances(avg, lst, model, norm)
    # sort the distances from the farthest to the closest and take the first, the farthest
    dist.sort()
    dist = dist[0]
    # find i bigger then the num of the word that we want to include to use in 'topn' parameter
    i = 1000
    while wv.similar_by_vector(avg, topn=i)[-1][1] > dist:
        i *= 2
    return [(word, distance) for (word, distance) in wv.similar_by_vector(avg, topn=i) if distance >= dist]


def normal(vec):
    """
    return normalize vector by Euclidean norm ( L2 norm)
    :param vec: 1D vector
    :return: normalize vector
    """
    return vec / numpy.linalg.norm(vec)


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


def my_similarity(v1, v2):
    """
    the method that gensim use when calculate most similarity
    found at the github of gensim
    """
    return numpy.dot(gensim.matutils.unitvec(v1), gensim.matutils.unitvec(v2))


def distances(vec, lst, model, norm=False):
    """
    calculate the similarity (distance by cosine similarity metrica) from the vector and every word in the lst list
    by using the word vector of the word from the model
    :param vec: 1D vector
    :param lst: list of words
    :param model: Word2Vec model that include the words
    :param norm: use normalized vectors
    :return: list of distances, order by the original words index in the list
    """
    wv = model.wv if type(model) is not gensim.models.keyedvectors.Word2VecKeyedVectors else model
    res = []
    for i in lst:
        res.append(my_similarity(vec, wv.word_vec(i, use_norm=norm)))
    return res


def vectors_score(lst, model, method="var", norm=False, **kwargs):
    """
    make list of the dimension and score for every dimension. the score calculate by the method from the parameter on the
    list of the words from the lst parameter/
    :param lst: list of the words
    :param model: word2vec model
    :param method: the score method
    :param norm: if true use the normalized vectors
    :param kwargs: for special parameters for the selected method
    :return: list of (dimension number, score of the dimension)
    """
    return {
        'var': var(lst, model, norm),
        'ent': entropy_model(lst, model, kwargs['decimals'] if 'decimals' in kwargs else 0)
    }[method]


def var(lst, model, norm=False):
    """
    calculate the variance of each dimension in the word vectors of the words in lst
    and return list of the dimension and its variance
    if norm is true it will use normalized vectors
    for example in the vectors [1,2,3], [4,5,6] , [1,5,9]
    the result will be like this [(variance of (1,4,1),0),(variance of (2,5,5),1),(variance of (3,6,9),2)]
    :param lst: list of words
    :param model: Word2Vec model with the words
    :param norm: use normalized vectors
    :return: list of tuple of dimension and its variance (dim,variance) sorted by dim index
    """
    wv = model.wv if type(model) is not gensim.models.keyedvectors.Word2VecKeyedVectors else model
    a = numpy.array([])
    for i in lst:
        a = numpy.append(a, wv.word_vec(i, use_norm=norm))
    a = a.reshape(len(lst), len(wv.word_vec(lst[0])))
    res = numpy.var(a, axis=0)
    var_array = []
    for e in range(len(res)):
        var_array.append([e, res[e]])
    return var_array


def entropy_model(lst, model, decimals=0):
    """
    make list of dimension and score of every dimension, the score calculate by its entropy. the entropy calculate by
    the vectors of the words from lst. the method round the values, with decimals 0 its integers with decimals 1 its one
    digit after the point and so on
    :param lst:list of words
    :param model:word2vec model
    :param decimals:number of digits after the points, for the round method
    :return:list of (dimension number, entropy of the dimension)
    """
    arr = numpy.vstack([(model.wv.get_vector(word)) for word in lst])
    return matrix_entropy(arr, decimals)


def matrix_entropy(data, decimals=0):
    """
    make list of dimension and score of every dimension, the score calculate by its entropy.
    the method round the values, with decimals 0 its integers with decimals 1 its one digit after the point and so on
    :param data: numpy matrix
    :param decimals: number of digits after the points, for the round method
    :return: list of (dimension number, entropy of the dimension)
    """
    trans = data.transpose()
    trans = numpy.around(trans, decimals=decimals)
    return list(zip(range(data.shape[1]), [entropy(t) for t in trans]))


def entropy(data_vector):
    """
    calculate entropy of vector
    :param data_vector: vector with the data values
    :return: float value of the entropy
    """
    # convert the values to non-negative integer values with unique return inverse
    unq, converted_data = numpy.unique(data_vector, return_inverse=True)
    # the function 'bincount' count number of occurrences of each value in array of non-negative ints
    count = numpy.bincount(converted_data)
    # create list of the probability of every value from the values list
    p = count / float(len(data_vector))
    return -numpy.sum(p * numpy.log2(p))


def radius(lst, model, norm=False):
    """
    calculate the furthest distance from the avg vector of the list of the words and the words. its use aa the radius in
    the function "the mean"
    :param lst:list of words
    :param model:word2vec model
    :param norm:if True use normalized vectors
    :return:the distance of the furthest words from the avg vector, float number
    """
    wv = model.wv if type(model) is not gensim.models.keyedvectors.Word2VecKeyedVectors else model
    dist = []
    for word in lst:
        temp = my_similarity(wv.word_vec(word, use_norm=False), avg_vec_model(lst, model, norm))
        dist.append((word, temp))
    dist.sort(key=lambda k: k[1])
    return dist[0][1]


def remove_dim(lst, model, dim_num, method="var", by_values=False, norm=False, **kwargs):
    """
    make from the model and the word list (lst) new model with new vectors with only dim_num dimension
    the vector calculate by calculate the variance of the dimension of the word vectors of "lst"
    then take only the dimension with the smaller variance
    the number of the dimension in the result is:
    if dim num is between -1 to 1 it calculate vector_size * dim_num else it the original dim num
    then if dim num > 0  dim num is the number of the dimension in the result
    if  dim num <0 dim num is the number of dimension to remove
    and drop the (model.wv.vectors_size - dim_num ) dimension with the biggest variance
    if by_values is true then dim_num represents variance size,
    all the dimensions with variance less then or equal to dim_num will include in the result
    and if by_values is true and dim_num <0 then the include dimensions is the dimension with variance less then
    or equal to the largest variance value + dim_num (dim_num is negative)
    :param lst: list of words to calculate the score of the vectors
    :param model: Word2Vec model
    :param dim_num: vector size of the new model
    :param by_values: filter the dimensions by the size of their score method
    :param norm: use normalized vectors
    :return: KeyedVectors object with the new vectors
    """
    wv = model.wv if type(model) is not gensim.models.keyedvectors.Word2VecKeyedVectors else model
    if dim_num == 0:
        new_model = KeyedVectors(wv.vectors.shape[1])
        new_model.add(wv.index2word, wv.vectors)
        return new_model
    if not by_values:
        #  if dim num is percent of the vector size
        if -1 < dim_num < 1:
            dim_num = int(wv.vector_size * dim_num)
        if dim_num < 0:
            dim = wv.vector_size + dim_num
        else:
            dim = dim_num
    arr = []
    score_lst = (vectors_score(lst, model, method=method, norm=norm, kwargs=kwargs))
    # sort by the score from smaller to bigger and that is also from the similar to the different
    score_lst.sort(key=lambda k: k[1])
    if not by_values:
        # list of dimension, the list size is dim_num,
        # the dimensions in the list is the most similar\good dimension - the smaller score
        for i in score_lst[:dim]:
            arr.append(i[0])
    else:
        if dim_num < 0:
            dim_num = score_lst[-1][1] + dim_num
        # take the dimensions with score less than or equal to dim_num
        for i in score_lst:
            if i[1] <= dim_num:
                arr.append(i[0])
    # sort by dimension index
    arr.sort()
    # take all the vector and for each vector take only the elements in the arr list
    # its in fact remove the not necessary dimension from the vectors
    vectors = wv.vectors[:, arr]
    new_model = KeyedVectors(len(arr))
    new_model.add(wv.index2word, vectors)
    return new_model


def remove_dim_change_model(lst, model, dim_num, method="var", by_values=False, norm=False, **kwargs):
    """
    make from the model and the word list (lst) new model with new vectors with only dim_num dimension
    the vector calculate by calculate the variance of the dimension of the word vectors of "lst"
    then take only the dimension with the smaller variance
    the number of the dimension in the result is:
    if dim num is between -1 to 1 it calculate vector_size * dim_num else it the original dim num
    then if dim num > 0  dim num is the number of the dimension in the result
    if  dim num <0 dim num is the number of dimension to remove
    and drop the (model.wv.vectors_size - dim_num ) dimension with the biggest variance
    if by_values is true then dim_num represents variance size,
    all the dimensions with variance less then or equal to dim_num will include in the result
    and if by_values is true and dim_num <0 then the include dimensions is the dimension with variance less then
    or equal to the largest variance value + dim_num (dim_num is negative)
    :param lst: list of words to calculate the score of the vectors
    :param model: Word2Vec model
    :param dim_num: vector size of the new model
    :param by_values: filter the dimensions by the size of their score method
    :param norm: use normalized vectors
    :return: KeyedVectors object with the new vectors
    """
    wv = model.wv if type(model) is not gensim.models.keyedvectors.Word2VecKeyedVectors else model
    if dim_num == 0:
        new_model = KeyedVectors(wv.vectors.shape[1])
        new_model.add(wv.index2word, wv.vectors)
        return new_model
    if not by_values:
        #  if dim num is percent of the vector size
        if -1 < dim_num < 1:
            dim_num = int(wv.vector_size * dim_num)
        if dim_num < 0:
            dim = wv.vector_size + dim_num
        else:
            dim = dim_num
    arr = []
    score_lst = (vectors_score(lst, model, method=method, norm=norm, kwargs=kwargs))
    # sort by the score from smaller to bigger and that is also from the similar to the different
    score_lst.sort(key=lambda k: k[1])
    if not by_values:
        # list of dimension, the list size is dim_num,
        # the dimensions in the list is the most similar\good dimension - the smaller score
        for i in score_lst[:dim]:
            arr.append(i[0])
    else:
        if dim_num < 0:
            dim_num = score_lst[-1][1] + dim_num
        # take the dimensions with score less than or equal to dim_num
        for i in score_lst:
            if i[1] <= dim_num:
                arr.append(i[0])
    # sort by dimension index
    arr.sort()
    # take all the vector and for each vector take only the elements in the arr list
    # its in fact remove the not necessary dimension from the vectors
    vectors = wv.vectors[:, arr]
    model.wv.vectors = vectors
    model.wv.vectors_norm = None
    model.wv.init_sims()
    return model


def remove_words_from_lst(lst, model, num=1, norm=False):
    """
    return new word list in size of len(lst)-num of the most similar words in lst
    calculate by the cosine similarity (distance in cosine similarity metrica) to the avg vector
    and remove the "farthest" words
    then calculate the avg vector again (by recursion) and remove the new farthest word and so on
    until the length of the res is right
    :param lst:words array
    :param model:Word2Vec model that include the words
    :param num:number of words to remove
    :param norm: use normalized vectors
    :return:array of words
    """
    wv = model.wv if type(model) is not gensim.models.keyedvectors.Word2VecKeyedVectors else model
    if num <= 0:
        return lst
    dist = []
    for word in lst:
        temp = my_similarity(wv.word_vec(word, use_norm=norm), avg_vec_model(lst, model, norm))
        dist.append((word, temp))
    dist.sort(key=lambda k: k[1])
    del dist[0]
    res = [e[0] for e in dist]
    if num <= 1:
        return res
    return remove_words_from_lst(res, model, num - 1, norm)


def remove_dim_and_words(lst, model, dim_num=-0.3, num_of_words_to_remove=0, method="var", by_values=False, norm=False,
                         **kwargs):
    """
    remove from the words list "lst" "num_of_words_to_remove" words
    then remove dimension by dim_num with the new list of words
    and return the new model and the new words
    :param lst: list of words
    :param model: Word2Vec model that include lst
    :param dim_num: number of dimension of the result
                    if dim num is between -1 to 1 it calculate vector_size * dim_num else it is the original dim num
                    then if dim num > 0  dim num is the number of the dimension in the result
                    if  dim num <0 dim num is the number of dimension to remove
    :param num_of_words_to_remove: number of words to remove
    :param by_values: filter the dimensions by the size of their variance
    :param norm: use normalized vectors
    :return: KeyedVectors object with the new vectors , list of the new words
    """
    new_lst = remove_words_from_lst(lst, model, num_of_words_to_remove, norm)
    return remove_dim(new_lst, model, dim_num, method=method, by_values=by_values, norm=norm, kwargs=kwargs), new_lst


def plot(data_x, data_y, x_label="", y_label="", title="", save_path="res.png", save=True, arg="", xlim=None,
         ylim=None):
    """
    plot graph and save it as png file
    :param data_x: the values in the data_vector axis
    :param data_y: the values in the y axis
    :param x_label: the label to show on the data_vector axis
    :param y_label: the label to show on the y axis
    :param title: the title of the graph
    :param save_path: the path to save the result
    :param save: if true save the graph to file else only show him
    :param arg: argument for plot func (see pyplot documents)
    :param xlim: the range of the data_vector axis
    :param ylim:  the range of the y axis
    :return:
    """
    plt.plot(data_x, data_y, arg)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.gca().invert_xaxis()
    if save:
        plt.savefig(save_path, dpi=200)
    plt.show()


def check_words(lst, model, classified_words_file='classified words.xlsx', norm=False):
    """
    get list of words word2vec model and xls file with classified words and check how much words after one mean are
    good words how much words don't exist in the xls and in the 100 most similar words how much words are good words
    :param lst: list of words
    :param model: word2vec model
    :param classified_words_file: xls file with classified words
    :param norm: use normalized vectors
    :return:
    """
    data = ps.ExcelFile(classified_words_file)
    data = data.parse('words list')
    first100 = 0
    not_count = 0
    count = 0
    counter = 0
    for word, index in one_mean(lst, model, norm):
        counter += 1
        if word in (data[data['type'] == 1]['name']).values:
            count += 1
        if word not in data['name'].values:
            not_count += 1
        if counter == 20:
            first100 = count
    return count, not_count, first100


def check_words_english(lst, model, classified_words_file='animals words.xlsx',till_recall=False ,norm=False):
    """
    get list of words word2vec model and xls file with classified words and check how much words after one mean are
    good words how much words don't exist in the xls and in the 100 most similar words how much words are good words
    :param lst: list of words
    :param model: word2vec model
    :param classified_words_file: xls file with classified words
    :param norm: use normalized vectors
    :return:
    """
    datasrc = ps.ExcelFile(classified_words_file)
    data = datasrc.parse('words list')
    wrong = datasrc.parse('wrong words')
    max_recall=len(data['name'])
    first20 = 0
    not_count = 0
    count = 0
    counter = 0
    ap = 0
    onemean= one_mean(lst, model, norm)
    if till_recall:
        onemean=onemean[:max_recall]
    for word, index in onemean:
        counter += 1
        if word in (data['name']).values:
            count += 1
            ap += count / counter
        if word not in data['name'].values and word not in wrong['wrong words'].values:
            not_count += 1
        if counter == 20:
            first20 = count
    ap = ap / counter
    recall = count / max_recall
    precision = count/counter
    return count, not_count, first20, ap, recall, precision, counter


def eng_output(lst, model, num_words_remove=0, file_name='eng_output.xls', classified_words_file='animals words.xlsx',
               norm=False):
    model = model.wv if type(model) is not gensim.models.keyedvectors.Word2VecKeyedVectors else model
    wb = Workbook()
    sheet1 = wb.add_sheet('result')
    calculate_result(classified_words_file, lst, model, norm, num_words_remove, sheet1)
    sheet2 = wb.add_sheet('result till recall')
    calculate_result(classified_words_file, lst, model, norm, num_words_remove, sheet2,till_recall=True)
    wb.save(file_name)


def calculate_result(classified_words_file, lst, model, norm, num_words_remove, sheet1,till_recall=False):
    sheet1.write(0, 0, "the words:")
    for i in range(1, len(lst) + 1):
        sheet1.write(0, i, lst[i - 1])
    sheet1.write(3, 0, "vector dim")
    sheet1.write(3, 1, "words number")
    sheet1.write(3, 2, "num of vec")
    sheet1.write(3, 3, "num of result")
    sheet1.write(3, 4, "good results")
    sheet1.write(3, 5, "good results in the first 20")
    sheet1.write(3, 6, "Precision (good res / all res)")
    sheet1.write(3, 7, "Recall")
    sheet1.write(3, 8, "f1")
    sheet1.write(3, 9, "Average Precision")
    sheet1.write(3, 10, "num of not classified words")
    sheet1.write(3, 11, "removed word")
    sheet1.write(3, 12, "radius")
    line = 4
    removed_words = []
    current_removed_word = ""
    for i in range(num_words_remove + 1):
        new_lst = remove_words_from_lst(lst, model, i)
        #res = one_mean(new_lst, model, norm)
        count, not_count, first20, ap, recall,precision,counted = check_words_english(new_lst, model, classified_words_file,till_recall)
        rand = radius(new_lst, model)
        for i in lst:
            if i not in new_lst:
                if i not in removed_words:
                    removed_words.append(i)
                    current_removed_word = i
        sheet1.write(line, 0, len(model.vectors[0]))
        sheet1.write(line, 1, len(new_lst))
        sheet1.write(line, 2, len(model.vectors))
        sheet1.write(line, 3, counted)
        sheet1.write(line, 4, count)
        sheet1.write(line, 5, first20)
        sheet1.write(line, 6, precision)
        sheet1.write(line, 7, recall)
        sheet1.write(line, 8, (2 * precision * recall) / (precision + recall))
        sheet1.write(line, 9, ap)
        sheet1.write(line, 10, not_count)
        sheet1.write(line, 11, current_removed_word)
        sheet1.write(line, 12, rand.item())
        line += 1


def output_res(lst, model, save_path="output.xls", steps=numpy.arange(0, -1.5, -0.2), num_words_remove=0, method="var",
               by_values=True,
               classified_words_file='classified words.xlsx', norm=False):
    """
    make xls file with data on the result of one mean on the model before anf after the remove of the dimensions
    and the words. the data include the number of results, the number of the good results
    and the number of the good results in the first 100 results
    :param method: the method to calculate the dimension to remove
    :param lst: list of the words
    :param model: word2vec model
    :param save_path: the path of the output file
    :param steps: range, by this range the function remove dimension from the original model
    :param num_words_remove: number of words to remove from the lst, the function remove 1 word,
    save the results in the file then remove the next words. the function will do it over again
    for every step (= number of dimension, by 'step' parameter)
    :param by_values:  if true the number of dimension to remove (=step parameter) will calculate as the variance size
    and not by the dimension quantity
    :param classified_words_file: xls file with classified words
    :param norm: use normalized vectors
    :return:
    """
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
    for dim_num in steps:
        for word_num in range(num_words_remove + 1):
            new_model, new_lst = remove_dim_and_words(lst, model, dim_num, word_num, method=method, by_values=by_values,
                                                      norm=norm)
            res = one_mean(new_lst, new_model, norm)
            count, not_count, first100 = check_words(new_lst, new_model, classified_words_file,
                                                     norm=norm)
            sheet1.write(line, 0, len(new_model.vectors[0]))
            sheet1.write(line, 1, len(new_lst))
            sheet1.write(line, 2, len(new_model.vectors))
            sheet1.write(line, 3, len(res))
            sheet1.write(line, 4, count)
            sheet1.write(line, 5, first100)
            sheet1.write(line, 6, count / len(res))
            sheet1.write(line, 7, not_count)
            line += 1
    if save_path == "output.xls":
        save_path = method + " "
        save_path += "by value" if by_values == True else "by dim num"
        save_path += " " + str(steps[0]) + " to " + str(numpy.around(steps[-1]))
        if num_words_remove > 0:
            save_path += " remove " + str(num_words_remove) + " words"
        steps += ".xls"
    wb.save(save_path)


def output_graph(lst, model, steps=numpy.arange(0, -1.5, -0.2), by_values=True, method="var",
                 classified_words_file='classified words.xlsx', norm=False):
    """
    get list of words and word2vec model and how much dimension to remove plot 3 graph from the results,
    one of the number of the results in every number of dimensions
    one of the number of the "good" words in the results (classified by the file from classified_words_file parameter),
    and one of the number of "good" words in the first 100 results
    :param lst: list of words
    :param model: word2vec model
    :param steps: range, for the removing of the dimensions
    :param by_values: if true remove dimensions by the variance size and not by the quantity of the dimensions
    :param method: the method to calculate the dimension to remove
    :param classified_words_file: xls file with classified words
    :param norm: use normalized vectors
    :return:
    """
    xdata = []
    firstarray = []
    goodresarray = []
    resnumarray = []
    for dim_num in steps:
        new_model = remove_dim(lst, model, dim_num, method=method, by_values=by_values, norm=norm)
        res = one_mean(lst, new_model, norm)
        count, not_count, first100 = check_words(lst, new_model, classified_words_file, norm=norm)
        xdata.append(new_model.vector_size)
        firstarray.append(first100)
        goodresarray.append(count)
        resnumarray.append(len(res))
    title = "" if len(steps) == 1 else "remove dim by step of " + str(steps[1] - steps[0])
    if title != "" and by_values is True:
        title += " variance size"
    plot(xdata, resnumarray, "number of dimension", "number of result", title, "result number.png")
    plot(xdata, goodresarray, "number of dimension", "number of the good results", title, "good results.png", arg='g')
    plot(xdata, firstarray, "number of dimension", "number of the good words in the first 100 results", title,
         "number of first 100 good results.png", arg='m', ylim=(0, 100))


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
    print(len(model.wv.vectors), "number of vectors")
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
