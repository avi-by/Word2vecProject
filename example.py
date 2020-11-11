import gensim
import utils
from current_words_list import lst
model = gensim.models.Word2Vec.load("./word2vec models/heb 56 epochs 300 dim/w2v_56_300dim_heb")
print("model details: ")
utils.model_details(model)
print("the average vector of the words list",utils.avg_vec_model(lst,model))
print("result without change to the vectors or the words: ",utils.one_mean(lst,model))
print("the variance of every dim in the vectors of the words: ",utils.my_var(lst,model))
print("the variance of every dim sorted by the variance value")
variance=(utils.my_var(lst,model))
variance.sort(key=lambda k:k[1])
print(variance)
model_after_remove_dim=utils.remove_dim(lst,model,-30)
print("after remove 30 dimensions the length of the vectors is: ",model_after_remove_dim.vector_size)
print("result after remove 30 dimensions: ",utils.one_mean(lst,model))
# print("create xls file with the result for the lst with remove 0 - 10 words and remove 0 , 30 , 60 ,90 dimension")
# utils.output_res(lst,model,savepath ="output.xls")
