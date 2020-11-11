# Research in NLP

train new text example:    


```
import train_new_text_with_codec as tr    
tr.train("example.txt","utf-8","modelFileName") 
```
that's it!  

there is more parameters but the default of the function is very good  
the default is: 56 epochs, 300 dim, CBOW , windows size 5,  
min count 10, negative sample with 5 negative words ,4 workers
