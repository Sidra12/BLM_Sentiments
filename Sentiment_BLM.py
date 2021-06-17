#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import nltk
#nltk.download('stopwords')
import re
plt.show()
from sklearn.feature_extraction.text import *
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from sklearn.preprocessing import *
from numpy import argmax
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv(r'/Users/sidraaziz/PycharmProjects/SentimentAnalysis/data.csv') 


# In[2]:


dataset.head()


# In[3]:


dataset['Sentiment'] = dataset['Sentiment'].astype(float)
#dataset['Sentiment_Map'] = dataset['Sentiment_Map'].astype(float)

dataset.dtypes

import matplotlib.pyplot as plt

dataset['Sentiment'] = dataset['Sentiment'].astype(float)
#dataset['Sentiment_Map'] = dataset['Sentiment_Map'].astype(float)
print("Number of rows per score :")
print(dataset['Sentiment'].value_counts())

# Function to map scores to sentiment
def map_sentiment(score_received):
    if 0 > score_received >= -1:
        return -1
    elif score_received == 0:
        return 0
    elif score_received <= 1:
        return 1
# Mapping scores to sentiment into three categories
dataset['Sentiment_Map'] = [ map_sentiment(x) for x in dataset['Sentiment']]
# Plotting the sentiment distribution
plt.figure()
pd.value_counts(dataset['Sentiment_Map']).plot.bar(title="Sentiment distribution in dataset")
plt.xlabel("Sentiment")
plt.ylabel("No. of rows in dataset")
plt.show()


# In[4]:


stop_words = stopwords.words('english')
stop_words.append('rt')

dataset['Tweets'] = dataset['Tweets'].str.lower()
print(dataset['Tweets'])
print("——————————— Remove Stop Word——————————")
dataset['Stopworded_Tweets'] = dataset['Tweets'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


# In[5]:


from nltk.stem import PorterStemmer
ps = PorterStemmer()
print("——————————— Stemming —————————")
dataset['Stemmed_Tweets'] = dataset['Stopworded_Tweets'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))


# In[6]:


from gensim.utils import simple_preprocess

# Tokenize the text column to get the new column 'tokenized_text'
dataset['Tokenized_Stem_Tweets'] = [simple_preprocess(line, deacc=True) for line in dataset['Stemmed_Tweets']]
print(dataset['Tokenized_Stem_Tweets'].head(10))


# In[7]:


# Create a function to clean the tweets
def cleanTxt(text):
 text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
 text = re.sub('#', '', text) # Removing '#' hash tag
 text = re.sub('RT[\s]+', '', text) # Removing RT
 text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
 
 return text


# Clean the tweets
dataset['Tweets'] = dataset['Tweets'].apply(cleanTxt)

# Show the cleaned tweets
dataset.head()


# In[7]:


pip install gensim


# In[8]:


from gensim.utils import simple_preprocess

# Tokenize the text column to get the new column 'tokenized_text'
dataset['Tokenized_Stem_Tweets'] = [simple_preprocess(line, deacc=True) for line in dataset['Stemmed_Tweets']]
print(dataset['Tokenized_Stem_Tweets'].head(10))

dataset.head()


# In[9]:


from sklearn.model_selection import train_test_split
# Train Test Split Function
def split_train_test(top_data_df_small, test_size=0.3, shuffle_state=True):
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[['Tweets', 'Location', 'Sentiment', 'Stopworded_Tweets', 'Stemmed_Tweets', 'Tokenized_Stem_Tweets']], 
                                                        dataset['Sentiment_Map'], 
                                                        shuffle=shuffle_state,
                                                        test_size=test_size, 
                                                        random_state=15)
    print("Value counts for Train sentiments")
    print(Y_train.value_counts())
    print("Value counts for Test sentiments")
    print(Y_test.value_counts())
    print(type(X_train))
    print(type(Y_train))
    X_train = X_train.reset_index()
    X_test = X_test.reset_index()
    Y_train = Y_train.to_frame()
    Y_train = Y_train.reset_index()
    Y_test = Y_test.to_frame()
    Y_test = Y_test.reset_index()
    print(X_train.head())
    return X_train, X_test, Y_train, Y_test

# Call the train_test_split
X_train, X_test, Y_train, Y_test = split_train_test(dataset)


# In[53]:


from gensim.models import Word2Vec
import time
w2v_model = Word2Vec(min_count=20,
                     window=2,
                     size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=3)



w2v_model.build_vocab(dataset['Tokenized_Stem_Tweets'], progress_per=10000)

#summarise loaded model:
print(w2v_model)
print()
#summarise vocab
words = list(w2v_model.wv.vocab)
print(words)
print()
# access vector for one word
print(w2v_model['white'])
print()
# save model
w2v_model.save('model.bin')
print()
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)
print()


# In[36]:


X = w2v_model[w2v_model.wv.vocab]


# In[37]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
result = pca.fit_transform(X)


# In[38]:


import matplotlib.pyplot as plt
plt.scatter(result[:, 0], result[:, 1])


# In[41]:


words = list(w2v_model.wv.vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
words = list(w2v_model.wv.vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    
plt.show()


# In[49]:


# train model
w2v_model = Word2Vec(dataset['Tokenized_Stem_Tweets'], min_count=1)

# fit a 2d PCA model to the vectors
X = w2v_model[w2v_model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# create a scatter plot of the projection
plt.figure(figsize=(15,15))
plt.scatter(result[:, 0], result[:, 1])
words = list(w2v_model.wv.vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:


# define training data
sentences = [['I', 'am', 'going', 'crazy', 'about this', 'for', 'you'],
            ['Me', 'dont', 'like', 'crazy', 'things'],
            ['You', 'are', 'crazy'],
            ['Us', 'can do', 'crazy'],
            ['Them', 'are', 'crazy', 'of all']]
# train model
model = Word2Vec(sentences, min_count=1)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['crazy'])
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)


# In[ ]:





# In[12]:


X = model[model.wv.vocab]


# In[13]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
result = pca.fit_transform(X)


# In[14]:


import matplotlib.pyplot as plt
plt.scatter(result[:, 0], result[:, 1])


# In[ ]:





# In[15]:


words = list(model.wv.vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
words = list(model.wv.vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))


# In[16]:


# train model
model = Word2Vec(sentences, min_count=1)
# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
plt.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()


# In[17]:


from gensim.scripts.glove2word2vec import glove2word2vec
import gensim.downloader as api

# download the model and return as object ready for use
model_glove_twitter = api.load("glove-twitter-25")


# In[18]:


# get similar items
model_glove_twitter.wv.most_similar("hopelessness",topn=10)


# In[58]:


# get similar items
model_glove_twitter.wv.most_similar("anxiety",topn=10)


# In[57]:


# get similar items
model_glove_twitter.wv.most_similar("depression",topn=10)


# In[60]:


# get similar items
model_glove_twitter.wv.most_similar("happiness",topn=10)


# In[61]:


# get similar items
model_glove_twitter.wv.most_similar("sadness",topn=10)


# In[67]:


#what doesn't fit?
model_glove_twitter.wv.doesnt_match(["happy","funny","depressed","love","sad"])


# In[68]:


# show weight vector for trump
model_glove_twitter['depression'],model_glove_twitter['suicide']


# In[70]:





# In[89]:


import pandas as pd
from sklearn.metrics import jaccard_score
from gensim.models.word2vec import Word2Vec


# In[93]:


import pandas as pd
from sklearn.metrics import jaccard_score

phrases=["barrack obama","barrack h. obama","barrack hussein obama","michelle obama","donald trump","melania trump"]
query="barack hussain obama"

results_glove=[]
results_jaccard=[]


# In[94]:


def compute_jaccard(t1,t2):
    
    intersect = [value for value in t1 if value in t2] 
    
    union=[]
    union.extend(t1)
    union.extend(t2)
    union=list(set(union))
    
    
    jaccard=(len(intersect))/(len(union)+0.01)
    return jaccard


# In[97]:


for p in phrases:
    tokens_1=[t for t in p.split() if t in model.wv.vocab]
    tokens_2=[t for t in query.split() if t in model.wv.vocab]
    
    #compute jaccard similarity
    jaccard=compute_jaccard(tokens_1,tokens_2)
    results_jaccard.append([p,jaccard])
    
    #compute cosine similarity using word embedings 
    cosine=0
    if (len(tokens_1) > 0 and len(tokens_2)>0):
        cosine=model_glove_twitter.wv.n_similarity(tokens_1,tokens_2)
        results_glove.append([p,cosine])


# In[98]:


print("Phrases most similar to '{0}' using glove word embeddings".format(query))
pd.DataFrame(results_glove,columns=["phrase","score"]).sort_values(by=["score"],ascending=False)


# In[99]:


print("Phrases most similar to '{0}' using jaccard similarity".format(query))
pd.DataFrame(results_jaccard,columns=["phrase","score"]).sort_values(by=["score"],ascending=False)


# In[104]:


pip install tokenizer


# In[ ]:


import gensim.downloader as api

wv = api.load('word2vec-google-news-300')

vec_king = wv['king']


# In[ ]:




