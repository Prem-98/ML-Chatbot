# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 19:05:17 2021

@author: LENOVO
"""

import string
import textblob
import pandas as pd 
import streamlit as st 
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import random
from nltk.tokenize import word_tokenize
from textblob import Word
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.svm import SVC


# Web Modeling


st.title(' "SARTHI" THE CHATBOT FOR YOUR QUESTIONS : ')

#st.sidebar.header('This is a Chatbot for Supervised Learning Q & A ')
st.subheader('Please ask question related to Supervised Learning ')

que = st.text_input("",key='question')


data=pd.read_csv('final_data.csv',encoding = 'unicode_escape',)

data['question_clean'] = data['Questions'].str.replace('[^\w\s]','')# Removing Punctuations
data['question_clean'] = data['question_clean'].apply(word_tokenize)# Tokenization
data['question_clean'] = data['question_clean'].apply(lambda x: [word.lower() for word in x]) # Converting all Characters to Lowercase
stop_words = set(stopwords.words('english'))
data['question_clean'] = data['question_clean'].apply(lambda x: [word for word in x if word not in stop_words])#Removing stop words
data['question_clean'] = [' '.join(map(str, l)) for l in data['question_clean']] # joining into string to perform lemmatization
data['question_clean'] = data['question_clean'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
data['Answer'] = data['Answer'].str.replace('[^\w\s]','')# Removing Punctuations
sw_nltk = stopwords.words('english')
sw_nltk.extend(['explain','describe','image','`','detail','list','ÃÂÃ','ÃÃÂÃ','Â','ÂÂ','use','used','ââââ','â','ââ','âââ'])
data['question_clean'] = data['question_clean'].str.split().map(lambda x: [word for word in x if word not in sw_nltk])
data['question_clean']=data['question_clean'].apply(' '.join)
data=data.dropna(how='any')

def cleaner(x):
    return [a for a in (''.join([a for a in x if a not in string.punctuation])).lower().split()]
# Model
Pipe = Pipeline(
    [ ('bow',CountVectorizer(analyzer=cleaner)),
     ('tfidf',TfidfTransformer()), 
     ('classifier',SVC()) 
    ])
Pipe.fit(data['question_clean'],data['Answer'])

#Greetings
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey","howdy")
GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]
def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

#i=(1,2,3,4,5,6,7,8,9,10,11)
#for i in i:
    #if i:
        #o=print('value {}'.format(i))
#def on_change():
    #st.experimental_rerun()

st.subheader('This is the related Asnwer for your question.')

if que:
    que=que.lower()
    if(que!='bye'):
        if(que=='thanks'or que=='thank you' or que=='ok thanks'):
            st.write('You are welcome....')
        else:
            if(greeting(que)!=None):
                st.write(greeting(que))                    
            elif(Pipe.predict([que])[0]=='Explanation'):
                st.write(" Sorry I can't understand please rephrase the question")                   
            else:
                st.write(Pipe.predict([que])[0])
                #que=st.text_input('Qestion',on_change=on_change)
                
                    
    else:
        st.write('Bye! take care..............')
        
        




    
    
