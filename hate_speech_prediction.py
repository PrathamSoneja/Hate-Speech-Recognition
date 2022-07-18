# import dependencies
##sklearn libs
import contractions
##nlp libs
from nltk.tokenize import RegexpTokenizer
import spacy
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
sp = spacy.load('en_core_web_sm')
# essential libs
import pandas as pd
import regex as re
import numpy as np
import warnings
import pickle

warnings.filterwarnings('ignore')

stopwords_nltk =  set(stopwords.words('english'))
logreg = pickle.load(open('LRmodel.sav', 'rb'))
vectoriser = pickle.load(open('TFIDFVectoriser.sav', 'rb'))

def cleaning_text(x):
  #preprocessing of text
  x = x.encode('ascii','ignore')
  x = x.decode()
  x = x.lower()
  x = contractions.fix(x)
  x = ' '.join([word for word in x.split() if not word in set(stopwords.words('english'))])
  x =  re.sub('[^a-zA-Z0-9]', ' ', x)
  x = ' '.join(x.split())
  x = re.sub(r"((http|ftp|https):\/\/)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)", "", x)
  x = re.sub('\\t+', '', x)
  x = re.sub('\d+\. ', '', x)
  pattern = "#[\w]*?"
  x = re.sub(pattern,'',x)
  return x


def tokenize_text(txt):
    #only take words or numbers in
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+', gaps=False)
    tokens = tokenizer.tokenize(txt)
    tokens = [token.lower() for token in tokens if not token.lower() in stopwords_nltk] #lowercase
    return tokens


def lemmatize(txt):
    text = sp(txt)
    #get all sentences
    sentence_lst = list(text.sents)
    lemma_words = []
    for sentence in sentence_lst:
        for word in sentence:
            lemma_words.append(word.lemma_)
    #return back as a string
    return ' '.join(lemma_words)


def token_lemma(txt):
    txt = ' '.join(txt)
    text = sp(txt)
    #get all sentences
    sentence_lst = list(text.sents)
    lemma_words = []
    for sentence in sentence_lst:
        for word in sentence:
            lemma_words.append(word.lemma_)
    #return back as a string
    return ' '.join(lemma_words)

def preprocess(text):
  clean_text = cleaning_text(text)
  tokens = tokenize_text(text)
  lemma = lemmatize(text)
  tok_lemma = token_lemma(tokens)
  return [clean_text, tokens, lemma, tok_lemma]



def predict(text):
    temp = preprocess(text)
    text_tok_lemma = temp[-1]
    return logreg.predict(vectoriser.transform([text_tok_lemma]))


# Web app

import streamlit as st

st.title('Hate Speech Predictor')
st.write('')

col1, col2 = st.columns([10, 1])
with col1:
    text_msg = st.text_input('Enter the text:', placeholder='text')

st.write('')

if text_msg == '':
    pass
else:
    pred = predict(text_msg)
    if pred[0] == 1:
        st.write('This would be hateful towards other people.')

    else:
        st.write('This would not be hateful towards other people.')
