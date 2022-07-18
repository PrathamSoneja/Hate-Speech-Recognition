# import dependencies
##sklearn libs
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
##nlp libs
from nltk.tokenize import RegexpTokenizer
import spacy
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
sp = spacy.load('en_core_web_sm')
# essential libs
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import regex as re
import numpy as np
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn')


df_red = pd.read_csv('reddit_all.csv')
df_gab = pd.read_csv('gab_clean.csv')
df_red.drop('Unnamed: 0', axis=1, inplace=True)
df_gab.drop('Unnamed: 0', axis=1, inplace=True)
df_red['red'] = 1
df_gab['red'] = 0

df = df_red.append(df_gab, ignore_index=True)

stopwords_nltk =  set(stopwords.words('english'))

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

temp = pd.DataFrame(columns = ['clean_text','tokens','lemma','tok_lemma'])
for i in range(len(df['text'])):
  temp.loc[len(temp.index), ] = preprocess(df['text'][i])

dataset = pd.concat([df[['text','hate']], temp], axis = 1)
dataset.drop_duplicates(subset=['lemma'], inplace=True)
data = dataset[['tok_lemma','hate']]

X = data['tok_lemma']
y = data['hate']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=28, test_size = 0.33)

vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000, min_df = 3)
logreg = LogisticRegression(C = 2, n_jobs = -1, max_iter = 1000)

X_train_tfidf = vectoriser.fit_transform(X_train)
X_test_tfidf = vectoriser.transform(X_test)

logreg.fit(X_train_tfidf, y_train)
pred_lr = logreg.predict(X_test_tfidf)

accuracy = np.round(accuracy_score(y_test, pred_lr), 4)

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
