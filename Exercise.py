import pandas as pd
import streamlit as slt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pickle

review=pd.read_csv('reviews.csv')
review=review.rename(columns={'text':'review'},inplace=False)

X = review.review
y = review.polarity
#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.6, random_state=1)

vector = CountVectorizer(stop_words = 'english',lowercase=False)
# fit the vectorizer on the training data
vector.fit(X_train)
X_transformed = vector.transform(X_train)
X_transformed.toarray()
# for test data
X_test_transformed = vector.transform(X_test)
naivebayes = MultinomialNB()
naivebayes.fit(X_transformed, y_train)

saved_model = pickle.dumps(naivebayes)
s = pickle.loads(saved_model)

slt.header('Sentimental analysis')
input=slt.text_input("Enter the text")

vec = vector.transform([input]).toarray()
if slt.button('Analyse'):
    analyse=str(list(s.predict(vec))[0]).replace('0', 'NEGATIVE').replace('1', 'POSITIVE')
    slt.write(analyse)