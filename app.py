import streamlit as st
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
nltk.download('stopwords')
def transform_text(text):
    punctuation = set(string.punctuation)
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    stopwords = nltk.corpus.stopwords.words('english')
    text = [word for word in text if word.isalnum() and word not in stopwords and word not in punctuation]
    #apply stemming
    text = [ps.stem(word) for word in text]
    return " " .join(text)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS Spam Classifier')
input_sms = st.text_area('Enter the message')

if st.button("Predict"):
    #Preprocessing
    transformed_sms = transform_text(input_sms)
    #Vectorize
    vector_input = tfidf.transform([transformed_sms])
    #predict
    result = model.predict(vector_input)[0]
    #Display
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')

