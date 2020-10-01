import json
import pandas as pd

import sys
from sklearn.externals import joblib
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def main():
    #if len(sys.argv) == 2:
        #print('Will provide prediuction for following desciption:',sys.argv[2])
       # query = sys.argv[2]
        query = 'Extend customer account in AEP'
        model_filepath = 'classifier.pkl'
        
        # load model
        model = joblib.load("../models/classifier.pkl")
        
        classification_labels = model.predict([query])[0]
        print (classification_labels)
        classification_results = dict(zip(['priority_2', 'priority_3', 'priority_4', 'priority_5'], classification_labels))
        
        print (classification_results)
        
    #else:
     #   print('Please provide the incident description')

if __name__ == '__main__':
    main()