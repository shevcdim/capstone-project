import json
import pandas as pd

import sys
import joblib
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
    
    if len(sys.argv) == 2:
        query = sys.argv[1]
        model_filepath = 'classifier.pkl'
        
        # load model
        model = joblib.load("../models/classifier.pkl")
        
        classification_labels = model.predict([query])[0]
        print ('Evaluated minimum priority is :',int(classification_labels))
        
    else:
        print('Please provide the incident description')

if __name__ == '__main__':
    main()