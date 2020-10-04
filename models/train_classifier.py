# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 19:05:46 2020

@author: shevcdim
"""
import pandas as pd
import numpy as np
import re
import sys
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
import pickle
    
def load_data(database_filepath):
    '''
    connect to DB on database_filepath, read table incmsg into pandas DF
    extract messages content into X   
    categories values into y and category labels into category_label
    '''
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name='incmsg', con=engine)
    #print(df.columns)
    
    X = df.description
    y= df.Priority
    category_label = ['2','3','4','5']
    print(category_label)
    return X, y, category_label


def tokenize(text):
    '''
    use wonltk for tokenizing and lemmatizing
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)      
    return clean_tokens


def build_model():
    '''
    buid ml pipeline using featureunion for text vectorizing and RF
    leverage gridsearch for optimal parameters (for speed only left 1 parameter)
    several classifiers were tried

    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
              #  ('vect', CountVectorizer(min_df = 5, #minimum numbers of documents a word must be present in to be kept
              #         ngram_range= (1,2), #to indicate that we want to consider both unigrams and bigrams.
              #         stop_words ='english')),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        
        #('clf', MultiOutputClassifier(RandomForestClassifier()))
        #('clf', RandomForestClassifier())
        ('clf', LinearSVC())
    ])

    parameters = {
    #    'clf__n_estimators': [100]
     #   'clf__estimator__min_samples_split': [2]
         #'clf__loss': ['hinge', 'squared_hinge']
          'clf__max_iter':[1000]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Build a text report showing the main classification metrics using classification_report from SKlearn

    '''
    y_pred = model.predict(X_test)
  
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
        database_filepath = '../data/IncidentPriority.db' 
        model_filepath = 'classifier.pkl'
  
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
            
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        
        print('Trained model saved!')
     
   
if __name__ == '__main__':
    main()    
    
    
    
    