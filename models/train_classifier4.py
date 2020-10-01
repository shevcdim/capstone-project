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
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(X_train, y_train):
    '''
    buid ml pipeline using featureunion for text vectorizing and RF
    leverage gridsearch for optimal parameters (for speed only left 1 parameter)

    '''
    count_vect = CountVectorizer(min_df = 5, #minimum numbers of documents a word must be present in to be kept
                       ngram_range= (1,2), #to indicate that we want to consider both unigrams and bigrams.
                       stop_words ='english')
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = LinearSVC().fit(X_train_tfidf, y_train)
    
    return clf, count_vect


def evaluate_model(model, X_test, Y_test, category_names, count_vect):
    '''
    Build a text report showing the main classification metrics using classification_report from SKlearn

    '''
    y_pred = model.predict(count_vect.transform(X_test))
    #convert results to DF
   # Y_pred = pd.DataFrame(data=y_pred, 
   #                       index=Y_test.index, 
    #                      columns=category_names)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    #if len(sys.argv) == 3:
        database_filepath = '../data/IncidentPriority.db' 
        model_filepath = 'classifier.pkl'
  
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
            
        print('Building model...')
        model, cv = build_model(X_train, Y_train)
        
        print('Training model...')
       # model.fit(X_train, Y_train)
        
     #   display(model.get_params())
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names, cv)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        
        print('Trained model saved!')
        query = 'cannot log into fido'
        #query = 'I have outdated information on my credit report'
        
        classification_labels = model.predict(cv.transform([query]))[0]
        print (classification_labels)
        #classification_results = dict(zip(['priority_2', 'priority_3', 'priority_4', 'priority_5'], classification_labels))
        
        #print (classification_results)
'''
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
'''     
   
if __name__ == '__main__':
    main()    
    
    
    
    