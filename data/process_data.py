# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 17:27:24 2020

@author: shevcdim
"""
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_clean_data(messages_filepath):
    '''
    Load incidents with descriptions and priorities,
    Convert priority into 5 categories and drop other fields except of desctiprion and priority
    clean missing data
    '''
    # load data
    df = pd.read_csv(messages_filepath, delimiter = ';',encoding = "ISO-8859-1")
    #drop non needed field
    df.drop(['cmdb_code', 'cmdb', 
       'location', 'u_incident.caller_id.location.u_country',
       'u_incident.caller_id.location.u_country.u_region',
       'u_incident.caller_id.u_segment'], axis=1, inplace=True)
    #convert priority 1 to priority 2 as priority 1 is only manualy set up
    df['Priority'] = df['Priority'].map({1:2, 2:2, 3:3, 4:4, 5:5})
    # load categories for priority 
    
    categories_df = pd.get_dummies(df['Priority'], prefix='priority')
    df.drop(['Priority'], axis=1, inplace=True)
    #create final DF by from original DF and created priorities categories
    df = pd.concat([df, categories_df], axis=1)
    # drop duplicates and NA just in case
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df = df.sample(n=50000)    
    
    return df


 
def save_data(df, database_filename):
    '''
    upload data to DB table drmsg from df dataframe
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('incmsg', engine, index=False, if_exists='replace')  


def main():

        incidents_filepath = '2019oct-2020mar_inc.csv' 
        database_filepath = 'IncidentPriority.db'
        
        print('Loading data...\n    MESSAGES: {}\n'
              .format(incidents_filepath))
        df = load_clean_data(incidents_filepath)

        display(df.columns)
        display(df.head(5))
        display(df.shape)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
'''    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')
'''

if __name__ == '__main__':
    main()


