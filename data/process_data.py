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
    drop other fields except of desctiprion and priority
    clean missing data
    u_incident,u_incident.number,u_incident.priority,u_normalized_short_description,u_normalized_description

    '''
    # load data
    df = pd.read_csv(messages_filepath,delimiter = ',',encoding = "ISO-8859-1")
    #display(df.shape)
    df.drop(['u_incident', 'u_incident.cmdb_ci',
       'u_incident.caller_id.location',
       'u_incident.caller_id.location.u_country',
       'u_incident.caller_id.location.u_country.u_region',
       'u_incident.caller_id.u_segment',], axis=1, inplace=True)
    #convert priority 1 to priority 2 as priority 1 as it is only critical events manualy set by analyst
    df['Priority'] = df['u_incident.priority'].map({1:2, 2:2, 3:3, 4:4, 5:5})
    # load categories for priority 
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df['description'] = df['u_normalized_short_description'] + ' ' + df['u_normalized_description']
    
    df.drop(['u_incident.priority','u_normalized_short_description','u_normalized_description'], axis=1, inplace=True)
    #df = df.sample(n=10000) 
    
    return df


 
def save_data(df, database_filename):
    '''
    upload data to DB table drmsg from df dataframe
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('incmsg', engine, index=False, if_exists='replace')  


def main():

      
        incidents_filepath = 'normalized_incident.csv'
        database_filepath = 'IncidentPriority.db'
        
        print('Loading data...\n   from: {}\n'
              .format(incidents_filepath))
        df = load_clean_data(incidents_filepath)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database! ')
        

if __name__ == '__main__':
    main()


