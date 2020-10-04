# capstone-project
Incident Prioritization advisor Project
https://github.com/shevcdim/capstone-project

Table of Contents
Instructions
File Descriptions
Licensing, Authors, and Acknowledgements
Instructions:
ETL The first part of data pipeline is the Extract, Transform, and Load process. To load the data into an SQLite database, I am using the pandas dataframe .to_sql() method, which you can use with an SQLAlchemy engine, cleaning code is also included in the final ETL script, process_data.py.

Machine Learning Pipeline For the machine learning portion, I split the data into a training set and a test set. Then, I created a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification) using Random Forest classifier. Finally, I exported model to a pickle file. train_classifier.py

Example: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

python train_classifier.py ../data/DisasterResponse.db classifier.pkl

Flask App is used to display results in a Flask web app.

Follow this instruction to run the whole thing

Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Run the following command in the app's directory to run your web app. python run.py

Go to http://0.0.0.0:3001/

File Descriptions:
app | - template | |- master.html # main page of web app | |- go.html # classification result page of web app |- run.py # Flask file that runs app

data |- disaster_categories.csv # data to process |- disaster_messages.csv # data to process |- process_data.py |- InsertDatabaseName.db # database to save clean data to

models |- train_classifier.py |- classifier.pkl # saved model

README.md

Licensing, Authors, Acknowledgements
Dataset of messages and categories is provided and owned by Figure Eight
