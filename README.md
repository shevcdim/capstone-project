# capstone-project
Incident Prioritization advisor Project
https://github.com/shevcdim/capstone-project

Table of Contents
About (Business problem)
Instructions
File Descriptions
Licensing, Authors, and Acknowledgements
About (Business problem)
One of essential part for every reported incident is to be able to properly estimate priority of this Incident as it will drive further reaction and resolution time SLA from operational teams.
For refference we have 5 priorities for Incidents:
1 - Global Crisis resolution time 4 Hours
2 - Major Incident resolution time 24 hours
3 - Minor application or critical individual user problem - resolution time 72 hours
4 - Non critical user problem - resolution time 5 business days
5 - User request/how to - resolution time 10 business days
We have H1 2020 Incidents data with initial incident description captured by analyst on front line and resulted priority of this incident.
I created classifier to learn on Data dump of previous incident and levering NLTK tokenizer and lemmatizer + LinearSVM classifier to predict potential priority of new incident to assist analyst on front line when he or she assign the priority for new coming incidents
I will continue to work on optimizing model accurace and F1 score and consider using CNN as well as data quality need to addressed with support team for quality incident description.

Instructions:
Step 1. You need to unpack (7zip) raw incident data first normalized_incident.7z
Step 2. You need to go to data folder and run python file process_data.py. This code will process the raw data from incident file and create SQL light DB with the saved data for further use
Step 3. You need to go to folder models and run python file train_classifier.py. It will read the data from DB and build classifiation model , store it in file classifier.pkl for future use.
Step 3.1 You might need to install joblib (if not installed already) by pip install joblib
Step 4. You need to go to app folder and run python run.py followed by "description of incident"

Example: python run.py "Navision is not working for RCA factory"


File Descriptions:
data\process_data.py - read the raw data from normalized_incident.csv file, process it and put data frame into IncidentPriority.db
data\IncidentPriority.db - sql light DB to store incidents data frame 
data\normalized_incident.csv - raw data of incidents
models\train_classifier.py - process text description using NLTK
models\classifier.pkl
app\run.py


README.md

Licensing, Authors, Acknowledgements
Dataset of messages and categories is provided and owned by Figure Eight
