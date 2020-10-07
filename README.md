## Udacity capstone-project
# Project name: Incident Prioritization advisor Project
https://github.com/shevcdim/capstone-project

## Table of Contents
####    About (Business problem)
####    Details of project
####    Instructions
####    File Descriptions
####    Conclusion
####    Licensing, Authors, and Acknowledgements

**About (Business problem)**
    One of essential part for every reported incident is to be able to properly estimate priority of this Incident as it will drive further reaction and resolution time SLA from       operational teams.
    For refference we have 5 priorities for Incidents:
    1 - Global Crisis resolution time 4 Hours
    2 - Major Incident resolution time 24 hours
    3 - Minor application or critical individual user problem - resolution time 72 hours
    4 - Non critical user problem - resolution time 5 business days
    5 - User request/how to - resolution time 10 business days
    We have H1 2020 Incidents data with initial incident description captured by analyst on front line and resulted priority of this incident.
    I created classifier to learn on Data dump of previous incident and levering NLTK tokenizer and lemmatizer + LinearSVM classifier to predict potential priority of new              incident to assist analyst on front line when he or she assign the priority for new coming incidents
    I will continue to work on optimizing model accurace and F1 score and consider using CNN as well as data quality need to addressed with support team for quality incident           description.
Full story is available here https://medium.com/@shevchuk.dimitri/it-incident-priority-prediction-udacity-capstone-project-1f6463dbc6f2


**Instructions:**

Step 1. In the folder called data You need to unpack (zip) raw incident data files called 
    normalized_incident_desc_only.zip
    normalized_incident_no_description.zip

Step 2. In the same folder run python file process_data.py. This code will process the raw data from incident file and create SQL light DB with the saved data for further use

Step 3. Go to folder called models and run python file train_classifier.py. It will read the data from DB and build classifiation model , store it in file classifier.pkl for future use.

Step 3.1 You might need to install joblib (if not installed already) by pip install joblib

Step 4. Go to folder called app and run python run.py followed by "description of incident" , this will return forecasted classigication of your incident based on description provided
      Example: python run.py "Navision is not working for RCA factory"


**File Descriptions:**
data\process_data.py - read the raw data from normalized_incident.csv file, process it and put data frame into IncidentPriority.db
data\IncidentPriority.db - sql light DB to store incidents data frame 
data\normalized_incident_desc_only.csv raw data of incidents with inc number and description
data\normalized_incident_no_description.csv - raw data of incidents with inc number and other fields incl priority
models\train_classifier.py - process text description of incident using NLTK tokenizer with LinearSVM classifier leveraging pipeline with featureunion
models\classifier.pkl -linearSVM model save file
app\run.py - utility tool to clasify input text with incident description for Incident prioritization.

**Conclusion (Summarizing the problem end to end)**
The problem of forecasting Priority of new coming incident based on incident description can be solved with classification algorithms leveraging word tokenizing and vectorizing.
I was able to achieve 71% accuracy only on the test data using LinearSVC algorithm and this was best so far comparing to RandomForest Classifier.
I believe that in order to improve results further to 80+ I need to focus on data quality and ensure that historical incidents descriptions have clear and structured summary of what happened, where happend, with level of impact and other symptoms so that model can generalize data well

**Licensing, Authors, Acknowledgements**
Dataset used for this project is property of Marc Inc 
