# capstone-project
Incident Prioritization advisor Project
https://github.com/shevcdim/capstone-project

Table of Contents
About (Business problem)
Details of project
Instructions
File Descriptions
Conclusion
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

Details of project
What algorithm were used:
    In this project following algorithm were used
        nltk word_tokenize and WordNetLemmatizer to tokenize and lemmatize(convert to initial base form) text desciption
        sklearn CountVectorizer to convert text description of incidents into to a matrix of token counts leveraging NLTK tokenizeing
        sklearn TfidfTransformer Transform a token count matrix to a normalized tf-idf representation 
        sklearn LinearSVC (Linear Support Vector Classification) was selected as multiclass classification algorithm, capable of handling large datasets.
        
Hyperparameter if done on what parameters, what were the old and final parameters
        i try following parameters for LinearSVC classifier:
            tol - Tolerance for stopping criteria. [1e-5,1e-4, 1e-3],
            max_iterint - The maximum number of iterations to be run [500, 1000, 1500],
            dual - Select the algorithm to either solve the dual or primal optimization problem.[True, False]
      
        Best parameters set found on development set:
        {'clf__dual': False, 'clf__max_iter': 500, 'clf__tol': 1e-05} 
               
Challenges faced
        biggest challenge which I faced was the accuracy of the model as well as precision and recall.
        I also tried Random forest classification and some simple CNN, but so far LinearSVC gives best results
        I think it have a lot to do with the quality of data, as support analytics are often put mistakes in the words, capture inaccurate details of the issue so it is difficult for model to generalize.
        I plan to work with our support team to increase quality of incident description to improve the results

Two aspect that you find interesting, how would you go about improving them
        First aspect I found interesting is the quality of initial data and impact of it for the model resul. I already had a plan to work with support organization to increase the quality of data capture
        Second aspect I found interesting is that LinearSVC shows better result and much faster learning time vs RandomForest. I expected faster learning time, but not the results. This is why I choose Linear SVC to improve model further

Results section, all model results
Evaluating model...
              precision    recall  f1-score   support

           2       0.56      0.28      0.37       413
           3       0.43      0.17      0.24      1932
           4       0.66      0.65      0.65     15075
           5       0.74      0.80      0.77     21432

    accuracy                           0.71     38852
   macro avg       0.60      0.47      0.51     38852
weighted avg       0.69      0.71      0.70     38852
        
        

Instructions:
Step 1. In the folder called data You need to unpack (7zip) raw incident data file called normalized_incident.7z
Step 2. In the same folder run python file process_data.py. This code will process the raw data from incident file and create SQL light DB with the saved data for further use
Step 3. Go to folder called models and run python file train_classifier.py. It will read the data from DB and build classifiation model , store it in file classifier.pkl for future use.
Step 3.1 You might need to install joblib (if not installed already) by pip install joblib
Step 4. Go to folder called app and run python run.py followed by "description of incident" , this will return forecasted classigication of your incident based on description provided
  Example: python run.py "Navision is not working for RCA factory"


File Descriptions:
data\process_data.py - read the raw data from normalized_incident.csv file, process it and put data frame into IncidentPriority.db
data\IncidentPriority.db - sql light DB to store incidents data frame 
data\normalized_incident.csv - raw data of incidents
models\train_classifier.py - process text description of incident using NLTK tokenizer with LinearSVM classifier leveraging pipeline with featureunion
models\classifier.pkl -linearSVM model save file
app\run.py - utility tool to clasify input text with incident description for Incident prioritization.

Conclusion (Summarizing the problem end to end)
The problem of forecasting Priority of new coming incident based on incident description can be solved with classification algorithms leveraging word tokenizing and vectorizing.
I was able to achieve 71% accuracy only on the test data using LinearSVC algorithm and this was best so far comparing to RandomForest Classifier and simple CNN.
I believe that in order to improve results further to 80+ I need to focus on data quality and ensure that historical incidents descriptions have clear and structured summary of what happened, where happend, with level of impact and other symptoms so that model can generalize data well

Licensing, Authors, Acknowledgements
Dataset used for this project is property of Marc Inc 
