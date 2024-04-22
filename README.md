# Credit_Card_Fraud  
Credit Card Fraud Detection Project 4 Team 6  
Dataset located at: https://drive.google.com/drive/folders/18y-7oFjkxsOnaDOWkhwiNVZtF26WwLJ-?usp=drive_link  
## An Overview of the project and its purpose:  
To create a binary classification machine learning model to identify fraudulent transactions within a population of 555718 transactions.  The labelled dataset consisted of the following 22 features:  

1. Trans_date_trans_time: Timestamp of the transaction (date and time).  
2. Cc_num:Unique customer identification number.  
3. MerchantThe merchant involved in the transaction.  
4. Category Transaction type (e.g., personal, childcare). b
5. Amt:Transaction amount.  
6. First: Cardholder's first name.  
7. Last Cardholder's last name.  
8. Gender: Cardholder's gender.  
9. Street: Cardholder's street address.  
10. City: Cardholder's city of residence.
11. State: Cardholder's state of residence.
12. Zip: Cardholder's zip code.
13. Lat:Latitude of cardholder's location.
14. Long: Longitude of cardholder's location.
15. City_pop:Population of the cardholder's city.
16. Job:Cardholder's job title.
17. Dob: Cardholder's date of birth.
18. Trans_num: Unique transaction identifier.
19. Unix_time: Transaction timestamp (Unix format).
20. Merch_lat:Merchant's location (latitude).
21. Merch_long: Merchant's location (longitude).
22. Is_fraud:Fraudulent transaction indicator (1 = fraud, 0 = legitimate). This is the target variable for classification purposes.

The project was broken into 5 parts.
1.  Descriptive analysis:  a review of the features and how they relate to the target variable.
2.  Target variable review  
3.  Feature Engineering and Data Preprocessing   
4.  Selection of the objective metric(s) for model assessment.
5.  Assessing various types of machine learning algorithims with minimal tuning.
6.  Selecting "best" algorithim from the set of algroithims assessed for hyperparamater tuning.

## Part 1 Descriptive Analysis  

## Part 2 Target Variable Review
The dataset consisted of 555718 transaction records of which 2145 were fraudulent (0.3860%).  The labels therefore are highly imbalanced and steps were taken to address the impact of this imbalance on the models.

## Part 3 Feature Engineering and Data Preprocessing
After performing a descriptive analysis of the data a number of features were dropped or created then encoded/transformed and scaled:   
1. *Dropped Features*: 'First' Name, 'Last' Name, 'cc_num', 'street', 'city', 'state', 'dob', 'Trans_num', 'Unix_time', 'Lat', 'Long', 'Merch_lat', 'Merch_long'
2. *Created Features*: 'Region', an amalgamation of U.S. States according to the  U.S. Bureau of Economic Analysis. Age_years, the age of the cardholder based on the difference between today's date and the cardholders DOB. Distance_km, the distance between the cardholders latitude and longitude and the merchants latitude and longitude.
3. *Encoded Features*: binary or get_dummies encoding - 'catagories', 'gender', 'region'. Target encoding - merchants, jobs.
4. *Transformed Features*: amt (Transaction Amount) due to the very high dispersion in this feature it was transformed by the natural log.
5. The resulting dataset after all preprocessing steps were applied was scaled using the standard_scaler, and split into training and testing sets with 75% of the data used for training and 25% was used for testing.  Due to the very high imbalance in target labels (classes) the training and test splits were reviewed to ensure an adequate number of labels were assigned to each set.  
                `Average class probability in data set:     0.003860`
                `Average class probability in training set: 0.003839`
                `Average class probability in test set:     0.003923`

## Part 4 Selection of the objective metrics for model assessment
As recommended by Scikit Learn[^1] Balanced Accuracy is a more appropriate objective metric over accuracy when working with imbalanced classes in the target, as it weighs each sample according to the inverse prevlaence of its true class.  It is this metric that is used to assess each model's performance.

## Part 5 Assessing various machine learning algorithms.
1. Logistic Regression  
`     Confusion Matrix: LogisticRegression  
            Predicted Legitimate 0	Predicted Fraudulent 1  
Legitimate 0	138369	                    16  
Fraudulent 1	   524	                    21    
Accuracy Score : 0.9961131505074498  
Balanced Accuracy Score: 0.5192082453121577  
Classification Report  
              precision    recall  f1-score   support`  

           0       1.00      1.00      1.00    138385  
           1       0.57      0.04      0.07       545  
`
    accuracy                           1.00    138930  
   macro avg       0.78      0.52      0.54    138930  
weighted avg       0.99      1.00      0.99    138930 `   

[^1][sklearn.metrics.balanced_accuracy_score](Scikit Learn Balanced Accuracy Score)  



