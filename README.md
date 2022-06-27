# Absenteeism-Prediction-Project
## OBJECTIVE:            
To Predict ABSENTEEISM at a company during work time. Absenteeism meaning absence from work during normal working hours, resulting in temporary incapacity to execute regular working activity.
 
## PROBLEM:  
The business environment today is more competitive than ever before, this leads to increased pressure in the workplace therefore it is reasonable to expect that unachievable business goals and elevated risk of unemployment can raise people stress levels, often a continuous presence of such factors becomes detrimental to persons health. Sometimes this may result in minor illness and can develop a long-term condition.
We will be solving the problem from the point of view of the person in charge of productivity in the company, so we will focus on predicting absenteeism of an employee during work time based on certain characteristics. We want to know whether an employee can be expected to be missing for a specific number of hours in a given workday. And having such information in advance can improve our decision making, by reorganizing the work process in a way that will allow us to avoid a lack of productivity and increase the quality of work generated in organization.

## DATA PREPROCESSING:
Dataset for analysis is obtained from primary data sources.
This data set has following columns:
Columns = [‘ID’, 'Reason for Absence', 'Date', 'Transportation Expense', 'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children', 'Pets', 'Absenteeism Time in Hours']
1.	Dropped ID column.
2.	Pre-processing on 'Reason for Absence' column:
Performed OneHotEncoding, dropped a column to avoid multicollinearity, and grouped into 4 new columns, and added to data frame.
3.	Pre-processing on 'Date' column:
Changed datatype to date time, extracted 2 new columns namely ‘month_value’ and ‘weekday’ from Date column and then dropped it.
4.	Pre-processing on ' Education ' column:
Values in this column is encoded as: 1 means high school, 2 means graduate, 3 means post graduate and 4 means phd. Since Number of occurrences of 1 is much higher than other values, so for analysis 1 is encoded as 0 and other values that is 2,3,4 is encoded as 1.
5.	Pre-processing on 'Absenteeism Time in Hours' column (target variable):
Since here we use Logistic Regression to predict absenteeism, we have to classify our data namely ‘Moderately absent’ and ‘Excessively absent’. For that purpose, we use median of the respective column and encode as 0(Moderately absent) for values below median and 1(Excessively absent) for values above median.

## ANALYSIS METHODOLOGY:
Tools use for project are Python(for data preprocessing and model building), MySQL Workbench( for dumping predicted data to database) and Tableau for visualization and interpretation.

Logistic Regression is used to solve this business problem. Python libraries used are Pandas, NumPy, Sklearn, Pickle and pymysql.
Then wrote a python script to created a python module which can clean and preprocess data as per requirment and returns the predicted outputs along with preprocessed data in a data set, which is then dumped into MySQL database, from where data is imported to Tableau to create visualizations and data interpretations.

## TABLEAU DSAHBOARD LINK:
https://public.tableau.com/app/profile/sachin.patel8488/viz/AbsenteeismAnalysisDashboard/TransportationExpense
