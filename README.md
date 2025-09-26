# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook / Google Collab

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: sanjay kumar B
RegisterNumber:212224230242
/*
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x

y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
## Placement data 
<img width="1138" height="205" alt="ml5" src="https://github.com/user-attachments/assets/cb73d2ef-fcde-423e-9695-3a76e97a9930" />

## Salary data
<img width="1135" height="216" alt="mlsa" src="https://github.com/user-attachments/assets/affce756-9435-40cb-abd3-117550e08433" />

## Checking null function
<img width="477" height="427" alt="mlc" src="https://github.com/user-attachments/assets/afed0fcf-44da-42ac-8a69-bf211041882f" />


## Data duplicate
<img width="247" height="52" alt="mldupl" src="https://github.com/user-attachments/assets/3c125299-6293-4b9a-bfee-6605e926dc47" />

## Print data
<img width="1142" height="487" alt="mlp" src="https://github.com/user-attachments/assets/c52310eb-11ba-41d1-ac63-15f330db52eb" />

## Data status
<img width="757" height="390" alt="mlds" src="https://github.com/user-attachments/assets/62acaad4-32ff-478a-ab6a-48da443fb4be" />


## Y-prediction array
<img width="1092" height="85" alt="y prediv" src="https://github.com/user-attachments/assets/5ee2ceec-5498-49dd-9b68-b2af9ee3855d" />

## Accuracy value
<img width="513" height="67" alt="acc" src="https://github.com/user-attachments/assets/bf8ddf92-1833-463b-9629-81844fc13907" />


## Confusion array
<img width="466" height="87" alt="on" src="https://github.com/user-attachments/assets/3ae806b8-08da-4d96-b84d-bb0d2d07ab9d" />

## Classification report
<img width="1137" height="43" alt="classreport" src="https://github.com/user-attachments/assets/8d61aea9-493c-4f92-92f9-d1d9c2cf0ac3" />

## Prediction of LR
<img width="1136" height="57" alt="prediction" src="https://github.com/user-attachments/assets/b0b976ec-29da-4ccf-887f-f13d5a0b5e5b" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
