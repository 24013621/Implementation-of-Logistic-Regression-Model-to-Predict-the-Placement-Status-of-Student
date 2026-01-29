# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: AHAMED JASEER SHA E
RegisterNumber: 212224040015

```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(r"C:\Users\admin\Desktop\Placement_Data.csv")
data.head()

datal = data.copy()
datal = datal.drop(["sl_no", "salary"], axis=1)
datal.head()

print("Missing values:\n", datal.isnull().sum())
print("Duplicate rows:", datal.duplicated().sum())

le = LabelEncoder()
datal["gender"] = le.fit_transform(datal["gender"])
datal["ssc_b"] = le.fit_transform(datal["ssc_b"])
datal["hsc_b"] = le.fit_transform(datal["hsc_b"])
datal["hsc_s"] = le.fit_transform(datal["hsc_s"])
datal["degree_t"] = le.fit_transform(datal["degree_t"])
datal["workex"] = le.fit_transform(datal["workex"])
datal["specialisation"] = le.fit_transform(datal["specialisation"])
datal["status"] = le.fit_transform(datal["status"])

x = datal.iloc[:, :-1]
y = datal["status"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)

classification_report_output = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_output)
```

## Output:
<img width="493" height="494" alt="image" src="https://github.com/user-attachments/assets/cfcb1d60-a296-4933-a292-c1c1eaaf3331" />

HEAD
<img width="1095" height="297" alt="image" src="https://github.com/user-attachments/assets/be94c15c-3a11-4e1f-be7e-6704fd9788df" />

COPY 
<img width="940" height="294" alt="image" src="https://github.com/user-attachments/assets/6814dab7-ed6d-4dc1-b79c-d2beebdb94be" />

FIT TRANSFORM
<img width="914" height="608" alt="image" src="https://github.com/user-attachments/assets/f42077d2-fac9-4fbe-85f0-c61aea1d2a06" />

LOGISTIC REGRESSION 
<img width="637" height="239" alt="image" src="https://github.com/user-attachments/assets/9ded6345-3502-4b5f-8a4e-fa8bf72bf4a4" />

ACCURACY SCORE 
<img width="360" height="146" alt="image" src="https://github.com/user-attachments/assets/9eb85d81-008e-476b-b590-fbd690c09617" />

CONFUSION MATRIX 
<img width="379" height="138" alt="image" src="https://github.com/user-attachments/assets/a8ff41b5-2d67-4d73-9a02-1ab3b3076584" />

CLASSIFICATION REPORT 
<img width="485" height="241" alt="image" src="https://github.com/user-attachments/assets/a911f098-011a-4b4e-8c1a-a1c90a9c8338" />

PREDICTION
<img width="1305" height="127" alt="image" src="https://github.com/user-attachments/assets/95154dd7-aad3-4e04-8e1c-7c830c87cb43" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
