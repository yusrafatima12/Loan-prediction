import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('loan.csv', sep='\t')
df = df.dropna()

x = df[[
    "Gender", "Married", "Dependents", "Education",
    "SelfEmployed", "Income", "CoIncome", "LoanAmount",
    "Term", "CreditHistory", "Area"
]]

y = df["LoanStatus"]

categorical_cols = [
    "Gender", "Married", "Dependents",
    "Education", "SelfEmployed", "Area"
]

for col in categorical_cols:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col])

train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

model = LogisticRegression(max_iter=1000)

model.fit(train_x, train_y)

y_pred = model.predict(test_x)
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(test_y, y_pred)

# print("Confusion Matrix:")
# print(cm)

# from sklearn.metrics import classification_report

# print("Classification Report:")
# print(classification_report(test_y, y_pred))

from sklearn.metrics import accuracy_score, precision_score, recall_score , f1_score
accuracy = accuracy_score(test_y, y_pred)
recall = recall_score(test_y, y_pred)
f1 = f1_score(test_y, y_pred)
precision = precision_score(test_y, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)