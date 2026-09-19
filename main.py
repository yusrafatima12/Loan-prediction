import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('loan.csv', sep='\t')# print(df.head())
# print(df.columns)
# print(df.info())
# print(df.dtypes)
# print(df.shape)

#  Dataset cleaning
# print(df.isnull().mean())
# print(df.describe())
# print(df.info())


# Missing values in all columns
# print(df.isnull().sum())
# print(df.dropna(inplace=True))
# df['col'].fillna(df['col'].mean())


# print(df.duplicated().sum())
# print(df.drop_duplicates(inplace=True))

from sklearn.preprocessing import LabelEncoder
obj = LabelEncoder()
df['col']=obj.fit_transform(df['col'])
df = pd.get_dummies(df, columns=['category_col'], drop_first=True)

# print(df['ID'].count())

# print(df['Education'].unique())

# print(df['SelfEmployed'].value_counts())

# print(df.groupby('SelfEmployed')['LoanStatus'].count())
# print(df.groupby('SelfEmployed')['LoanStatus'].value_counts())

# pd.crosstab(df['CreditHistory'], df['LoanStatus']).plot(kind='bar')
# plt.title('Credit History vs Loan Approval')
# plt.xlabel('Credit History')
# plt.ylabel('Number of Applicants')
# plt.xticks(rotation=0)
# plt.show()


# plt.bar(df['CreditHistory'], df['LoanStatus'] , color = ['teal', 'orange'])

# plt.title('Credit History vs Loan Approval')
# plt.xlabel('Credit History')
# plt.ylabel('Loan Status')
# plt.show()


# plt.bar(df['Education'], df['LoanStatus'] , color = "skyblue")

# plt.title('Education vs Loan Approval')
# plt.xlabel('Education')
# plt.ylabel('Loan Status')
# plt.show()


# df['LoanStatus'].value_counts().plot(kind='pie', autopct='%1.1f%%')

# plt.title('Loan Approval')
# plt.ylabel('')
# plt.show()

# plt.scatter(df['Income'], df['LoanAmount'], c=['grey'])
# plt.title('Income vs Loan Amount')
# plt.xlabel('Income')
# plt.ylabel('Loan Amount')
# plt.show()




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    ConfusionMatrixDisplay
)

x = df[[
    "Gender", "Married", "Dependents", "Education", "SelfEmployed",
    "Income", "CoIncome", "LoanAmount", "Term", "CreditHistory", "Area"
]].copy()

y = df['LoanStatus']

categorical_cols = ["Gender", "Married", "Dependents", "Education", "SelfEmployed", "Area"]
le = LabelEncoder()
for col in categorical_cols:
    x[col] = le.fit_transform(x[col])

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

model = LogisticRegression(max_iter=1000)
model.fit(train_x_scaled, train_y)

y_pred = model.predict(test_x_scaled)
# y_pred_proba = model.predict_proba(test_x_scaled)[:, 1]

# cm = confusion_matrix(test_y, y_pred)
# print("Confusion Matrix:\n", cm)

# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
# plt.show()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y , y_pred)
print("Confusion Matrix:\n", cm)

