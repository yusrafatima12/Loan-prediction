import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('loan.csv', sep='\t')
df = df.dropna()

x = df[[
    "Gender", "Married", "Dependents", "Education",
    "SelfEmployed", "Income", "CoIncome", "LoanAmount",
    "Term", "CreditHistory", "Area"
]].copy()

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

# new_applicant = [[
#     1, 1, 1, 0, 0,
#     5000, 2000, 150, 360, 1, 1
# ]]

# new_applicant = scaler.transform(new_applicant)

# prediction = model.predict(new_applicant)

# print("Prediction:", prediction)



new_data = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "1",
    "Education": "Graduate",
    "SelfEmployed": "No",
    "Income": 5000,
    "CoIncome": 2000,
    "LoanAmount": 150,
    "Term": 360,
    "CreditHistory": 1,
    "Area": "Urban"
}

new_data["Gender"] = 1
new_data["Married"] = 1
new_data["Education"] = 0
new_data["SelfEmployed"] = 0
new_data["Area"] = 2

print(new_data)

new_data = [[
    new_data["Gender"],
    new_data["Married"],
    new_data["Dependents"],
    new_data["Education"],
    new_data["SelfEmployed"],
    new_data["Income"],
    new_data["CoIncome"],
    new_data["LoanAmount"],
    new_data["Term"],
    new_data["CreditHistory"],
    new_data["Area"]
]]

new_data = scaler.transform(new_data)

prediction = model.predict(new_data)

print("Prediction:", prediction)