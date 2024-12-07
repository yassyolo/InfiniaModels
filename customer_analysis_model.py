import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from imblearn.over_sampling import SMOTE
import joblib

data = {
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female',
               'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'TransactionFrequency': [2, 3, 2, 1, 3, 2, 3, 1, 3, 2, 2, 1, 3, 1, 3, 2, 1, 2, 3, 2, 1],
    'AverageTransactionAmount': [150.00, 200.00, 120.00, 300.00, 250.00, 180.00, 190.00, 210.00, 240.00, 180.00,
                                  170.00, 220.00, 210.00, 240.00, 180.00, 160.00, 250.00, 230.00, 260.00, 280.00, 210.00],
    'TotalTransactionAmount': [6000.00, 8000.00, 5200.00, 12000.00, 10000.00, 7200.00, 8200.00, 9500.00, 8700.00, 7500.00,
                               6800.00, 8500.00, 7000.00, 9500.00, 7800.00, 6500.00, 11000.00, 10000.00, 11500.00, 9800.00, 10500.00],
    'LoanAmountRequested': [10000, 15000, 12000, 18000, 20000, 13000, 15000, 11000, 16000, 14000,
                            11000, 13000, 14500, 17000, 15500, 16000, 18000, 14500, 17000, 16500, 19000],
    'LoanAmountApproved': [9000, 14000, 11000, 17000, 18000, 12000, 13000, 10000, 15000, 13000,
                           10500, 12500, 14000, 16000, 15000, 14000, 17000, 13000, 16000, 15500, 18000],
    'AccountBalance': [5000, 7000, 6000, 12000, 8000, 9000, 9000, 7000, 10000, 9500,
                       8000, 9500, 7800, 9800, 8200, 9200, 11000, 10200, 10500, 9900, 10800],
    'AccountAge': [2, 3, 1, 5, 4, 3, 2, 3, 4, 5, 2, 4, 5, 3, 1, 4, 5, 2, 3, 4, 3],
    'MonthlyIncome': [3000, 4000, 2500, 5000, 4500, 3500, 3800, 4200, 4900, 4100,
                      3200, 4300, 4600, 4900, 5100, 4000, 5400, 4600, 4700, 4800, 5100],
}

df = pd.DataFrame(data)

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

df['Age_Income'] = df['AccountAge'] * df['MonthlyIncome']
df['Transaction_Age'] = df['TransactionFrequency'] * df['AccountAge']
df['Avg_Transaction_Amount_Per_AccountAge'] = df['AverageTransactionAmount'] / (df['AccountAge'] + 1)
df['Loan_Income_Ratio'] = df['LoanAmountRequested'] / (df['MonthlyIncome'] + 1)
df['Balance_Income_Ratio'] = df['AccountBalance'] / (df['MonthlyIncome'] + 1)
df['Loan_Difference'] = df['LoanAmountRequested'] - df['LoanAmountApproved']
df['Transaction_Per_Income'] = df['TransactionFrequency'] / (df['MonthlyIncome'] + 1)
df['Loan_Per_AccountAge'] = df['LoanAmountRequested'] / (df['AccountAge'] + 1)
df['Transaction_Loan_Ratio'] = df['TotalTransactionAmount'] / (df['LoanAmountRequested'] + 1)

X = df.drop(columns='LoanAmountApproved')
y = df['LoanAmountApproved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

if len(y_train.value_counts()) > 1 and min(y_train.value_counts()) > 1:
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
else:
    print("Skipping SMOTE due to insufficient class imbalance.")
    X_train_resampled, y_train_resampled = X_train, y_train

scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

model = RandomForestRegressor(random_state=42)

param_dist = {
    'n_estimators': [100, 200, 500],
    'max_depth': [1, 3, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=60,
    cv=7,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
random_search.fit(X_train_resampled, y_train_resampled)

best_model = random_search.best_estimator_
cv_scores = cross_val_score(best_model, X_train_resampled, y_train_resampled, cv=5, scoring='neg_mean_squared_error')

print("Random Forest - Best Parameters:", random_search.best_params_)
print("Cross-validated mean negative MSE:", cv_scores.mean())

y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-Squared:", r2)

importances = best_model.feature_importances_
print("Feature Importance:")
for feature, importance in zip(X.columns, importances):
    print(f"{feature}: {importance}")

gender_analysis = df.groupby('Gender').agg({
    'AverageTransactionAmount': 'mean',
    'TotalTransactionAmount': 'sum',
    'LoanAmountRequested': 'mean',
    'LoanAmountApproved': 'mean',
    'AccountBalance': 'mean',
    'AccountAge': 'mean',
    'MonthlyIncome': 'mean'
}).reset_index()

pd.set_option('display.max_columns', None)

for index, row in gender_analysis.iterrows():
    gender = 'Female' if row['Gender'] == 0 else 'Male'
    print(f"Gender: {row['Gender']} represents {gender}")
    print(f"AverageTransactionAmount:\nFor {gender.lower()}s (Gender = {row['Gender']}), the average transaction amount is ${row['AverageTransactionAmount']:.2f}.")
    print(f"TotalTransactionAmount:\n{gender}s made a total of ${row['TotalTransactionAmount']:.2f} in transactions.")
    print(f"LoanAmountRequested:\nOn average, {gender.lower()}s requested ${row['LoanAmountRequested']:.2f} in loans.")
    print(f"LoanAmountApproved:\nOn average, {gender.lower()}s were approved for ${row['LoanAmountApproved']:.2f} in loans.")
    print(f"AccountBalance:\nThe average account balance for {gender.lower()}s is ${row['AccountBalance']:.2f}.")
    print(f"AccountAge:\nThe average account age for {gender.lower()}s is {row['AccountAge']} years.")
    print(f"MonthlyIncome:\nThe average monthly income for {gender.lower()}s is ${row['MonthlyIncome']:.2f}.")

joblib.dump(best_model, 'best_random_forest_model.pkl')