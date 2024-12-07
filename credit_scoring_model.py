import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from scipy.stats import randint, uniform
import joblib

np.random.seed(42)

data = pd.DataFrame({
    'NetMonthlyIncome': np.random.randint(2000, 15000, size=1000),
    'FixedMonthlyExpenses': np.random.randint(500, 3000, size=1000),
    'PermanentContractIncome': np.random.randint(0, 10000, size=1000),
    'TemporaryContractIncome': np.random.randint(0, 7000, size=1000),
    'CivilContractIncome': np.random.randint(0, 7000, size=1000),
    'BusinessIncome': np.random.randint(0, 20000, size=1000),
    'PensionIncome': np.random.randint(0, 3000, size=1000),
    'FreelanceIncome': np.random.randint(0, 7000, size=1000),
    'OtherIncome': np.random.randint(0, 5000, size=1000),
    'HasApartmentOrHouse': np.random.randint(0, 2, size=1000),
    'HasCommercialProperty': np.random.randint(0, 2, size=1000),
    'HasLand': np.random.randint(0, 2, size=1000),
    'HasMultipleProperties': np.random.randint(0, 2, size=1000),
    'HasPartialOwnership': np.random.randint(0, 2, size=1000),
    'NoProperty': np.random.randint(0, 2, size=1000),
    'VehicleCount': np.random.randint(0, 5, size=1000),
    'MaritalStatus': np.random.randint(0, 2, size=1000),
    'HasOtherCredits': np.random.randint(0, 2, size=1000),
    'NumberOfHouseholdMembers': np.random.randint(1, 7, size=1000),
    'MembersWithProvenIncome': np.random.randint(0, 7, size=1000),
    'Dependents': np.random.randint(0, 6, size=1000),
    'IsRetired': np.random.randint(0, 2, size=1000),
    'YearsAtJob': np.random.randint(0, 40, size=1000),
    'MonthsAtJob': np.random.randint(0, 480, size=1000),
    'TotalWorkExperienceYears': np.random.randint(0, 40, size=1000),
    'TotalWorkExperienceMonths': np.random.randint(0, 480, size=1000),
    'EducationLevel': np.random.randint(1, 6, size=1000),
    'LoanAmount': np.random.randint(1000, 100000, size=1000),
    'LoanTermMonths': np.random.randint(6, 240, size=1000),
    'InterestRate': np.random.uniform(1.0, 8.0, size=1000),
    'AccountBalance': np.random.randint(0, 30000, size=1000),
    'LoanApproval': np.random.randint(0, 2, size=1000),
    'LoanRepayment': np.random.randint(100, 5000, size=1000),
    'HasLoans': np.random.randint(0, 2, size=1000),
    'PaidAllLoansOnTime': np.random.randint(0, 2, size=1000)
})



feature_weights = {
    'NetMonthlyIncome': 2.6,
    'FixedMonthlyExpenses': 2.6,
    'PermanentContractIncome': 1.8,
    'TemporaryContractIncome': 1.5,
    'CivilContractIncome': 1.3,
    'BusinessIncome': 1.7,
    'PensionIncome': 1.0,
    'FreelanceIncome': 1.2,
    'OtherIncome': 1.0,
    'LoanRepayment': 2.6,
    'AccountBalance': 1.6,
    'NumberOfHouseholdMembers': 1.2,
    'Dependents': 1.2,
    'YearsAtJob': 1.2,
    'MonthsAtJob': 1.2,
    'EducationLevel': 1.1,
    'HasApartmentOrHouse': 1.0,
    'HasCommercialProperty': 0.8,
    'HasLand': 0.8,
    'VehicleCount': 0.6,
    'MaritalStatus': 1.2,
    'IsRetired': 0.5,
    'LoanAmount': 2.0,
    'LoanTermMonths': 1.2,
    'InterestRate': 1.5,
    'HasLoans': 1.5,
    'PaidAllLoansOnTime': 1.2,
}

def apply_feature_weights(X, feature_weights):
    X_weighted = X.copy()
    for feature, weight in feature_weights.items():
        if feature in X_weighted.columns:
            X_weighted[feature] *= weight
    return X_weighted

def feature_engineering(data):
    data['DisposableIncome'] = data['NetMonthlyIncome'] - (data['FixedMonthlyExpenses'] + data['LoanRepayment'])
    data['LoanRepaymentBurden'] = data['LoanRepayment'] / (data['NetMonthlyIncome'] + 1)
    data['AccountBalanceToLoanRatio'] = data['AccountBalance'] / (data['LoanAmount'] + 1)
    data['AccountBalanceToIncomeRatio'] = data['AccountBalance'] / (data['NetMonthlyIncome'] + 1)
    data['WeightedIncome'] = (data['PermanentContractIncome'] * 1.5 +
                              data['TemporaryContractIncome'] * 1.2 +
                              data['CivilContractIncome'] * 1.1 +
                              data['BusinessIncome'] * 1.3 +
                              data['PensionIncome'] * 1.0 +
                              data['FreelanceIncome'] * 1.1 +
                              data['OtherIncome'] * 1.0)

    data['PropertyScore'] = (data['HasApartmentOrHouse'] * 3 +
                             data['HasLand'] * 2 +
                             data['VehicleCount'] * 1 +
                             data['HasPartialOwnership'] * 1.5)

    data['HouseholdScore'] = (data['NumberOfHouseholdMembers'] - data['Dependents'] +
                              data['MembersWithProvenIncome'] * 2)

    data['IncomeLoanAmountInteraction'] = data['NetMonthlyIncome'] * data['LoanAmount']
    data['ExpensesDisposableIncomeInteraction'] = data['FixedMonthlyExpenses'] * data['DisposableIncome']
    data['DependencyRatio'] = data['Dependents'] / (data['NumberOfHouseholdMembers'] + 1)
    data['ProvenIncomeProportion'] = data['MembersWithProvenIncome'] / (data['NumberOfHouseholdMembers'] + 1)
    data['JobExperienceRatio'] = data['MonthsAtJob'] / (data['TotalWorkExperienceMonths'] + 1)
    data['EducationIncomeInteraction'] = data['EducationLevel'] * data['NetMonthlyIncome']
    data['IncomeToDebtRatio'] = data['NetMonthlyIncome'] / (data['LoanRepayment'] + data['FixedMonthlyExpenses'] + 1)
    data['SavingsRate'] = data['AccountBalance'] / (data['NetMonthlyIncome'] + 1)
    data['PropertyVehicleScore'] = data['HasApartmentOrHouse'] * 3 + data['HasLand'] * 2 + data['VehicleCount'] * 1.5
    data['EffectiveLoanTerm'] = data['LoanTermMonths'] / (data['NetMonthlyIncome'] + 1)
    data['LoanTermPropertyInteraction'] = data['LoanTermMonths'] * data['PropertyScore']
    data['JobEducationInteraction'] = data['JobExperienceRatio'] * data['EducationLevel']
    data['IncomeVariability'] = data['NetMonthlyIncome'] - data[['PermanentContractIncome', 'TemporaryContractIncome', 'CivilContractIncome', 'BusinessIncome', 'PensionIncome', 'FreelanceIncome', 'OtherIncome']].max(axis=1)
    data['TotalIncome'] = data[['PermanentContractIncome', 'TemporaryContractIncome', 'CivilContractIncome', 'BusinessIncome', 'PensionIncome', 'FreelanceIncome', 'OtherIncome']].sum(axis=1)
    data['LoanRepaymentIncomeDebtInteraction'] = data['LoanRepayment'] * data['IncomeToDebtRatio']
    data['NetIncomePerHouseholdMember'] = data['NetMonthlyIncome'] / (data['NumberOfHouseholdMembers'] + 1)
    data['DebtPerHouseholdMember'] = (data[['LoanRepayment', 'FixedMonthlyExpenses']].sum(axis=1) / (data['NumberOfHouseholdMembers'] + 1))
    data['SavingsPerHouseholdMember'] = data['AccountBalance'] / (data['NumberOfHouseholdMembers'] + 1)
    data['DebtToIncomeRatio'] = (data[['LoanRepayment', 'FixedMonthlyExpenses']].sum(axis=1) / data['NetMonthlyIncome'] + 1)
    data['DebtServiceCoverageRatio'] = data['NetMonthlyIncome'] / (data[['LoanRepayment', 'FixedMonthlyExpenses']].sum(axis=1) + 1)
    data['IncomeConsistencyScore'] = data[['PermanentContractIncome', 'TemporaryContractIncome', 'CivilContractIncome', 'BusinessIncome', 'PensionIncome', 'FreelanceIncome', 'OtherIncome']].std(axis=1) / data[['PermanentContractIncome', 'TemporaryContractIncome', 'CivilContractIncome', 'BusinessIncome', 'PensionIncome', 'FreelanceIncome', 'OtherIncome']].mean(axis=1)
    data['StableIncomeScore'] = data[['PermanentContractIncome', 'TemporaryContractIncome', 'CivilContractIncome', 'BusinessIncome', 'PensionIncome', 'FreelanceIncome', 'OtherIncome']].max(axis=1) / data['NetMonthlyIncome']
    data['IncomeAndDebtInteraction'] = (data['NetMonthlyIncome'] ** 2) / (data[['LoanRepayment', 'FixedMonthlyExpenses']].sum(axis=1) + 1)
    data['IncomeSavingsInteraction'] = data['NetMonthlyIncome'] * data['SavingsRate']
    data['TotalLoanCost'] = data['LoanAmount'] * (1 + (data['InterestRate'] / 100) * (data['LoanTermMonths'] / 12))
    data['InterestRateToIncomeRatio'] = data['InterestRate'] / (data['NetMonthlyIncome'] + 1)
    data['HasOtherCreditsImpact'] = data['HasOtherCredits'] * data['IncomeToDebtRatio']

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(data[['NetMonthlyIncome', 'LoanRepaymentBurden',
                                             'PropertyScore', 'HouseholdScore',
                                             'AccountBalanceToLoanRatio', 'AccountBalanceToIncomeRatio',
                                             'IncomeLoanAmountInteraction', 'ExpensesDisposableIncomeInteraction',
                                             'DependencyRatio', 'ProvenIncomeProportion', 'JobExperienceRatio',
                                             'EducationIncomeInteraction', 'IncomeToDebtRatio', 'SavingsRate',
                                             'PropertyVehicleScore', 'EffectiveLoanTerm', 'LoanTermPropertyInteraction',
                                             'JobEducationInteraction', 'IncomeVariability', 'TotalIncome',
                                             'LoanRepaymentIncomeDebtInteraction', 'NetIncomePerHouseholdMember',
                                             'DebtPerHouseholdMember', 'SavingsPerHouseholdMember',
                                             'DebtToIncomeRatio', 'DebtServiceCoverageRatio',
                                             'IncomeConsistencyScore', 'StableIncomeScore',
                                             'IncomeAndDebtInteraction', 'IncomeSavingsInteraction',
                                             'TotalLoanCost', 'InterestRateToIncomeRatio', 'HasOtherCreditsImpact']])

    poly_feature_names = poly.get_feature_names_out(['NetMonthlyIncome', 'LoanRepaymentBurden',
                                                     'PropertyScore', 'HouseholdScore',
                                                     'AccountBalanceToLoanRatio', 'AccountBalanceToIncomeRatio',
                                                     'IncomeLoanAmountInteraction',
                                                     'ExpensesDisposableIncomeInteraction',
                                                     'DependencyRatio', 'ProvenIncomeProportion', 'JobExperienceRatio',
                                                     'EducationIncomeInteraction', 'IncomeToDebtRatio', 'SavingsRate',
                                                     'PropertyVehicleScore', 'EffectiveLoanTerm',
                                                     'LoanTermPropertyInteraction',
                                                     'JobEducationInteraction', 'IncomeVariability', 'TotalIncome',
                                                     'LoanRepaymentIncomeDebtInteraction',
                                                     'NetIncomePerHouseholdMember',
                                                     'DebtPerHouseholdMember', 'SavingsPerHouseholdMember',
                                                     'DebtToIncomeRatio', 'DebtServiceCoverageRatio',
                                                     'IncomeConsistencyScore', 'StableIncomeScore',
                                                     'IncomeAndDebtInteraction', 'IncomeSavingsInteraction',
                                                     'TotalLoanCost', 'InterestRateToIncomeRatio',
                                                     'HasOtherCreditsImpact'])

    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
    data = pd.concat([data, poly_df], axis=1)
    joblib.dump(poly, 'polyy.pkl')
    return data


data = feature_engineering(data)

X = data.drop(columns=['LoanApproval'])
y = data['LoanApproval']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

X_weighted = apply_feature_weights(pd.DataFrame(X_scaled, columns=X.columns), feature_weights)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_weighted, y)

enn = NearestNeighbors(n_neighbors=5)
enn.fit(X_resampled)

distances, indices = enn.kneighbors(X_resampled)
neighbors_avg = distances.mean(axis=1)
threshold = np.percentile(neighbors_avg, 75)
safe_indices = np.where(neighbors_avg < threshold)[0]

X_resampled_enn = X_resampled.iloc[safe_indices]
y_resampled_enn = y_resampled.iloc[safe_indices]

X_resampled = pd.concat([X_resampled, X_resampled_enn], axis=0)
y_resampled = pd.concat([y_resampled, y_resampled_enn], axis=0)

X_resampled = np.concatenate((X_resampled, X_resampled_enn), axis=0)
y_resampled = np.concatenate((y_resampled, y_resampled_enn), axis=0)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

gb_model = GradientBoostingClassifier(random_state=42)

param_dist = {
    'n_estimators': randint(100, 600),
    'learning_rate': uniform(0.001, 0.4),
    'max_depth': randint(6, 12),
    'max_features': ['sqrt', 'log2', None],
    'max_leaf_nodes': randint(10, 60),
    'min_impurity_decrease': uniform(0.0, 0.2),
    'min_samples_leaf': randint(1, 12),
    'min_samples_split': randint(2, 20),
    'subsample': uniform(0.0, 0.1)
}

random_search = RandomizedSearchCV(estimator=gb_model, param_distributions=param_dist, n_iter=40, cv=8,
                                   scoring='roc_auc', n_jobs=-1, verbose=1, random_state=42)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
joblib.dump(best_model, 'best_gradient_boosting_model.pkl')

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

feature_columns = X.columns.tolist()

with open('feature_columns.pkl', 'wb') as f:
    joblib.dump(feature_columns, f)

print("Best Hyperparameters:")
print(random_search.best_params_)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"\nROC AUC Score: {roc_auc:.4f}")

def calculate_credit_score(probability, min_score=300, max_score=850):
    probability = np.clip(probability, 0, 1)
    credit_score = min_score + (max_score - min_score) * probability
    return credit_score


def assess_risk(probability, credit_score, min_score=300, max_score=850):
    risk_categories = {
        'Low': (min_score + 0.7 * (max_score - min_score), max_score),
        'Medium': (min_score + 0.4 * (max_score - min_score), min_score + 0.7 * (max_score - min_score)),
        'High': (min_score, min_score + 0.4 * (max_score - min_score))
    }

    for category, (low_threshold, high_threshold) in risk_categories.items():
        if low_threshold <= credit_score <= high_threshold:
            return category

    return 'Unknown'

example_data = pd.DataFrame({
    'NetMonthlyIncome': [3000],
    'FixedMonthlyExpenses': [1500],
    'PermanentContractIncome': [3000],
    'TemporaryContractIncome': [0],
    'CivilContractIncome': [0],
    'BusinessIncome': [0],
    'PensionIncome': [0],
    'FreelanceIncome': [0],
    'OtherIncome': [0],
    'HasApartmentOrHouse': [1],
    'HasCommercialProperty': [0],
    'HasLand': [0],
    'HasMultipleProperties': [0],
    'HasPartialOwnership': [0],
    'NoProperty': [0],
    'VehicleCount': [2],
    'MaritalStatus': [1],
    'HasOtherCredits': [0],
    'NumberOfHouseholdMembers': [4],
    'MembersWithProvenIncome': [2],
    'Dependents': [1],
    'IsRetired': [0],
    'YearsAtJob': [5],
    'MonthsAtJob': [60],
    'TotalWorkExperienceYears': [6],
    'TotalWorkExperienceMonths': [72],
    'EducationLevel': [2],
    'LoanAmount': [15000],
    'LoanTermMonths': [36],
    'InterestRate': [5.0],
    'AccountBalance': [0],
    'LoanRepayment': [449.56],
    'HasLoans': [1],
    'PaidAllLoansOnTime': [1]
})

example_data = feature_engineering(example_data)
example_data_scaled = scaler.transform(example_data)

prediction = best_model.predict_proba(example_data_scaled)

print(f"Prediction for example data: {prediction[0]}")
probability_positive_class = prediction[0, 1]
print(f"Probability for example data: {probability_positive_class:.4f}")
credit_score = calculate_credit_score(probability_positive_class)
print(f"Credit Score: {credit_score:.2f}")
risk_category = assess_risk(probability_positive_class, credit_score)
print(f"Risk Category: {risk_category}")