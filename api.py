from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)
credit_scoring_model = joblib.load('best_gradient_boosting_model.pkl')
scaler = joblib.load('scaler.pkl')
poly = joblib.load('polyy.pkl')

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

    poly_features = poly.transform(data[['NetMonthlyIncome', 'LoanRepaymentBurden',
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
    return data


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



with open('feature_columns.pkl', 'rb') as f:
    feature_columns = joblib.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    df = pd.DataFrame([data])

    df = feature_engineering(df)
    df_scaled = scaler.transform(df)

    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        return jsonify({
            'error': 'Missing columns',
            'expected_columns': feature_columns,
            'missing_columns': missing_columns
        })

    prediction = credit_scoring_model.predict_proba(df_scaled)
    probability_positive_class = prediction[0, 1]

    credit_score = calculate_credit_score(probability_positive_class)
    risk_category = assess_risk(probability_positive_class, credit_score)

    return jsonify({
        'probability': probability_positive_class,
        'credit_score': credit_score,
        'risk_category': risk_category
    })
arima_model = None

@app.route('/forecast', methods=['POST'])
def forecast():
    global arima_model

    input_data = request.json

    df = pd.DataFrame(input_data)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.asfreq('W')

    df['Inflow'] = df['TransactionFees'] + df['AccountMaintenanceFees'] + df['LoanRepayments']
    df['Outflow'] = df['LoanDisbursements']
    df['NetCashFlow'] = df['Inflow'] - df['Outflow']

    train_data = df['NetCashFlow'][:-1]

    arima_model = ARIMA(train_data, order=(1, 1, 1)).fit()

    forecast_steps = 1
    forecast = arima_model.forecast(steps=forecast_steps)
    forecast_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(weeks=1), periods=forecast_steps, freq='W')
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'ForecastedNetCashFlow': forecast})
    forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m-%d')

    historical_data_output = {str(key): value for key, value in df['NetCashFlow'][:-forecast_steps].to_dict().items()}  # Only include historical data

    result = {
        "HistoricalData": historical_data_output,
        "ForecastData": forecast_df.set_index('Date').rename_axis(None).to_dict(orient='records'),
    }

    return jsonify(result)

@app.route('/customer', methods=['POST'])
def customer():
    input_data = request.json
    df = pd.DataFrame(input_data)

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

    best_model = joblib.load('best_random_forest_model.pkl')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    predictions = best_model.predict(X_scaled)

    gender_analysis = df.groupby('Gender').agg({
        'AverageTransactionAmount': 'mean',
        'TotalTransactionAmount': 'sum',
        'LoanAmountRequested': 'mean',
        'LoanAmountApproved': 'mean',
        'AccountBalance': 'mean',
        'AccountAge': 'mean',
        'MonthlyIncome': 'mean'
    }).reset_index()

    response = {
        'GenderAnalysis': []
    }

    for index, row in gender_analysis.iterrows():
        gender = 'Female' if row['Gender'] == 0 else 'Male'
        gender_info = {
            'Gender': gender,
            'AverageTransactionAmount': row['AverageTransactionAmount'],
            'TotalTransactionAmount': row['TotalTransactionAmount'],
            'LoanAmountRequested': row['LoanAmountRequested'],
            'LoanAmountApproved': row['LoanAmountApproved'],
            'AccountBalance': row['AccountBalance'],
            'AccountAge': row['AccountAge'],
            'MonthlyIncome': row['MonthlyIncome']
        }
        response['GenderAnalysis'].append(gender_info)

    response['Predictions'] = predictions.tolist()
    return jsonify(response)

password_strength_model = joblib.load('password_strength_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


@app.route('/predict_strength', methods=['POST'])
def predict_strength():
    try:
        data = request.get_json()
        password = data['password']
        password_vectorized = vectorizer.transform([password])

        predicted_strength = password_strength_model.predict(password_vectorized)
        strength_mapping = {0: 0, 1: 0.5, 2: 1}
        strength_score = strength_mapping.get(predicted_strength[0], 0)

        return jsonify({'strength_score': strength_score})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

