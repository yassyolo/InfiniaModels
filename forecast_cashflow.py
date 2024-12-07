import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

data = {
    'Date': pd.date_range(start='2023-01-01', periods=20, freq='W'),
    'TransactionFees': [5000, 3000, 4000, 2000, 1500, 2500, 3000, 4000, 3500, 2800, 5000, 6000, 4200, 3800, 4500, 6000,
                        3200, 4800, 5100, 5300],
    'AccountMaintenanceFees': [1000, 2000, 2000, 1500, 6000, 4000, 3000, 4000, 5000, 4500, 3500, 4000, 4500, 3000, 3500,
                               4000, 3800, 3500, 4300, 4800],
    'LoanDisbursements': [400000, 80000, 0, 0, 2000, 0, 1000, 1500, 0, 3000, 2000, 0, 4000, 0, 0, 0, 0, 1000, 0, 1500],
    'LoanRepayments': [500000, 2000, 3000, 1000, 0, 5000, 6000, 4000, 5000, 6000, 7000, 7500, 6000, 5000, 5500, 6000,
                       6500, 7000, 6000, 6000]
}

df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

df['Inflow'] = df['TransactionFees'] + df['AccountMaintenanceFees'] + df['LoanRepayments']
df['Outflow'] = df['LoanDisbursements']
df['NetCashFlow'] = df['Inflow'] - df['Outflow']


def check_stationarity(series, name):
    result = adfuller(series.dropna())
    adf_stat = result[0]
    p_value = result[1]
    critical_values = result[4]

    print(f"ADF Statistic for '{name}': {adf_stat:.4f}")
    print(f"P-value for '{name}': {p_value:.4e}")
    print("Critical Values for Test:")
    for key, value in critical_values.items():
        print(f"   {key} critical value: {value:.4f}")

    if p_value <= 0.05:
        print(f"Conclusion: The series '{name}' is stationary (p-value <= 0.05).\n")
    else:
        print(
            f"Conclusion: The series '{name}' is not stationary (p-value > 0.05). Consider applying transformations or differencing.\n")


check_stationarity(df['NetCashFlow'], "Original Series")

min_val = df['NetCashFlow'].min()
shift_value = abs(min_val) + 1 if min_val <= 0 else 0
df['Shifted'] = df['NetCashFlow'] + shift_value
check_stationarity(df['Shifted'], "Shifted Series")

try:
    df['BoxCox'], lam = stats.boxcox(df['Shifted'])
    print(f"Box-Cox Transformation Lambda: {lam:.4f}")
    check_stationarity(df['BoxCox'], "Box-Cox Transformed")
except ValueError as e:
    print("Box-Cox transformation failed:", e)

df['FirstDiff'] = df['NetCashFlow'].diff()
check_stationarity(df['FirstDiff'], "First Differencing")

if 'BoxCox' in df.columns:
    df['BoxCoxDiff'] = df['BoxCox'].diff()
    check_stationarity(df['BoxCoxDiff'], "Box-Cox + First Differencing")

plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(df['NetCashFlow'], label="Original Series")
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(df['Shifted'], label="Shifted Series")
plt.legend()

if 'BoxCox' in df.columns:
    plt.subplot(4, 1, 3)
    plt.plot(df['BoxCox'], label="Box-Cox Transformed")
    plt.legend()

plt.subplot(4, 1, 4)
plt.plot(df['FirstDiff'], label="First Differencing")
plt.legend()

plt.tight_layout()
plt.show()
