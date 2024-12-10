The InfiniaModels repository, written entirely in Python, is designed to support the InfiniaBankSystem repository. It contains various models essential for the banking system's functionality.

Models Overview
Card Fraud Detection Model: Utilizes logistic regression and random forest classifiers to detect fraudulent card transactions. The model balances the dataset with undersampling and optimizes using grid search for the best parameters.

Cashflow Prediction Model: Analyzes and predicts the bank's cash flow for the upcoming weeks using various financial indicators and machine learning techniques.

Password Strength Checker: Uses TF-IDF vectorization and a voting classifier (Random Forest and Gradient Boosting) to evaluate the strength of passwords. The model is trained with SMOTE to handle class imbalances.

Customer Analysis Model: Provides detailed analysis based on gender, including transaction frequency, average transaction amount, loan amounts requested and approved, account balance, and monthly income. The model uses RandomForestRegressor for predictions.

Loan Application Model: Manages loan applications and statuses, including loan approval predictions and risk assessments.

Chatbot: Uses OpenAI to provide customers with a good experience in receiving guidance within the system.
