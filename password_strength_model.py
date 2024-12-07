import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib

data = {
    "password": [
        "12345", "password", "12345678", "letmein123", "Pa$$w0rd",
        "StrongPassword123!", "SuperSecure123!", "qwerty", "admin", "123qwe",
        "P@ssword123", "letme1n", "Welcome@123", "abcd1234", "abcdef"
    ],
    "strength": [0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 1, 1, 1, 0, 0]
}

df = pd.DataFrame(data)

print("Dataset Preview:")
print(df.head())

print("\nClass Distribution:")
print(df['strength'].value_counts())

sns.countplot(x='strength', data=df, palette='cool', hue='strength', legend=False)
plt.title("Password Strength Distribution")
plt.show()

X = df['password']
y = df['strength']

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
X = vectorizer.fit_transform(X)

smallest_class_size = min(df['strength'].value_counts())

smote = SMOTE(random_state=42, k_neighbors=min(5, smallest_class_size - 1))

X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nResampled Class Distribution:")
print(Counter(y_resampled))

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
gb_clf = GradientBoostingClassifier(random_state=42)

voting_clf = VotingClassifier(estimators=[
    ('rf', rf_clf),
    ('gb', gb_clf)
], voting='soft')

cross_val_scores = cross_val_score(voting_clf, X_train, y_train, cv=5, scoring='accuracy')
print("\nCross-Validation Accuracy Scores:", cross_val_scores)
print("\nAverage Cross-Validation Accuracy:", np.mean(cross_val_scores))

param_grid = {
    'rf__n_estimators': [50, 100, 200, 300],
    'rf__max_depth': [5, 10, 20, 30, None],
    'rf__min_samples_split': [2, 5, 10, 15],
    'rf__min_samples_leaf': [1, 2, 4, 6],
}

random_search = RandomizedSearchCV(
    estimator=voting_clf,
    param_distributions=param_grid,
    n_iter=50,
    cv=5,
    random_state=42,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
print("Best Hyperparameters:", random_search.best_params_)

# Step 6: Final Training and Testing
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# Balanced Accuracy Score
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
print("\nBalanced Accuracy Score:", balanced_accuracy)

probas = best_model.predict_proba(X_test)
fpr, tpr, _ = roc_curve(y_test, probas[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

custom_passwords = ["123456", "Hello@123", "WeakPass", "S3cur3!"]
custom_vectorized = vectorizer.transform(custom_passwords)
predictions = best_model.predict(custom_vectorized)

print("\nPassword Strength Predictions:")
for pwd, strength in zip(custom_passwords, predictions):
    print(f"Password: {pwd}, Strength: {strength}")

joblib.dump(best_model, 'password_strength_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')