

# Hyperparameter Tuning 

## 1. Introduction to Hyperparameter Tuning

Hyperparameter tuning is a critical step in the machine learning pipeline that involves optimizing the parameters that govern the training process of a model. Unlike model parameters, which are learned during training (like weights in a neural network), hyperparameters are set prior to training and significantly influence model performance. Common hyperparameters include learning rates, regularization parameters, and the number of estimators in ensemble methods.

Effective hyperparameter tuning can lead to improved model accuracy, better generalization to unseen data, and reduced overfitting. Techniques for hyperparameter tuning include:

- **Grid Search**: An exhaustive search over specified hyperparameter values.
- **Random Search**: A randomized approach to search hyperparameter combinations.
- **Bayesian Optimization**: A more sophisticated method that uses past evaluations to inform future searches.

## 2. Example: Titanic Dataset

### Data Description

The Titanic dataset is a well-known dataset used for binary classification tasks, specifically predicting whether a passenger survived based on features such as age, sex, and ticket class. Key features include:

- **Survived**: Target variable (1 = survived, 0 = did not survive)
- **Pclass**: Ticket class (1st, 2nd, or 3rd)
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger
- **Fare**: Ticket fare
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

### Code Implementation

Here is the Python code snippet used for hyperparameter tuning with the Titanic dataset:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
# Load dataset
data = pd.read_csv('train.csv')

# Display the first few rows of the dataset
print(data.head())
# Drop irrelevant features
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Fill missing values for 'Age' and 'Embarked'
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Convert categorical variables to numerical
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

# Define features and target variable
X = data.drop('Survived', axis=1)
y = data['Survived']
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train a Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Define the hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           scoring='accuracy', cv=3, verbose=2, n_jobs=-1)

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Display the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)
# Train the model with the best parameters
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

# Make predictions
y_pred_tuned = best_rf.predict(X_test)

# Evaluate the tuned model
print("Tuned Model Accuracy:", accuracy_score(y_test, y_pred_tuned))
print(classification_report(y_test, y_pred_tuned))

```

Output:


![Screenshot (78)](https://github.com/user-attachments/assets/98848ee9-ac56-4fdf-ad91-f5c120eabc00)


### Results and Model Performance

**Initial Model Performance:**
- **Accuracy:** 0.7989 (approximately 79.89%)
- **Classification Report:**
  - Precision (Class 0): 0.82
  - Recall (Class 0): 0.84
  - F1-score (Class 0): 0.83
  - Support (Class 0): 105
  - Precision (Class 1): 0.76
  - Recall (Class 1): 0.74
  - F1-score (Class 1): 0.75
  - Support (Class 1): 74
- **Overall Accuracy:** 0.80

**Best Hyperparameters from Grid Search:**
- **n_estimators:** 100
- **max_depth:** None
- **min_samples_split:** 10

**Tuned Model Performance:**
- **Tuned Model Accuracy:** 0.8379 (approximately 83.79%)
- **Classification Report:**
  - Precision (Class 0): 0.83
  - Recall (Class 0): 0.91
  - F1-score (Class 0): 0.87
  - Support (Class 0): 105
  - Precision (Class 1): 0.86
  - Recall (Class 1): 0.73
  - F1-score (Class 1): 0.79
  - Support (Class 1): 74
- **Overall Accuracy:** 0.84

The hyperparameter tuning process led to a notable improvement in the model's performance, with accuracy increasing from approximately 79.89% to 83.79%.

## 4. Conclusion

The hyperparameter tuning process significantly improved the model's performance in predicting Titanic survival. The accuracy increased from approximately 79.89% to 83.79%, demonstrating the importance of fine-tuning hyperparameters to enhance model effectiveness.

---

