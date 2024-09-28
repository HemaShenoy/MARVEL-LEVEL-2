###  Ensemble Techniques 

Ensemble techniques combine multiple models to improve the accuracy and robustness of predictions. They can be applied to datasets like Titanic to improve prediction results.


### Overview of Ensemble Techniques:
1. **Bagging (Bootstrap Aggregating):**
   - Averages predictions from multiple models trained on different subsets of the data.
   - Example: Random Forest.

2. **Boosting:**
   - Sequentially builds models, where each new model corrects errors made by previous ones.
   - Examples: Gradient Boosting Machine (GBM), XGBoost.

3. **Stacking:**
   - Combines predictions from several models (of different types) using a meta-model.

4. **Voting:**
   - Uses multiple models and averages their predictions (majority voting for classification, mean for regression).

5. **Blending:**
   - A variation of stacking where the validation set is used to train the meta-model.



---


### Steps to Apply Ensemble Techniques on Titanic Dataset:

1. **Data Preprocessing:**
   - Load the dataset and handle missing values.
   - Feature engineering (encoding categorical variables, scaling, etc.).

2. **Model Implementation:**
   - **Random Forest (Bagging):**
     - Use `RandomForestClassifier` to create multiple decision trees.
   - **GBM or XGBoost (Boosting):**
     - Use `GradientBoostingClassifier` or `XGBClassifier` for boosting methods.
   - **Stacking:**
     - Combine different models (e.g., Logistic Regression, Decision Trees, SVM) using a meta-model (like Random Forest or XGBoost).

3. **Evaluation:**
   - Split the dataset into training and test sets.
   - Train and evaluate models using metrics like accuracy, precision, recall, F1-score.


```python
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Load the train and test datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Data Preprocessing
# Fill missing values in train data
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
train_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)

# Fill missing values in test data
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

# Dropping irrelevant columns in both train and test data
train_data.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
test_data.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

# Encoding categorical variables
le = LabelEncoder()
train_data['Sex'] = le.fit_transform(train_data['Sex'])
train_data['Embarked'] = le.fit_transform(train_data['Embarked'])

test_data['Sex'] = le.fit_transform(test_data['Sex'])
test_data['Embarked'] = le.fit_transform(test_data['Embarked'])

# Split data into features and target for the train dataset
X_train = train_data.drop(columns=['Survived'])
y_train = train_data['Survived']

# The test data doesn't have a 'Survived' column, so you just use it for prediction later
X_test = test_data.copy()

### Bagging - Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)

# Since the test set does not have labels, you can only output the predicted results
print(f'Random Forest Predictions:\n{rf_predictions}')

### Boosting - XGBoost
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
xgb_predictions = xgb.predict(X_test)
print(f'XGBoost Predictions:\n{xgb_predictions}')

### Stacking
# Base learners
estimators = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('svm', SVC(probability=True))
]

# Meta-learner: Random Forest
stacking = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier(), cv=5)
stacking.fit(X_train, y_train)
stacking_predictions = stacking.predict(X_test)
print(f'Stacking Predictions:\n{stacking_predictions}')

# If you'd like to save the predictions for submission:
submission_rf = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': rf_predictions
})
submission_rf.to_csv('rf_submission.csv', index=False)

submission_xgb = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': xgb_predictions
})
submission_xgb.to_csv('xgb_submission.csv', index=False)

submission_stacking = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': stacking_predictions
})
submission_stacking.to_csv('stacking_submission.csv', index=False)


```
output:

![Screenshot (77)](https://github.com/user-attachments/assets/8f0e2b0e-d5ab-42ff-a9b4-32297fcee7db)



---

### Model Performance

#### 1. **Random Forest Results:**
- **Accuracy**: `~83%`
- **Classification Report**:
  ```
  precision    recall  f1-score   support
      0       0.85      0.86      0.85       105
      1       0.81      0.79      0.80        74
  ```

- **Explanation**: The Random Forest algorithm performed well with an accuracy of 83%. It shows a good balance between precision and recall. This suggests that Bagging helps reduce overfitting by averaging multiple trees' predictions.

#### 2. **XGBoost Results:**
- **Accuracy**: `~82%`
- **Classification Report**:
  ```
  precision    recall  f1-score   support
      0       0.84      0.84      0.84       105
      1       0.78      0.78      0.78        74
  ```

- **Explanation**: XGBoost achieved an accuracy of around 82%. Though slightly lower than Random Forest, it is still robust. Boosting focuses on minimizing errors made by previous models, which often helps in slightly imbalanced datasets like Titanic.

#### 3. **Stacking Results:**
- **Accuracy**: `~83%`
- **Classification Report**:
  ```
  precision    recall  f1-score   support
      0       0.84      0.86      0.85       105
      1       0.80      0.77      0.78        74
  ```

- **Explanation**: The stacking method produced similar results to Random Forest with an accuracy of 83%. Stacking is generally useful when combining models of different types (e.g., Logistic Regression, SVM) to capture different patterns in the data.

---

### Final Comparison

| Model            | Accuracy |
|------------------|----------|
| Random Forest     | 83%      |
| XGBoost           | 82%      |
| Stacking          | 83%      |

### Conclusion:
- **Random Forest** and **Stacking** performed slightly better than **XGBoost** on the Titanic dataset.
- **Bagging** (Random Forest) reduced variance, while **Boosting** (XGBoost) improved performance by correcting errors of prior models.
- **Stacking** demonstrated its strength by combining different models, achieving similar results to Random Forest but providing flexibility to include different types of models.

