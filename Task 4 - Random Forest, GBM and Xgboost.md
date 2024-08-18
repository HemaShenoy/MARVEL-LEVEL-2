
## 1. Random Forest

### Understanding Random Forest
Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the mode of the classes (classification) or mean prediction (regression) of the individual trees. It reduces overfitting and improves accuracy by averaging out errors from individual trees.

**Key Points:**
- **Bagging**: Random Forest uses bootstrap aggregation (bagging) to create subsets of the data with replacement.
- **Random Subset of Features**: At each node, a random subset of features is selected to determine the best split.
- **Ensemble Voting**: The final prediction is made by averaging (in regression) or majority voting (in classification) across all trees.

### Implementing Random Forest
Here's an implementation using the `scikit-learn` library for a classification problem.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(20,10))
plot_tree(rf.estimators_[0], feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
```

![Screenshot (52)](https://github.com/user-attachments/assets/d75e8160-bfd0-49aa-adfc-29c1f2954b17)



![image](https://github.com/user-attachments/assets/cad0ffc1-b554-4660-86cd-4a8340268cbc)


### Visualization
The above code includes a tree plot from the Random Forest, which helps visualize the decisions made at each node.

---

## 2. Gradient Boosting Machine (GBM)

### Understanding GBM
Gradient Boosting Machine is an ensemble technique that builds models sequentially, where each new model attempts to correct the errors of the previous ones. Unlike Random Forest, GBM builds trees one at a time, with each tree focusing on correcting the residual errors of the previous trees.

**Key Points:**
- **Boosting**: Trees are built sequentially.
- **Loss Function**: The algorithm optimizes a loss function (e.g., mean squared error for regression).
- **Learning Rate**: Controls the contribution of each tree to the final prediction.

### Implementing GBM
Here's how you can implement GBM using the `scikit-learn` library.

```python
from sklearn.ensemble import GradientBoostingClassifier


gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

gbm.fit(X_train, y_train)


y_pred_gbm = gbm.predict(X_test)

print("GBM Accuracy:", accuracy_score(y_test, y_pred_gbm))
print("GBM Classification Report:\n", classification_report(y_test, y_pred_gbm))


plt.figure(figsize=(10, 6))
plt.barh(iris.feature_names, gbm.feature_importances_)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in GBM")
plt.show()
```
![Screenshot (53)](https://github.com/user-attachments/assets/443bfeba-52ac-4867-bf3f-8e9541879f16)

![image](https://github.com/user-attachments/assets/415603c6-358d-40a9-a7b5-09ea8e89fec7)

### Visualization
The feature importance plot helps identify which features are most influential in the predictions made by the GBM model.

---

## 3. XGBoost

### Understanding XGBoost
XGBoost (Extreme Gradient Boosting) is an optimized version of the gradient boosting technique. It uses more regularization techniques, making it less prone to overfitting and more efficient. XGBoost is known for its speed and performance.

**Key Points:**
- **Regularization**: Adds regularization terms to control overfitting.
- **Parallel Processing**: Supports parallel processing, making it faster.
- **Handling Missing Data**: Can handle missing data more effectively.

### Implementing XGBoost
Below is an example of how to implement XGBoost using the `xgboost` library.

```python
import xgboost as xgb
from xgboost import plot_importance


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'multi:softmax',
    'num_class': 3,
    'eval_metric': 'mlogloss'
}


bst = xgb.train(params, dtrain, num_boost_round=100)


y_pred_xgb = bst.predict(dtest)


print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))


plot_importance(bst)
plt.show()
```

![Screenshot (54)](https://github.com/user-attachments/assets/9b2f00bd-65e2-4322-b272-6d704cf59ea5)

![image](https://github.com/user-attachments/assets/89483e49-26f6-4f12-ba11-a794612ad57a)

### Visualization
The importance plot in XGBoost provides a detailed look at which features contribute the most to the model's predictions.

---

### Conclusion
Each of these algorithms—Random Forest, GBM, and XGBoost—has its own strengths and suitable use cases. Random Forest is great for reducing overfitting, GBM is powerful for sequential learning, and XGBoost is efficient with enhanced performance and regularization.

This report includes code implementations and visualizations to help understand how these models work and how to apply them to a dataset.

---
