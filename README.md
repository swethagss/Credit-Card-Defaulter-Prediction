# Credit Card Default Prediction

This repository contains a Jupyter Notebook for predicting credit card defaults using machine learning models. The analysis includes data preprocessing, exploratory analysis, model training, evaluation, and deployment.

---

## Table of Contents

- [Libraries and Dataset](#Libraries-and-Dataset)
- [Data Preprocessing](#Data-Preprocessing)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)
- [Model Training & Selection](#Model-Training-&-Selection)
- [Model Evaluation](#Model-Evaluation)
- [Deployment](#Deployment)
- [Conclusion](#Conclusion)


## Libraries and Dataset

The notebook begins by importing the necessary libraries for data manipulation, visualization, and machine learning, including:

- `pandas` and `numpy` for data manipulation
- `matplotlib` and `seaborn` for visualization
- `scikit-learn` for machine learning
- `xgboost` for gradient boosting

The dataset consists of credit card customer records, including features related to payment history, balance, and more. It is split into training and test sets.

## Data Preprocessing

Key preprocessing steps include:
- **Feature Scaling**: Normalized using MinMaxScaler.
- **Handling Imbalance**: Techniques like under-sampling and SMOTE.
- **Train-Test Split**: Ensuring robust evaluation.

## Exploratory Data Analysis

EDA is performed to understand the distribution and relationships within the dataset, including:
- Visualizing feature distributions
- Correlation analysis with the target variable
- Identifying and treating outliers

## Model Training & Selection

The primary model used for predicting credit card defaults is Logistic Regression, chosen for its simplicity and interpretability. After establishing a baseline performance with Logistic Regression, additional models were tested to explore potential improvements in predictive accuracy. These models include:

1. **Support Vector Machine (SVC)**: A robust classifier that seeks to maximize the margin between different classes.Improved class separation but was computationally expensive.
2. **Random Forest Classifier**: An ensemble learning method that combines multiple decision trees to improve model performance and reduce overfitting.Provided better handling of class imbalance and robustness to overfitting, with significant performance gains.
3. **XGBoost Classifier**: An advanced gradient boosting algorithm known for its efficiency and accuracy in classification tasks.Achieved the highest F1-scores and ROC-AUC metrics, making it the top performer.

Each of these models was trained on the same preprocessed training data, and their performance was compared against the baseline Logistic Regression model. The aim was to determine whether the use of more complex models could lead to better predictive accuracy and overall model performance.

## Best Model Selection: Attempt 2 with Under-Sampling and XGBoost

After evaluating various models and approaches, the best model was selected based on **Attempt 2**, which used **Under-Sampling with XGBoost**.

**Using SMOTE** didn’t effectively increase recall in this case, likely due to the introduction of noise or overlapping examples between classes. Instead, **Under-Sampling** was chosen to better focus the model on the minority class. This method reduces the dominance of the majority class, enabling the model to identify more true positives in the minority class, thus increasing recall. This approach is particularly beneficial when the priority is to capture more instances of the minority class, even if it means potentially increasing false positives.

In summary, the **XGBoost** model with under-sampling was selected as the best model, offering a balanced trade-off between precision and recall, and effectively addressing the class imbalance issue.

## Model Evaluation

### Baseline Model: Logistic Regression

The **Logistic Regression** model achieved an AUC of **0.7474**, indicating moderate ability in distinguishing between defaulters and non-defaulters. However, it struggled with recall for the minority class.

### Best Model: XGBoost Classifier

The **XGBoost** model, optimized with **Optuna**, outperformed the baseline with an AUC of 0.7764. This model showed improved recall and better overall discrimination between classes.

### ROC Curve and Gini Coefficient Analysis
- **AUC:** The XGBoost model achieved an AUC of **0.7764**, suggesting good discrimination between events and non-events.
- **Gini Coefficient:** The model's Gini Coefficient was calculated as **0.5529**. This further confirms that the model is effective in its predictions. The Gini Coefficient ranges from -1 to 1, where a value closer to 1 signifies a perfect model, 0 indicates a model with no discriminative power, and -1 signifies a perfectly incorrect model.

### Insights
- **AUC of 0.77:** The model is good at distinguishing between events and non-events.
- **Gini Coefficient of 0.55:** This value supports the conclusion that the XGBoost model is effective in its predictions.

In summary, the **XGBoost model**, with its higher AUC and solid Gini Coefficient, emerged as the superior model, providing better predictive accuracy and class discrimination compared to the baseline Logistic Regression model.

## Deployment
Deploy the model using a Streamlit app `(app.py)`. The app allows users to input various data about credit card holders to predict whether a customer will default or not. To use the app, follow the link provided below:


## Conclusion
The notebook concludes with a summary of the model performances, highlighting the best-performing model and potential areas for further improvement.



