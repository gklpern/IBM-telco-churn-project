# IBM-telco-churn-project
This project is about predicting customer churn in the telecommunications industry using machine learning techniques

##Requirements
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
torch


You need to Install required packages:
pip install -r requirements.txt



In this project, we aim to develop a machine learning model that predicts whether a customer will churn based on a such features:

    Service usage patterns
    Customer demographics
    Billing and payment information
    Identify high-risk customers early
    Take proactive actions (e.g., targeted offers)
    Reduce churn rates
    Increase customer satisfaction and overall profitability

## Dataset Description

We used the IBM Telco Customer Churn dataset, which includes detailed information on over 7,000 customers and consists of 35 columns. The dataset includes:

    Customer Identity and Location: Customer ID, Country, State, City, Zip Code
    Geographical Information: Latitude, Longitude, Lat Long
    Demographics: Gender, Senior Citizen status, Partner, Dependents
    Service Details: Tenure (in months), Phone Service, Internet Service, Online Security, Online Backup, etc.
    Contract and Billing: Contract type, Paperless Billing, Payment Method, Monthly and Total Charges
    Churn Information: Churn Label, Churn Value, Churn Score, CLTV, Churn Reason


## Data Preprocessing

The preprocessing steps were executed in the 1_EDA_preprocessing.ipynb notebook.

    Missing Values Handling
        Specifically handled Total Charges column by dropping null values
        Rows with missing values were removed
        
    Encoding Categorical Variables
        Used Label Encoding to transform object  types into numerical values
        Checked dtype of each column

    Feature Scaling
        Applied StandardScaler to normalize numerical columns

    Feature Selection
        Dropped non-informative fields
        Removed features with a single unique value
        
    Exploratory Data Analysis (EDA)
        Used correlation matrix to observe relationships
        Visualized feature distributions and churn relationships through multiple plots

    Final Dataset
        The cleaned and processed dataset was saved as Processed_telco_customer.csv



## Feature Engineering

Feature engineering was a critical step in enhancing the predictive capability of the dataset. The goal was to create new, meaningful features and transform existing ones to better capture patterns in customer behavior and identify potential churn risks.

Several new features were engineered throughout the process:

    Estimated Lifetime Charges: This feature was created by multiplying Monthly Charges with Tenure Months, representing the expected total amount a customer might have paid if they had remained subscribed.

    Additional Charges: Although this feature was initially developed to capture extra billing beyond regular charges, further analysis showed it did not contribute significantly to model performance and was removed.

    Zone ID: To detect regional churns, customers were grouped into five geographic clusters based on their Latitude and Longitude using K-Means Clustering.

    Bundled Services Count: This feature represented customer engagement by counting how many services a customer subscribed

    Synthetic Churn Score: A custom churn risk indicator was created by combining multiple service usage metrics and customer characteristics with weighted importance. This aggregated score served as a proxy for churn propensity and was designed to improve model interpretability.

![image](https://github.com/user-attachments/assets/8b64dab5-06fa-4480-b158-425295a8dec2)


![image](https://github.com/user-attachments/assets/95daff75-ca74-41ea-879b-978b052628ff)


##Models Used And Rationale


In the Churn Prediction project, several different models were implemented to evaluate their effectiveness in predicting customer churn. The aim was to observe how each model handles the problem and which approaches yield the best results.
Logistic Regression (Baseline Model)

A basic Logistic Regression model was used initially without any feature scaling or hyperparameter tuning. Logistic regression is a well-known linear model that provides a solid baseline for binary classification tasks. Although it may not outperform tree-based models in some cases, it offered a clear baseline performance for this project.
Logistic Regression with Scaling and GridSearchCV

To improve upon the baseline, hyperparameter tuning was applied using GridSearchCV, combined with feature scaling. Since logistic regression benefits from scaled features for optimization, this approach sought to maximize the model’s predictive power by finding the best parameters.
Basic Neural Network With PyTorch (MLP)

A more complex model was designed using a Multi-Layer Perceptron (MLP) implemented in PyTorch. This neural network had 4 hidden layers with decreasing units (128 → 64 → 32 → 16) and ReLU activation functions. The output layer used a sigmoid activation for binary classification. This model aimed to capture more complex patterns and answer whether deeper models outperform simpler ones for this problem.
Logistic Regression and Neural Network without 'Churn Score'

During further analysis, it was realized that in real-world scenarios, some engineered features like the "Churn Score" may not be available. Hence, this feature was removed, and the logistic regression and neural network models were retrained to observe the impact of excluding this highly correlated feature on model performance.
Advanced Neural Network (with Dropout and Batch Normalization)

After dropping the "Churn Score," performance decreased slightly. To compensate, a deeper neural network architecture with batch normalization and dropout (drop rate = 0.3) was implemented. These regularization techniques aimed to reduce overfitting and improve generalization, resulting in a more robust model despite the absence of the "Churn Score."
Model Files and Storage

All trained models were saved for reproducibility and easy access. Scikit-learn models were stored with joblib, while PyTorch models were saved using torch.save. Models such as:

    logreg_scaled_dropped.pkl (scaled logistic regression without Churn Score)
    logreg_scaled_gridsearch.pkl (scaled logistic regression with hyperparameter tuning)
    logreg_no_scaling.pkl (basic logistic regression without scaling)
    ann_basic.pth (basic ANN with Churn Score)
    ann_complex_dropped_churnscore.pth (advanced ANN without Churn Score)

## Mode Evaluation

Multiple models were developed and evaluated to understand the effects of feature and model selection.

    The basic logistic regression achieved an accuracy of 89%, precision 0.87, recall 0.86, and F1-score 0.87. However, it struggled with predicting the minority class (churn=1), showing low recall due to data imbalance.
    Applying feature scaling and GridSearchCV improved results to 93% accuracy, precision 0.90, recall 0.91, and F1-score 0.91, reducing the impact of class imbalance.
    The basic neural network (MLP) with scaled data scored slightly lower, with 88% accuracy, precision, recall, and F1 all at 0.88. This showed that more complex models do not always guarantee better performance and that inappropriate model choice can worsen results.

After removing the "Churn Score" feature (which had shown very high correlation during exploratory data analysis), performance dropped:

    Logistic regression accuracy fell to 80%, with precision 0.76, recall 0.65, and F1-score 0.70.
    The advanced ANN with dropout and batch normalization (trained without the Churn Score) achieved better results than logistic regression, with 84% accuracy, 0.80 precision, 0.72 recall, and 0.76 F1-score.

This demonstrated that while the "Churn Score" was an important predictive feature, a well-regularized neural network could partially compensate for its removal.

![image](https://github.com/user-attachments/assets/0db9d8e8-1366-41d6-9729-09be7d59303b)

![image](https://github.com/user-attachments/assets/81352df4-65df-449e-87e7-2f310101d549)


