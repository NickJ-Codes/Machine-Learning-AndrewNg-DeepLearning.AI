"""
Learn and implement XGBoost
Source: https://www.youtube.com/watch?v=GrJP9FLV3FE
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import  matplotlib.pyplot as plt

def clean_data(dataframe, verbose = False):
    #Drop columns that perfectly predict churn outcomes, i.e., they will exit if they have done an exit interview
    dataframe.drop(['Churn Label', 'Churn Score', 'CLTV', 'Churn Reason'],
                   axis = 1, inplace = True)

    # Some other columns only contain one value, and therefore won't be useful for classification
    # so drop these single value columns
    if verbose: print(f"{'-'*15} Dropping columns with single only values {'-'*15}")
    for column in dataframe.columns:
        numUniqueItems = len(dataframe[column].unique())
        if numUniqueItems == 1:
            dataframe.drop([column], axis = 1, inplace = True)
            if verbose: print(f"dropping column: {column}")

    # Drop customerID since this willl be unique for every customer, and drop lat long column since there are separate columns for lat and long
    # replace blank spaces too
    dataframe.drop(['CustomerID', 'Lat Long'], axis = 1, inplace = True)
    dataframe['City'] = dataframe['City'].str.replace(' ', '_', regex = True)

    # Replace spaces in column names with underscores
    dataframe.columns = dataframe.columns.str.replace(' ', '_')

    if verbose: print(dataframe.head())
    return dataframe

def replace_missing_data(dataframe, verbose = False):
    dataframe.loc[(dataframe['Total_Charges'] == ' '), 'Total_Charges'] =0
    dataframe['Total_Charges'] = pd.to_numeric(dataframe['Total_Charges'])

    dataframe.replace(' ', '_', regex = True, inplace = True)
    return dataframe

def splitXy(dataframe):
    X = dataframe.drop('Churn_Value', axis = 1).copy()
    # print(f"X data: {X.head()}")
    y = dataframe['Churn_Value'].copy()
    # print(f"y data: {y.head()}")

    return X, y

def main():
    # Import the data
    df_raw = pd.read_csv('../data/Telco_customer_churn.csv')

    df = clean_data(df_raw,verbose = False)
    df = replace_missing_data(df, verbose = False)
    print(df.dtypes)

    X, y = splitXy(df)

    X_encoded = pd.get_dummies(X, columns = ['City',
                                             'Gender',
                                             'Senior_Citizen',
                                             'Partner',
                                             'Dependents',
                                             'Phone_Service',
                                             'Multiple_Lines',
                                             "Internet_Service",
                                             'Online_Security',
                                             "Online_Backup",
                                             'Device_Protection',
                                             'Tech_Support',
                                             'Streaming_TV',
                                             'Streaming_Movies',
                                             'Contract',
                                             'Paperless_Billing',
                                             'Payment_Method'])
    print(X_encoded.head())
    print(y.unique())

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42, stratify=y)

    #hyperparameters, manualyl figured out through gridcV search
    gamma = 0.25
    learn_rate = 0.1
    max_depth = 4
    reg_lambda = 10

    # Initialize XGBoost classifier with correct parameters
    clf_xgb = xgb.XGBClassifier(random_state=42,  # Changed from seed to random_state
        objective='binary:logistic',
        gamma = gamma,
        eval_metric = 'aucpr',
        learn_rate = learn_rate,
        max_depth = max_depth,
        reg_lambda = reg_lambda,
        scale_pos_weight =3,
        subsample = 0.9,
        colsample_bytree = 0.5,
    )

    # Fit the model
    clf_xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=True
    )

    # Get predictions
    y_pred = clf_xgb.predict(X_test)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create and plot confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Did not leave", "Left"]
    )

    # Plot the confusion matrix
    disp.plot(cmap='Blues', values_format='d')

    # Add title and display the plot
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


