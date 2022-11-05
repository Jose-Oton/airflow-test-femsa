from typing import Callable, Optional

import json
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from sklearn.model_selection import train_test_split, KFold, cross_val_score  # to split the data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, \
    fbeta_score  # To evaluate our model
from sklearn.model_selection import GridSearchCV
# Algorithmns models to be compared
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np  # Math library


def build_python_operator(dag: DAG, task_name: str, function_name: Callable,
                          args: Optional[dict] = None, ) -> PythonOperator:
    python_function = PythonOperator(
        task_id='{}'.format(task_name),
        python_callable=function_name,
        op_kwargs=args,
        provide_context=True,
        dag=dag
    )
    return python_function


def read_csv():
    df_credit = pd.read_csv("/opt/airflow/plugins/includes/german_credit_data.csv", index_col=0)
    return df_credit


def csv_create_categorical_variables():
    df_credit = pd.read_csv("/opt/airflow/plugins/includes/german_credit_data.csv", index_col=0)
    # Let's look the Credit Amount column
    interval = (18, 25, 35, 60, 120)

    cats = ['Student', 'Young', 'Adult', 'Senior']
    df_credit["Age_cat"] = pd.cut(df_credit.Age, interval, labels=cats)
    df_credit.to_csv('/opt/airflow/plugins/includes/dataset_categorical_var.csv')

    return 0


def transform_data():
    df_credit = pd.read_csv("/opt/airflow/plugins/includes/dataset_categorical_var.csv", index_col=0)
    print(df_credit)

    df_credit['Saving accounts'] = df_credit['Saving accounts'].fillna('no_inf')
    df_credit['Checking account'] = df_credit['Checking account'].fillna('no_inf')

    # Purpose to Dummies Variable
    df_credit = df_credit.merge(pd.get_dummies(df_credit.Purpose, drop_first=True, prefix='Purpose'), left_index=True,
                                right_index=True)
    # Sex feature in dummies
    df_credit = df_credit.merge(pd.get_dummies(df_credit.Sex, drop_first=True, prefix='Sex'), left_index=True,
                                right_index=True)
    # Housing get dummies
    df_credit = df_credit.merge(pd.get_dummies(df_credit.Housing, drop_first=True, prefix='Housing'), left_index=True,
                                right_index=True)
    # Housing get Saving Accounts
    df_credit = df_credit.merge(pd.get_dummies(df_credit["Saving accounts"], drop_first=True, prefix='Savings'),
                                left_index=True, right_index=True)
    # Housing get Risk
    df_credit = df_credit.merge(pd.get_dummies(df_credit.Risk, prefix='Risk'), left_index=True, right_index=True)
    # Housing get Checking Account
    df_credit = df_credit.merge(pd.get_dummies(df_credit["Checking account"], drop_first=True, prefix='Check'),
                                left_index=True, right_index=True)
    # Housing get Age categorical
    df_credit = df_credit.merge(pd.get_dummies(df_credit["Age_cat"], drop_first=True, prefix='Age_cat'),
                                left_index=True, right_index=True)
    df_credit.to_csv('/opt/airflow/plugins/includes/transform_data.csv')

    return 0


def delete_cols():
    # Excluding the missing columns
    df_credit = pd.read_csv("/opt/airflow/plugins/includes/transform_data.csv", index_col=0)

    del df_credit["Saving accounts"]
    del df_credit["Checking account"]
    del df_credit["Purpose"]
    del df_credit["Sex"]
    del df_credit["Housing"]
    del df_credit["Age_cat"]
    del df_credit["Risk"]
    del df_credit['Risk_good']
    df_credit.to_csv('/opt/airflow/plugins/includes/delete_cols.csv')

    return 0


def train_model():
    df_credit = pd.read_csv("/opt/airflow/plugins/includes/delete_cols.csv", index_col=0)

    df_credit['Credit amount'] = np.log(df_credit['Credit amount'])
    # Creating the X and y variables
    X = df_credit.drop('Risk_bad', 1).values
    y = df_credit["Risk_bad"].values

    # Spliting X and y into train and test version
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # Seting the Hyper Parameters
    param_grid = {"max_depth": [3, 5, 7, 10, None],
                  "n_estimators": [3, 5, 10, 25, 50, 150],
                  "max_features": [4, 7, 15, 20]}

    # Creating the classifier
    model = RandomForestClassifier(random_state=2)

    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='recall', verbose=4)
    grid_search.fit(X_train, y_train)
    rf = RandomForestClassifier(max_depth=None, max_features=10, n_estimators=15, random_state=2)

    # trainning with the best params
    rf.fit(X_train, y_train)
    # Testing the model
    # Predicting using our  model
    y_pred = rf.predict(X_test)
    final_y = pd.DataFrame( {'y_test': y_test, 'y_pred': y_pred})
    final_y.to_csv('/opt/airflow/plugins/includes/final_y.csv')
    return 0


def write_results():
    df_credit = pd.read_csv("/opt/airflow/plugins/includes/final_y.csv", index_col=0)

    y_test = df_credit['y_test']
    y_pred = df_credit['y_pred']
    print('accuracy_score', accuracy_score(y_test, y_pred))
    print("\n")
    print('confusion_matrix',confusion_matrix(y_test, y_pred))
    print("\n")
    print('fbeta_score',fbeta_score(y_test, y_pred, beta=2))
    data = {'accuracy_score': accuracy_score(y_test, y_pred),
            'confusion_matrix': str(confusion_matrix(y_test, y_pred)),
            'fbeta_score': fbeta_score(y_test, y_pred, beta=2)}
    final_results = pd.DataFrame(data, index=[0])
    print(final_results)
    final_results.to_csv('/opt/airflow/plugins/includes/final_results.csv', index=False)
    return 0
