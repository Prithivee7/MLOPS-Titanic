import os
from azureml.core import Datastore, Dataset
from azureml.core.run import Run
import argparse
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


class TitanicClassification():
    def __init__(self, args):
        # Initialising run and workspace
        self.args = args
        self.run = Run.get_context()
        self.workspace = self.run.experiment.workspace
        os.makedirs('./model_metas', exist_ok=True)

    def get_files_from_datastore(self, container_name, file_name):
        datastore_paths = [
            (self.datastore, os.path.join(container_name, file_name))]
        data_ds = Dataset.Tabular.from_delimited_files(path=datastore_paths)
        dataset_name = self.args.dataset_name
        if dataset_name not in self.workspace.datasets:
            data_ds = data_ds.register(workspace=self.workspace,
                                       name=dataset_name,
                                       description=self.args.dataset_desc,
                                       tags={'format': 'CSV'},
                                       create_new_version=True)
        return data_ds

    def create_pipeline(self):
        # core logic
        self.datastore = Datastore.get(
            self.workspace, self.workspace.get_default_datastore().name)
        input_ds = self.get_files_from_datastore(
            self.args.container_name, self.args.input_train_csv)
        train_data = input_ds.to_pandas_dataframe()

        input_ds = self.get_files_from_datastore(
            self.args.container_name, self.args.input_test_csv)
        test_data = input_ds.to_pandas_dataframe()

        features_drop = ['PassengerId', 'Name', 'Cabin', 'Ticket']
        X_train = train_data.drop(features_drop, axis=1)
        y_train = train_data['Survived']
        X_train.drop('Survived', axis=1, inplace=True)
        X_test = test_data.drop(features_drop, axis=1)

        # Conversion of categorical values of sex feature to numerical values
        mapping = {'male': 1, 'female': 0}
        X_train['Sex'] = X_train['Sex'].map(mapping)
        X_test['Sex'] = X_test['Sex'].map(mapping)

        X_train.Age.fillna(X_train.Age.median(), inplace=True)
        X_test.Age.fillna(X_test.Age.median(), inplace=True)

        # Filling the missing values in Embarked feature with the most repeated value
        X_train.fillna(value='S', inplace=True)

        # Filling the missing values of Fare feature with the median in test data
        X_test.Fare.fillna(X_test.Fare.median(), inplace=True)

        mapping = {'S': 0, 'C': 1, 'Q': 2}
        X_train['Embarked'] = X_train['Embarked'].map(mapping)
        X_test['Embarked'] = X_test['Embarked'].map(mapping)

        # K-Fold Cross Validation
        k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

        # Logistic Regression
        logisticRegression = LogisticRegression()
        score = cross_val_score(
            logisticRegression, X_train, y_train, cv=k_fold)
        print(score)

        # kNN
        kNN = KNeighborsClassifier(n_neighbors=20)
        score = cross_val_score(kNN, X_train, y_train, cv=k_fold)
        print(score)

        # Random Forest
        RF = RandomForestClassifier(n_estimators=100)
        score = cross_val_score(RF, X_train, y_train, cv=k_fold)
        print(score)

        # Predicting on test data
        RF.fit(X_train, y_train)
        y_predicted = RF.predict(X_test)
        joblib.dump(RF, self.args.model_path)
        match = re.search('([^\/]*)$', self.args.model_path)
        # Upload Model to Run artifacts
        self.run.upload_file(name=self.args.artifact_loc + match.group(1),
                             path_or_stream=self.args.model_path)
        self.run.complete()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get arguments')
    parser.add_argument('--container_name', type=str,
                        help='Path to default datastore container')
    parser.add_argument('--input_train_csv', type=str, help='Input CSV file')
    parser.add_argument('--input_test_csv', type=str, help='Input CSV file')
    parser.add_argument('--dataset_name', type=str,
                        help='Dataset name to store in workspace')
    parser.add_argument('--dataset_desc', type=str, help='Dataset description')
    parser.add_argument('--model_path', type=str,
                        help='Path to store the model')
    parser.add_argument('--artifact_loc', type=str,
                        help='DevOps artifact location to store the model', default='')
    args = parser.parse_args()
    titanic_classifier = TitanicClassification(args)
    titanic_classifier.create_pipeline()
