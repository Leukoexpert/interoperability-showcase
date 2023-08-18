import csv
import os
import numpy as np
import pandas as pd
from typing import Type
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV
import joblib
from train import TrainConfig
from model_utils import get_auc
from model_utils import Model_Loader
from model_utils import get_metrics_binary
from utils import save_df_to_csv


class Model():

    def __init__(self, train_config: Type[TrainConfig], train_data, train_label, test_data, test_label):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.train_config = train_config
        self.method = "rf"


    def train_loop(self):
        """
        for both the specific model and the global model do sepertly

        1. load the model
        2. train the model
        3. save the model

        """

        # Train station specific mode
        station_spesific_model = self.create_model()
        station_spesific_model = self.train_model(self.train_data, self.train_label, station_spesific_model)
        self.save_model(station_spesific_model, self.train_config.get_station_model_path())

        # Train global model
        model = self.load_model()
        model = self.train_model(self.train_data, self.train_label, model)
        self.save_model(model, self.train_config.get_model_path())




    def test_loop(self):
        """
        For both the specific model and the global model do sepertly
        1. load the model from the local model path
        2. predict the test data
        3. evaluate the prediction
        4. save the evaluation
        :return:
        """
        # Test station specific model
        station_spesific_model = self.load_model()
        station_spesific_evaluation = self.evaluation(self.test_data, self.test_label, station_spesific_model, method=self.method)
        save_df_to_csv(self.train_config, station_spesific_evaluation, self.train_config.get_station_data_path(), f"{self.train_config.get_station_name()}_spesific_model_evaluation.csv")

        # Test global model
        model = self.load_model()
        evaluation = self.evaluation(self.test_data, self.test_label, model, method=self.method)
        save_df_to_csv(self.train_config, evaluation, self.train_config.get_data_path(), f"{self.train_config.get_station_name()}evaluation.csv")



    def load_model(self):
        self.model_loader = Model_Loader(self.train_config)
        self.model_load = self.model_loader.model_in_result_path()
        if self.model_load:
            model = self.model_loader.load_model()
        else:
            model = self.create_model()

        return model

    def prediction(self, model):
        print("prediction")
        prediction = model.predict_proba(self.test_data)
        prediction = prediction[:, 1]
        return prediction

    def prediction_binary(self, model):
        prediction_binary = model.predict(self.test_data)
        return prediction_binary

    def create_model(self):
        print("create model")
        model = RandomForestClassifier(max_depth=10, random_state=0, n_estimators=3)
        return model

    def train_model(self, train_data, train_label, model):
        print("train model with specific split")
        model.fit(train_data, train_label)
        return model


    def save_model(self, model, model_path):
        print("save model")
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        joblib.dump(model, os.path.join(model_path, "model.joblib"))

    def save_roc_curve(self):
        print("save roc curve")
        auc, fpr, tpr = get_auc(prediction=self.prediction, y_test=self.test_label)
        if auc is not None:
            ploting_roc_curve(fpr=fpr, tpr=tpr, train_config=self.train_config, auc=auc)

    def evaluation(self, test_data, test_label, model, method: str):
        print("evaluation on specific split")
        prediction = model.predict_proba(test_data)
        if prediction.shape[1] != 2:
            prediction = np.asarray(prediction).flatten()
        else:
            prediction = prediction[:, 1]
        eval_df = get_metrics_binary(prediction, test_label, method=method, threshold=0.2)
        return eval_df

    def evaluation(self):
       print("evaluation")
       eval_df = get_metrics_binary(self.prediction, self.test_label, method="rf", threshold=0.2)
       save_df_to_csv(self.train_config, eval_df, self.train_config.get_data_path(), "eval_df.csv")

    def evaluation_aggregation(self, eval_dfs):
        print("evaluation aggregation")
        # takes a list of eval_df dataframes and aggregates them
        n = len(eval_dfs)
        eval_df_acc = {'roc-score': 0, 'accuracy': 0, 'balanced_accuracy': 0, 'f1': 0, 'mcc': 0}
        for eval_df in eval_dfs:
            eval_df_acc['roc-score'] += eval_df['roc-score']
            eval_df_acc['accuracy'] += eval_df['accuracy']
            eval_df_acc['balanced_accuracy'] += eval_df['balanced_accuracy']
            eval_df_acc['f1'] += eval_df['f1']
            eval_df_acc['mcc'] += eval_df['mcc']

        eval_df_acc['roc-score'] = eval_df_acc['roc-score'] / n
        eval_df_acc['accuracy'] = eval_df_acc['accuracy'] / n
        eval_df_acc['balanced_accuracy'] = eval_df_acc['balanced_accuracy'] / n
        eval_df_acc['f1'] = eval_df_acc['f1'] / n
        eval_df_acc['mcc'] = eval_df_acc['mcc'] / n
        eval_df_acc = pd.DataFrame(eval_df_acc, index=[self.method])
        return eval_df_acc

    def evaluation_save(self, eval_df):
        save_df_to_csv(self.train_config, eval_df, self.train_config.get_data_path(), "eval_df.csv")

        # {'roc-score': roc_score, 'accuracy': accuracy, 'balanced_accuracy': balanced_accuracy, 'f1': f1,
        # 'mcc': mcc}
