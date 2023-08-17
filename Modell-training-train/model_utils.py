import os
from typing import Type
import joblib
import pandas as pd
from train import TrainConfig
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score, \
    matthews_corrcoef


class Model_Loader:

    def __init__(self, train_config: Type[TrainConfig]):
        self.train_config = train_config

    def model_in_result_path(self) -> bool:
        """
              check if the model is in the model path
              @return: True if the model is in the model path
              """
        if os.path.exists(os.path.join(self.train_config.get_model_path(), "model.joblib")):
            model_load = True
        else:
            model_load = False
        return model_load

    def load_model(self):
        """
        load the model from the model path
        only works for random forest, needs a generaliset metheod for all models
        # TODO: generalise the model loading
        # TODO: adding n estimators to the model shoudent be done here
        @return model
        """
        if self.model_in_result_path():
            model = joblib.load(os.path.join(self.train_config.get_model_path(), "model.joblib"))
            model.set_params(n_estimators= 3+model.n_estimators, warm_start=True)
        else:
            model = None
        return model



def split_data_into_data_target(data: pd.DataFrame, coloumn_name_label: str):
    X = data.loc[:, data.columns != coloumn_name_label]
    Y = data[coloumn_name_label]
    return X, Y


def get_auc(y_test: list, prediction: list):
    """
    calculate the auc for a prediction and the actual labels
    @param y_test: the actual labels
    @param prediction: a prediction with 0,1 as classes
    @return: the model auc, fpr and tpr
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_test, prediction)
        model_auc = auc(fpr, tpr)
        return model_auc, fpr, tpr
    except ValueError as e:
        print(f"{e} error with roc_curve")
        print(f"y_test: {y_test}")
        print(f"prediction: {prediction}")

    return None, None, None


def get_metrics_binary(prediction_test:list, y_test: list, method: str, threshold:float) -> pd.DataFrame:
    """
    generate the metrics accuracy, balanced accuracy, f1 score, mcc and auc for a specific threshold
    @param threshold: a selected threshold, it is suggested that the threshold should be equal to the distribution of
    the labels
    @param method: name of the model
    @param y_test: the actual labels related to the data of the input of themodel
    @param prediction_test: a list of probabilities from a prediction of a model
    """
    # choose the optimal threshold
    optimized_threshold = threshold
    # try the roc score calculation
    try:
        roc_score = roc_auc_score(y_score=prediction_test, y_true=y_test)
    except ValueError as e:
        print(f"{e} error with roc_auc_score")
        roc_score = None
    # calculate the binary labels out of the probabilities
    prediction_test = [1 if i >= optimized_threshold else 0 for i in prediction_test]

    # get the different metrics
    accuracy = accuracy_score(y_pred=prediction_test, y_true=y_test)
    balanced_accuracy = balanced_accuracy_score(y_pred=prediction_test, y_true=y_test)
    f1 = f1_score(y_pred=prediction_test, y_true=y_test)
    mcc = matthews_corrcoef(y_pred=prediction_test, y_true=y_test)
    # combine all metrics in a dic and then df
    metric_dic = {'roc-score': roc_score, 'accuracy': accuracy, 'balanced_accuracy': balanced_accuracy, 'f1': f1,
                  'mcc': mcc}
    metric_df = pd.DataFrame(metric_dic, index=[method])
    return metric_df
