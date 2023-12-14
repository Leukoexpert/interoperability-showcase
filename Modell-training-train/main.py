import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from train import TrainConfig
from data_loader import DataLoader
from model_utils import split_data_into_data_target, ModelLoader, get_metrics_binary, get_auc
from model import Model


def main():
    """
    Main function for training framework.
    1. Load config for train
    2. Load data from redcap / container
    3. Save column names and number of datapoints per column in results as csv
    4. model load
    5. model train
    6. model test
    7. model save in results
    """

    # 1. Load config for train
    train_config = TrainConfig()

    # 2. Load data from redcap / conteiner
    if train_config.get_station_name() == "Aachen":
        split_factor = 0
    else:
        split_factor = 0.8
    data_loader = DataLoader(train_config, split_factor)
    x_train, y_train = split_data_into_data_target(data_loader.train_data, "diagnosed_leuk")
    x_test, y_test = split_data_into_data_target(data_loader.test_data, "diagnosed_leuk")
    model = Model(train_config, x_train, y_train, x_test, y_test)
    #model.train_loop()
    model.test_loop()

if __name__ == '__main__':
    main()
