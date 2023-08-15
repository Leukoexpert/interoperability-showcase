from train import TrainConfig
from data_loader import DataLoader
from model_utils import split_data_into_data_target, Model_Loader, get_metrics_binary, get_auc
from model import Model
from plotting_utils import ploting_roc_curve


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
    data_loader = DataLoader(train_config)
    X_train, Y_train = split_data_into_data_target(data_loader.train_data, "diagnosed_leuk")
    X_test, Y_test = split_data_into_data_target(data_loader.test_data, "diagnosed_leuk")
    Model(train_config, X_train, Y_train, X_test, Y_test)
    # 3. Save column names and number of datapoints per column in results as csv
    # 4. model load
    # 5. model train
    # 6. model test
    # 7. model save in results


if __name__ == '__main__':
    main()
