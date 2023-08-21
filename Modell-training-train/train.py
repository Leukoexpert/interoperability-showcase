import os

class TrainConfig:
    def __init__(self):
        # PHT medic paths
        self.DATA_PATH = "/opt/train_data"
        self.RESULT_PATH = "/opt/pht_results"
        self.SHARED_COLUMNS = "/opt/pht_train/Modell-training-train/train_configuration_files/shared_columns.csv"
        self.ANAL_COLUMNS = "/opt/pht_train/Modell-training-train/train_configuration_files/analysis_columns.csv"
        self.ANAL_META = "/opt/pht_train/Modell-training-train/train_configuration_files/meta_data_analysis.csv"



        # PHT medic environment variable
        self.PHT_MEDIC = self.is_pht_medic()

        self.redcap_address, self.redcap_key, self.station_name, self.file_load_data, self.file_load_metadata = self.data_access()
        self.DATA_SPLIT_PATH = f"/opt/pht_train/Modell-training-train/train_configuration_files/{self.station_name}_data_split.csv"
        self.MODEL_PATH = "/opt/pht_results/model"
        self.MODEL_TESTING_PATH = "/opt/pht_train/models_for_testing"
        self.STATION_MODEL_PATH = f"/opt/pht_results/model_{self.station_name}"
        self.IMAGE_PATH = "/opt/pht_results/image"
        self.create_path(self.get_image_path())
        self.create_path(self.MODEL_PATH)


    def is_pht_medic(self):
        if os.environ.get('REDCAP_ADDRESS') is None:
            pht_medic = True
        else:
            pht_medic = False

        return pht_medic

    def data_access(self):

        if not self.PHT_MEDIC:
            # read in the environment variables from the docker container
            redcap_address = str(os.environ['REDCAP_ADDRESS'])
            redcap_key = str(os.environ['REDCAP_KEY'])
            station_name = str(os.environ['STATION_NAME'])
            file_load_data = str(os.environ['FILELOADDATA'])
            file_load_metadata = str(os.environ['FILELOADMETADATA'])
        else:
            # show files in the directory data path
            print("Data path: " + self.DATA_PATH)
            #print(os.listdir(self.DATA_PATH))
            station_name = "Tuebingen"
            file_load_data = self.DATA_PATH + '/Leukoexpert.csv'
            file_load_metadata = self.DATA_PATH + '/meta.csv'
            redcap_address = None
            redcap_key = None
        return redcap_address, redcap_key, station_name, file_load_data, file_load_metadata

    # getter methods
    def get_redcap_address(self):
        return self.redcap_address

    def get_redcap_key(self):
        return self.redcap_key

    def get_station_name(self):
        return self.station_name

    def get_file_load_data(self):
        return self.file_load_data

    def get_file_load_metadata(self):
        return self.file_load_metadata

    def get_data_path(self):
        return self.DATA_PATH

    def get_result_path(self):
        return self.RESULT_PATH

    def get_pht_medic(self):
        return self.PHT_MEDIC

    def get_shared_columns(self):
        return self.SHARED_COLUMNS

    def get_data_split_path(self):
        return self.DATA_SPLIT_PATH

    def get_model_path(self):
        return self.MODEL_PATH

    def get_station_model_path(self):
        return self.STATION_MODEL_PATH

    def get_anal_columns(self):
        return self.ANAL_COLUMNS

    def get_anal_meta(self):
        return self.ANAL_META

    def get_image_path(self):
        return self.IMAGE_PATH

    def get_model_testing_path(self):
        return self.MODEL_TESTING_PATH

    def create_path(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
