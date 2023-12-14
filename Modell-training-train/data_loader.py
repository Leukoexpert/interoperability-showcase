from typing import Type
import numpy as np
from train import TrainConfig
from redcap import Project
from utils import save_df_to_csv, convert_series_to_datetimes
import pandas as pd
import os


class DataLoader:
    def __init__(self, train_config: Type[TrainConfig], split_factor: float = 0.8):
        self.train_config = train_config
        data_in_data_path = self.data_in_data_path()
        if data_in_data_path:
            self.data = self.load_redcap_data_from_file()
            self.metadata = self.load_redcap_metadata_from_file()
        else:
            self.data = self.export_to_redcap_via_pycap()
            self.metadata = self.export_metadata_via_pycap()
        self.load_shared_columns_remove_non_common_columns()
        self.split_factor = split_factor
        self.transform_df()
        self.calculate_age_at_visit()
        #self.generate_data_overview()
        self.convert_label_to_binary()
        self.missing_values_branch_logic()
        self.one_hot_encode()
        self.replace_NaN()
        self.train_data, self.test_data = self.split_data_by_redcap_id()

    def data_in_data_path(self) -> bool:
        """
        check if the data is in the data path
        @return: True if the data is in the data path
        """
        if not (not os.path.exists(self.train_config.get_file_load_data()) or not os.path.exists(
                self.train_config.get_file_load_metadata())):
            file_load = True
        else:
            file_load = False
        return file_load

    def load_redcap_data_from_file(self) -> pd.DataFrame:
        """
         a simple function for reading the redcap files
        @type file_path: path to the redcap file
        """
        df = pd.read_csv(self.train_config.get_file_load_data(), index_col=0)
        return df

    def load_redcap_metadata_from_file(self) -> pd.DataFrame:
        """
         a simple function for reading the redcap files
        @type file_path: path to the redcap file
        """
        df = pd.read_csv(self.train_config.get_file_load_metadata(), index_col=0)
        return df

    def export_to_redcap_via_pycap(self) -> pd.DataFrame:
        """
        :param api_url: URL to the REDCAP API as String
        :param api_key: API Key for the Project
        :return: pandas dataframe of all records
        """
        project = Project(self.train_config.get_redcap_address(), self.train_config.get_redcap_key())
        df = project.export_records(format_type="df", raw_or_label="raw")
        return df

    def export_metadata_via_pycap(self) -> pd.DataFrame:
        """
        :param api_url: URL to the REDCAP API as String
        :param api_key: API Key for the Project
        :return: pandas dataframe of the metadata
        """
        project = Project(self.train_config.get_redcap_address(), self.train_config.get_redcap_key())
        df = project.export_metadata(format_type="df")
        # add the converting from the examination data
        df['form_name'].replace('examination_data_use_new_sheet_for_every_visit','examination_data')
        return df

    def generate_data_overview(self) -> None:
        """
        Generate an overview of the data and save it to the results folder

        """
        shorten_df = self.data.drop(columns=["age", "diagnosed_leuk"])
        save_df_to_csv(self.train_config, df=shorten_df.apply(lambda x: x.value_counts()),result_subfolder=self.train_config.get_station_name(), name="typisation_whole_df.csv")
        shorten_df_binary = self.data.drop(columns=["age"])
        label = "diagnosed_leuk"
        shorten_df_binary.loc[shorten_df_binary[label] != 2 , label] = 0
        shorten_df_binary.loc[shorten_df_binary[label] == 2, label] = 1
        index = []
        group = []
        value = []
        question = []
        for column in shorten_df_binary.columns:
            index.extend(shorten_df_binary.groupby([label])[column].value_counts().index.get_level_values(0).values)
            group.extend(shorten_df_binary.groupby([label])[column].value_counts().index.get_level_values(1).values)
            value.extend(shorten_df_binary.groupby([label])[column].value_counts().values)
            question.extend([column] * len(shorten_df_binary.groupby([label])[column].value_counts().values))
        overview_df = pd.DataFrame(list(zip(index, question, group, value)), columns=['Diagnosis', 'Questions','Answer', 'statistics'])
        over_questions = pd.pivot_table(overview_df, values='statistics', index = ['Diagnosis', 'Answer'], columns='Questions')
        save_df_to_csv(self.train_config, over_questions, self.train_config.get_station_name(), "data_overview_questions.csv")

        shorten_df_binary = self.data.drop(columns=["age"])
        label = "diagnosed_leuk"
        index = []
        group = []
        value = []
        question = []
        for column in shorten_df_binary.columns:
            index.extend(shorten_df_binary.groupby([label])[column].value_counts().index.get_level_values(0).values)
            group.extend(shorten_df_binary.groupby([label])[column].value_counts().index.get_level_values(1).values)
            value.extend(shorten_df_binary.groupby([label])[column].value_counts().values)
            question.extend([column] * len(shorten_df_binary.groupby([label])[column].value_counts().values))
        overview_df = pd.DataFrame(list(zip(index, question, group, value)),
                                   columns=['Diagnosis', 'Questions', 'Answer', 'statistics'])
        over_questions = pd.pivot_table(overview_df, values='statistics', index=['Diagnosis', 'Answer'],
                                        columns='Questions')
        save_df_to_csv(self.train_config, over_questions, self.train_config.get_station_name(),
                       "typisation_diagnosis.csv")
        df = pd.DataFrame()
        # get the column names
        df["column_name"] = self.data.columns
        # count the number of datapoints per column
        df["number_of_datapoints"] = list(self.data.count())
        # count the number of unique values per column
        df["number_of_unique_values"] = list(self.data.nunique())
        # count the number of missing values per column
        df["number_of_missing_values"] = list(self.data.isnull().sum())
        save_df_to_csv(self.train_config, df, self.train_config.get_station_name(), "data_overview.csv")

    def load_shared_columns_remove_non_common_columns(self) -> None:
        """
        select all fields which are not numeric
        numeric field could be definded in the metadata as radio or checkboxes but remove text
        loads the shared columns csv and removes all columns that are not included from self.data

        """
        # TODO check if this is the right way to do it with just using the analyse columns

        shared_columns = pd.read_csv(self.train_config.get_anal_columns(), index_col=0)
        shared_columns = shared_columns["name"].values.tolist()
        shared_columns.append("redcap_repeat_instrument")
        shared_columns.append("redcap_repeat_instance")
        # shared_columns.append(self.data.columns[0])
        self.data = self.data[shared_columns]

    def split_data_by_redcap_id(self):
        """
        if ther is a split csv file split the data into test_data/train_data by the redcap id
        if this file dose not exits chreat a random split of the data and save it to the results folder
        for one redcap id ther chan be multiple rows in the data they are all in the same group

        """
        if os.path.exists(self.train_config.get_data_split_path()):
            split = pd.read_csv(self.train_config.get_data_split_path(), index_col=0)
        else:
            split = pd.DataFrame()
            split["redcap_id"] = list(set(self.data.index))
            split["train"] = np.random.choice([True, False], len(self.data),
                                              p=[self.split_factor, 1 - self.split_factor])
            split["test"] = np.invert(split["train"])

            save_df_to_csv(
                self.train_config,
                split,
                self.train_config.get_station_name(),
                f"{self.train_config.get_station_name()}_data_split.csv")

        train_data = self.data.loc[split["train"].values]
        test_data = self.data.loc[split["test"].values]

        return train_data, test_data

    def convert_label_to_binary(self) -> None:
        """
        convert the labels from 1-103 to binary labels from 0-1
        @rtype: None
        """
        # the label is this column
        label = "diagnosed_leuk"
        diff_labels = "diagnosis"
        # convert all values that are not ALD to 0
        self.data.loc[self.data[label] != 2, label] = 0
        self.data.loc[self.data[diff_labels] == 4, label] = 0
        # convert all values that are ALD to 1
        self.data.loc[self.data[label] == 2, label] = 1

    def calculate_age_at_visit(self) -> None:
        """
        a function for the age calculation
        """
        # dob also date of birth is the reference
        self.data["dob"] = convert_series_to_datetimes(self.data["dob"])
        # the date_neuro_symp is the timestamp of the visit
        self.data["date_neuro_symp"] = convert_series_to_datetimes(self.data['date_neuro_symp'])
        # calculate the age if the dates are there
        age = []
        for patient in self.data.index:
            if self.data['date_neuro_symp'][patient] is not None and self.data['dob'][patient] is not None:
                age.append(self.data['date_neuro_symp'][patient].year - self.data['dob'][patient].year)
            else:
                age.append(None)
        # add age to data
        self.data['age'] = age
        # loose the columns which are no longer useful
        self.data = self.data.drop(columns=['dob', 'aso', 'date_neuro_symp'])

    def transform_df(self) -> None:
        """
        transform the data to a tidy form that means get the baseline information and the first visit
        """
        # change the name of the baseline which is NAN to baseline
        self.data['redcap_repeat_instrument'] = self.data['redcap_repeat_instrument'].replace(np.nan,
                                                                                              "basic_data_consent")
        # replace baseline repeat instance
        self.data['redcap_repeat_instance'] = self.data['redcap_repeat_instance'].replace(np.nan, 0)
        # for leipzig change labels in redcap_repeat_instrument to a simple form identical with tübingen
        self.data['redcap_repeat_instrument'] = self.data['redcap_repeat_instrument'].replace(
            "examination_data_use_new_sheet_for_every_visit", "examination_data")
        # get only examination und baseline information
        self.data = self.data[(self.data["redcap_repeat_instrument"] == "basic_data_consent") | (
                self.data["redcap_repeat_instrument"] == "examination_data")]
        # get only the first visit with the baseline
        self.data = self.data[(self.data["redcap_repeat_instance"] == 1) | (self.data["redcap_repeat_instance"] == 0)]
        # split into baseline and examination
        baseline_data = self.data[self.data["redcap_repeat_instrument"] == "basic_data_consent"]
        examination_data = self.data[self.data["redcap_repeat_instrument"] == "examination_data"]
        # get only interesting columns
        analysis_meta_df = pd.read_csv(self.train_config.get_anal_columns())
        analysis_meta_df_baseline = analysis_meta_df[analysis_meta_df["instrument"] == "basic_data_consent"]
        analysis_meta_df_examination = analysis_meta_df[analysis_meta_df["instrument"] == "examination_data"]
        baseline_data = baseline_data[analysis_meta_df_baseline["name"].values.tolist()]
        examination_data = examination_data[analysis_meta_df_examination["name"].values.tolist()]
        # merge them together
        self.data = baseline_data.merge(examination_data, left_index=True, right_index=True)
        # exclude data without labels
        self.data = self.data[self.data["diagnosed_leuk"].notna()]

    def replace_NaN(self):
        """
        replace the NaN values with -1
        """
        self.data = self.data.fillna(-1)

    def one_hot_encode(self):
        # get the columns which are interesting for the analysis
        anal_meta = pd.read_csv(self.train_config.get_anal_meta())
        # get the conditions for all columns
        for anal_col_index in  range(0, len(anal_meta)):
            metadata_col = anal_meta.loc[anal_col_index]
            if metadata_col['question_type'] == 'yes-no':
                i = 0
                for answer in range(metadata_col['answer_min'],metadata_col['answer_max']):
                    self.data[metadata_col['name']] = self.data[metadata_col['name']].replace(answer, i)
                    i = i + 1
            elif metadata_col['question_type'] == 'selected':
                for answer in range(metadata_col['answer_min'],metadata_col['answer_max']):
                    new_answer_name = metadata_col['name'] + '___' + str(answer)
                    self.data[new_answer_name] = 0
                    self.data[new_answer_name].loc[(self.data[metadata_col['name']] == answer)] = 1
                self.data.drop(columns=metadata_col['name'])

    def missing_values_branch_logic(self):
        # get the columns which are interesting for the analysis
        anal_meta = pd.read_csv(self.train_config.get_anal_meta())
        # get the conditions for all columns
        for anal_col_index in range(0, len(anal_meta)):
            metadata_col = anal_meta.loc[anal_col_index]
            branch_logics = metadata_col['branch_logic']
            if branch_logics is not np.nan:
                branch_logics = branch_logics.replace(' ','')
                if 'and' in branch_logics:
                    branch_logics = branch_logics.split('and')
                    column_name_1 = branch_logics[0].split("=")[0].replace("[", "").replace("]", "")
                    column_value_1 = int(branch_logics[0].split("=")[1].replace("\'", ""))
                    column_name_2 = branch_logics[1].split("=")[0].replace("[", "").replace("]", "")
                    column_value_2 = int(branch_logics[1].split("=")[1].replace("\'", ""))
                    for row in self.data.index:
                        condition_1 = self.data.loc[row, column_name_1]
                        condition_2 = self.data.loc[row, column_name_2]
                        row_value_1 = self.data.loc[row, metadata_col['name']]
                        if condition_1 != column_value_1 and condition_2 != column_value_2 and not np.isnan(row_value_1):
                            self.data.loc[row, metadata_col['name']] = -2
                        if condition_1 == column_value_1 and condition_2 == column_value_2 and np.isnan(row_value_1):
                            self.data.loc[row, metadata_col['name']] = -2
                    continue
                branch_logics = branch_logics.split('or')
                for branch_logic in branch_logics:
                    column_name = branch_logic.split("=")[0].replace("[", "").replace("]", "")
                    column_value = int(branch_logic.split("=")[1].replace("\'", ""))
                    for row in self.data.index:
                        condition = self.data.loc[row, column_name]
                        row_value = self.data.loc[row, metadata_col['name']]
                        if condition != column_value and not np.isnan(row_value):
                            self.data.loc[row, metadata_col['name']] = -2
                        if condition == column_value and np.isnan(row_value):
                            self.data.loc[row, metadata_col['name']] = -2