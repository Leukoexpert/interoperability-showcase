import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
# LEUKO Overview train RedCAP
import numpy as np
import pandas as pd

from functions_train import export_to_redcap_via_pycap, export_metadata_via_pycap, load_redcap_data_from_file, \
    load_redcap_metadata_from_file, get_instrument_df, get_record_id_to_instruments, get_transformation_dict_from_answers, \
    convert_series_to_datetimes, save_hist_plot, k_anonym_count, transformation_dict, transformation_dict_raw_label, \
    transformation_dict_label_short

# PHT medic paths
DATA_PATH = "/opt/train_data"
RESULT_PATH = "/opt/pht_results"

# set plot font size

if os.environ.get('REDCAP_ADDRESS') is None:
    PHT_MEDIC = True
else:
    PHT_MEDIC = False

if not PHT_MEDIC:
    # read in the environment variables from the docker container
    redcap_address = str(os.environ['REDCAP_ADDRESS'])
    redcap_key = str(os.environ['REDCAP_KEY'])
    station_name = str(os.environ['STATION_NAME'])
    file_load_data = str(os.environ['FILELOADDATA'])
    file_load_metadata = str(os.environ['FILELOADMETADATA'])
else:
    # show files in the directory data path
    print("Data path: " + DATA_PATH)
    print(os.listdir(DATA_PATH))
    station_name = "Tuebingen"
    file_load_data = DATA_PATH + '/Leukoexpert.csv'
    file_load_metadata = DATA_PATH + '/meta.csv'

# print the environment variable for the viewing
print("Station name: " + station_name)

# create folder for images
image_path = RESULT_PATH + '/image' + station_name

if not os.path.exists(image_path):
    os.makedirs(image_path)

# in this redcap there are more instruments
# in baseline included is the sociology data and genetics
# Examination data is the phenotype of the patient
# MRI is the image data

print("------------------------------------------------")
print("Start Report generation...")

if os.path.exists(file_load_data) and os.path.exists(file_load_metadata):
    file_load = True
else:
    file_load = False

if file_load:
    print("Continue with File loader")
    data = load_redcap_data_from_file(file_load_data)
    metadata = load_redcap_metadata_from_file(file_load_metadata)
else:
    print("Continue without File loader")
    # read the data from redcap system from
    data = export_to_redcap_via_pycap(api_url=redcap_address, api_key=redcap_key)
    # export the metadata from the redcap system
    metadata = export_metadata_via_pycap(api_url=redcap_address, api_key=redcap_key)

print("Data loading complete")
print("-----------------------------------------------------------")

print("Start Data processing...")

data = data.reset_index()

data['redcap_repeat_instrument'] = ['examination_data' if i == 'examination_data_use_new_sheet_for_every_visit' else i for i in data['redcap_repeat_instrument']]

data['redcap_repeat_instrument'] = data['redcap_repeat_instrument'].replace(np.nan, "basic_data_consent")

print("Start with the Baseline section")

print("Extract baseline information")
# Baseline section

baseline_df = get_instrument_df(redcap_data=data, redcap_metadata=metadata, instrument="basic_data_consent",
                                with_complete=True)

id_baseline = get_record_id_to_instruments(redcap_data=data, instrument="basic_data_consent")

baseline_df["record_id"] = list(id_baseline)

number_of_records = len(baseline_df)

sex_series = baseline_df['sex']

sex_series = sex_series.dropna()

table_sex = sex_series.value_counts()

table_sex.to_csv(os.path.join(image_path, 'sex_data.csv'), index=False)

sex_dict = get_transformation_dict_from_answers(metadata, 'sex')

table_sex.index = table_sex.index.astype(int)
table_sex.index = [sex_dict[x] for x in table_sex.index]

diagnosis_series = baseline_df["diagnosed_leuk"].dropna()


if station_name == 'Tuebingen':
    diagnosis_indizis = [round(x) - 1 for x in list(diagnosis_series.value_counts().index)]
    diagnosis_counts = list(diagnosis_series.value_counts())
    tabele_diagnosis = []
    tabele_diagnosis_count = []
    for i in zip(diagnosis_indizis, diagnosis_counts):
        tabele_diagnosis.append(list(transformation_dict)[i[0]])
        tabele_diagnosis_count.append(i[1])
    number_diagnosis = sum(tabele_diagnosis_count)
    tabele_diagnosis, tabele_diagnosis_count = k_anonym_count(tabele_diagnosis, tabele_diagnosis_count, 5)
    print(tabele_diagnosis)
    print(tabele_diagnosis_count)
    # combine table_diagnosis and table_diagnosis_index
    tabele_diagnosis = pd.DataFrame(list(zip(tabele_diagnosis, tabele_diagnosis_count)),
                                    columns=['diagnosis', 'count']).squeeze()
else:
    table_diagnosis = diagnosis_series.value_counts()
    table_diagnosis.index = table_diagnosis.index.map(transformation_dict_raw_label).map(
        transformation_dict_label_short)
    number_diagnosis = sum(table_diagnosis)
    tabele_diagnosis, tabele_diagnosis_count = k_anonym_count(list(table_diagnosis.index), list(table_diagnosis), 5)
    # combine table_diagnosis and table_diagnosis_index
    tabele_diagnosis = pd.DataFrame(list(zip(tabele_diagnosis, tabele_diagnosis_count)),
                                    columns=['diagnosis', 'count']).squeeze()

dod_df = convert_series_to_datetimes('fcl', baseline_df)
dob_df = convert_series_to_datetimes('dob', baseline_df)
print(dob_df)

# merge the two lists
records_first_exam = []
date_diagnosis = []

for i, g in dod_df.groupby(by='record_id'):
    records_first_exam.append(g.iloc[0]['record_id'])
    date_diagnosis.append(g.iloc[0]['fcl'])

doe_first_df = pd.DataFrame(list(zip(records_first_exam, date_diagnosis)),
                            columns=['record_id', 'date_diagnosis'])

final_df = pd.merge(dob_df, doe_first_df, how='inner', on='record_id')

final_df = final_df.dropna()

age = [final_df['date_diagnosis'][patient].year - final_df['dob'][patient].year for patient in
       final_df.index]

age = [num for num in age if 0 <= num <= 100]
age_file_path = os.path.join(image_path, 'age_diagnosis_hist_plot.png')

save_hist_plot(output_path=age_file_path, parameter=age, label_x="age at diagnosis", n=len(sex_series))

# genetic data extraction


print("Start with the Genetics section")

print("Extract genetic information")

genetic_df = get_instrument_df(redcap_data=data, redcap_metadata=metadata, instrument="genetics", with_complete=True)

id_genetics = get_record_id_to_instruments(redcap_data=data, instrument="genetics")

if not genetic_df.empty:

    genetic_df["record_id"] = list(id_genetics)

    gen_col = genetic_df['affected_gene']

    genetic_df = genetic_df[gen_col.notna()]
    number_of_genetic_masks = len(genetic_df)

    gen_transformation_dict = {'ABCD1': 'ABCD1', 'ARSA-Gen': 'ARSA',
                               'Notch3': 'NOTCH3', 'ABCD1-Gen': 'ABCD1', 'ARSA': 'ARSA', 'NOTCH3': 'NOTCH3',
                               'GFAP': 'GFAP', 'GALC': 'GALC', 'GBE1': 'GBE1', 'LMNB1': 'LMNB1', 'POLR3B': 'POLR3B',
                               'CST3': 'CST3', 'SPG11': 'SPG11', 'HTRA1': 'HTRA1', 'COL4A1': 'COL4A1', 'GLA': 'GLA',
                               'POL3B': 'POL3B', 'Cyp27A1': 'Cyp27A1', 'PLP1': 'PLP1', 'EIF2B5': 'EIF2B5',
                               'PHYH': 'PHYH',
                               'EPRS1': 'EPRS1', 'EIF2B1': 'EIF2B1', 'AARS2': 'AARS2', 'ARSA ': 'ARSA',
                               'eiF2B4': 'eiF2B4',
                               'POLR3B-Gen': 'POLR3B', 'CYP27A1': 'CYP27A1'}

    affected_genes = genetic_df['affected_gene']
    affected_genes = affected_genes[affected_genes.notna()]
    affected_genes = affected_genes.map(gen_transformation_dict)

    # plot of the different affected genes
    table_genes = affected_genes.value_counts()

    table_genes_list = list(table_genes)
    table_genes_index = list(table_genes.index)
    table_genes_index, table_genes_list = k_anonym_count(table_genes_index, table_genes_list, 5)

    genetic_file_path = os.path.join(image_path, "genetic_data_plot.png")

    # save table_genes_list and table_genes_index to a csv file
    df = pd.DataFrame({'Affected genes': table_genes_index, 'Number of patients': table_genes_list})
    df.to_csv(os.path.join(image_path, 'genetic_data.csv'), index=False)


else:
    print("No genes found")
    table_genes = []
    genetic_file_path = os.path.join(image_path, "genetic_data_plot.png")


# examination chapter
print("Start with the Examination section")

print("Extract examination information")

examination_df = get_instrument_df(redcap_data=data, redcap_metadata=metadata,
                                   instrument="examination_data",
                                   with_complete=True)

examination_id = get_record_id_to_instruments(redcap_data=data, instrument="examination_data")

if not examination_df.empty:

    examination_df["record_id"] = list(examination_id)

    number_of_examinations = len(examination_df)

    examination_col = examination_df['cog']

    examination_df = examination_df.loc[examination_col.notna(), :]

    table_examination_times = examination_df['record_id'].value_counts()

    table_visits_exam = examination_df['record_id'].value_counts()

    table_visits_count_exam = pd.Series(table_visits_exam).value_counts()

    patients_list = list(table_visits_count_exam)
    number_of_visits = list(table_visits_count_exam.index)

    exam_file_path = os.path.join(image_path, "exam_data_plot.png")
    # save number_of_visits and patients_list to a csv file
    df = pd.DataFrame({'Number of visits': number_of_visits, 'Number of patients': patients_list})
    df.to_csv(os.path.join(image_path, 'exam_data.csv'), index=False)

else:
    print("no examination found")


# Overview plot

baseline_index = ['yes'] * len(baseline_df['record_id'])
if not examination_df.empty:
    examination_index = list(examination_df['record_id'])
else:
    examination_index = []
if not genetic_df.empty:
    genetic_index = list(genetic_df['record_id'])
else:
    genetic_index = []

baseline_record = list(baseline_df['record_id'])

index_df = pd.DataFrame(list(baseline_index), columns=["index_baseline"])
index_df['has_exam'] = ['no'] * len(baseline_index)
index_df['has_mri'] = ['no'] * len(baseline_index)
index_df['has_genetics'] = ['no'] * len(baseline_index)

for i in range(0, len(baseline_index)):
    item = baseline_record[i]
    if item in examination_index:
        index_df.loc[i, 'has_exam'] = 'yes'
    if item in genetic_index:
        index_df.loc[i, 'has_genetics'] = 'yes'


table_masks = index_df.groupby(['has_exam',  'has_genetics']).size()

table_masks_reset = table_masks.reset_index()

number_accumulated_baseline = sum(table_masks)

number_accumulated_examination = sum(table_masks[list(table_masks_reset.index[table_masks_reset["has_exam"] == "yes"])])


number_accumulated_genetics = sum(
    table_masks[list(table_masks_reset.index[table_masks_reset["has_genetics"] == "yes"])])

table_generel = [number_accumulated_baseline, number_accumulated_examination,
                 number_accumulated_genetics]
table_generel_names = ['Baseline', 'Examination', 'Genetics']
# save tabel_generel to csv
table_generel_df = pd.DataFrame(table_generel, index=table_generel_names, columns=["Number of Patients"])
table_generel_df.to_csv(os.path.join(image_path, "overview_accumulated.csv"))


print("Data processing complete")

