# Interoperability-showcase
The Discovery Train is a Python-based PHT Train which has been developed to provide an overview of the available data for the Leuko-Expert project at the Leipzig, Aachen, and Tübingen PHT stations. The script generates tables that display the sex distribution, age distribution, diagnosis distribution, and number of visits while ensuring k-Anonymity of 5.

# Files
- main.py: This is the main script that orchestrates the data loading, processing, and analysis.
- functions_train.py: This script contains various functions used in the main script for data loading, processing, and analysis.

# Usage
The script is build to be used in the PHT environment.

# Dependencies
The Discovery Train project requires the following Python packages in the PHT master image:

- os
- pandas
- numpy

# Note
This project is designed to handle sensitive medical data. Please ensure that all data is handled in accordance with relevant data protection and privacy laws and regulations.