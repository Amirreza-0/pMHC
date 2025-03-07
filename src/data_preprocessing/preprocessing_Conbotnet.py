# load all data based on the .yaml file
import re
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

dataset_path = "../../database/Conbotnet"

# Load the data
mhc_seq = pd.read_csv(dataset_path + "/data/pseudosequence.2016.all.X.dat", sep="\t", header=None) # allele, pseudosequence
train = pd.read_csv(dataset_path + "/data/data.txt", sep="\t", header=None) # peptide, ic50, allele
test = pd.read_csv(dataset_path + "/data/test_ic50_2023.txt", sep="\t", header=None) # peptide, ic50, allele
binding = pd.read_csv(dataset_path + "/data/binding.txt", sep=" ", header=None) # X?, allele, pseudosequence, peptide, core?
test_binary = pd.read_csv(dataset_path + "/data/test_binary_2023.txt", sep="\t", header=None) # peptide, binding, allele


# get the unique alleles from test_binary and test
alleles_test_binary = test_binary[2].unique()
alleles_test = test[2].unique()

# compare
print("Alleles in test_binary but not in test: ", set(alleles_test_binary) - set(alleles_test))

# Merge test datasets based on the allele column
# First, ensure column names are consistent
test.columns = ["peptide", "ic50", "allele"]
test_binary.columns = ["peptide", "binding", "allele"]

# Outer join to keep all peptide-allele pairs from both datasets
test_all = pd.merge(test, test_binary, on=["allele", "peptide"], how="outer", suffixes=('', '_binary'))

# Add binding data based on the allele column
binding.columns = ["id", "allele", "mhc_pseudosequence", "peptide", "9mer"]
test_all = pd.merge(test_all, binding, on="allele", how="outer", suffixes=('', '_binding'))

# remove duplicates with the same allele and peptide
test_all.drop_duplicates(subset=["allele", "peptide"], inplace=True)

# add a new column named binding_label that is calculated based on ic50
test_all["binding_label"] = test_all["ic50"].apply(lambda x: 1 if x >= 0.428 else 0)

# if "binding" exists, replace the binding_label with the value in the "binding" column
test_all["binding_label"] = test_all["binding"].combine_first(test_all["binding_label"])

# Save the test_all DataFrame to a CSV file
# test_all.to_csv(dataset_path + "/data/test_all.csv", index=False)

# add binding_label to the train dataset based on the ic50 column
train.columns = ["peptide", "ic50", "allele"]
train["binding_label"] = train["ic50"].apply(lambda x: 1 if x >= 0.428 else 0)

# Save the train DataFrame to a CSV file
# train.to_csv(dataset_path + "/data/train.csv", index=False)



# Format and name columns in the mhc_seq DataFrame
mhc_seq.columns = ["allele", "conbot_pseudosequence"]

# Standardize allele names (similar to update_dataset function)
mask = mhc_seq["allele"].str.len() < 15
mhc_seq.loc[mask, "allele"] = (
    'HLA-' +
    mhc_seq.loc[mask, "allele"]
    .str.replace(':', '', regex=False)
)

# Prepare for merging with other datasets
mhc_seq["mhc_class"] = np.where(mhc_seq["allele"].str.contains('DP|DQ|DR|DM'), 'II', 'I')

# Add this standardized data to train and test datasets
test_all = pd.merge(test_all, mhc_seq, on="allele", how="left", suffixes=('', '_mhc_seq'))
train = pd.merge(train, mhc_seq, on="allele", how="left", suffixes=('', '_mhc_seq'))


missing_alleles = []
def get_MHC_sequences_(alleles, PMGen_harmonized_df, specie="homo"):
    """
    Get MHC sequences from PMGen_dataset dataset.
    Expects alleles is a pandas Series of allele names.
    """
    unique_alleles = alleles.unique()
    mapping = {}
    for allele in tqdm(unique_alleles, desc="Processing alleles"):
        # check if allele is split by "-", and process for each allele
        if "-" in allele and specie=="homo":
            allele_a = allele.split("-")[0]
            allele_b = allele.split("-")[1]
            seq_a = get_MHC_sequences_(pd.Series([allele_a]), PMGen_harmonized_df)
            seq_b = get_MHC_sequences_(pd.Series([allele_b]), PMGen_harmonized_df)
            mapping[allele] = "/".join([seq_a[0], seq_b[0]])
            continue
        pattern = re.compile(re.escape(allele))
        matching_rows = PMGen_harmonized_df[PMGen_harmonized_df['simple_allele'].str.contains(pattern, na=False)]
        if not matching_rows.empty:
            allele_set = set(allele)
            matching_rows = matching_rows.copy()
            matching_rows['distance'] = matching_rows['simple_allele'].apply(lambda x: len(set(x) - allele_set))

            # First try with sequences >= 75 characters
            filtered_rows = matching_rows[matching_rows['mhc_sequence'].str.len() >= 75]

            if not filtered_rows.empty:
                mapping[allele] = filtered_rows.loc[filtered_rows['distance'].idxmin(), 'mhc_sequence']
            else:
                # If no sequences with length >= 75, take the 5 longest
                longest_rows = matching_rows.nlargest(5, 'mhc_sequence').sort_values('distance')
                if not longest_rows.empty:
                    mapping[allele] = longest_rows.iloc[0]['mhc_sequence']
                else:
                    mapping[allele] = np.nan
        else:
            mapping[allele] = np.nan
    return alleles.map(mapping)


def update_dataset(dataset, PMGen_harmonized_df):
    # in the "simple_allele" column remove ":" and "*"
    PMGen_harmonized_df['simple_allele'] = PMGen_harmonized_df['simple_allele'].str.replace("[:*]", "", regex=True)
    # select homo specie
    PMGen_harmonized_df_homo = PMGen_harmonized_df[PMGen_harmonized_df['species'] == 'homo']
    # select homo, they have D in allele name
    dataset_homo = dataset[dataset["allele"].str.contains("D")].copy()  # Create a copy instead of a view

    # harmonize_dataset allele names
    # if allele_name contains _ then remove it
    dataset_homo['allele'] = dataset_homo['allele'].str.replace("_", "", regex=False)
    dataset_homo['allele'] = dataset_homo['allele'].str.replace("HLA-", "", regex=False)

    # Vectorize mhc_class determination
    dataset_homo['mhc_sequence'] = get_MHC_sequences_(dataset_homo['allele'], PMGen_harmonized_df_homo)

    # do for mice
    PMGen_harmonized_df_mice = PMGen_harmonized_df[PMGen_harmonized_df['species'] == 'mice']
    dataset_mice = dataset[dataset["allele"].str.contains("D") == False].copy()  # Create a copy instead of a view

    # Vectorize mhc_class determination
    dataset_mice['mhc_sequence'] = get_MHC_sequences_(dataset_mice['allele'], PMGen_harmonized_df_mice, specie="mice")

    # Concatenate the datasets
    dataset = pd.concat([dataset_homo, dataset_mice], ignore_index=True)

    dataset['mhc_class'] = "II"

    # print the number of missing values in the mhc_sequence column
    print("Number of missing values in mhc_sequence column: ", dataset['mhc_sequence'].isnull().sum())

    # print number of unique missing alleles
    print("Number of unique missing alleles: ", dataset[dataset['mhc_sequence'].isnull()]['allele'].nunique())

    return dataset


# get mhc sequence data from PMGen
PMGen_mapping = pd.read_csv("../../database/PMGen/data/harmonized_PMgen_mapping.csv")

# Update the train and test_all datasets with MHC sequences
train = update_dataset(train, PMGen_mapping)
test_all = update_dataset(test_all, PMGen_mapping)

# Save the updated train and test_all datasets to CSV files
train.to_csv(dataset_path + "/processed/train.csv", index=False)
test_all.to_csv(dataset_path + "/processed/test_all.csv", index=False)