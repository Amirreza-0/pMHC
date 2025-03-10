# load all data based on the .yaml file
import os
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
# drop the id column
binding.drop(columns="id", inplace=True)
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


def get_simple_allele(allele):
    # check length of allele
    if len(allele) != 8:
        return
    # Split the allele into gene part and number part
    gene = allele[:4]  # First 4 characters (e.g., DRB3 or DQA1)
    numbers = allele[4:]  # Remaining numbers (e.g., 0101 or 0501)

    # Split the numbers into two parts (2 digits each)
    part1 = numbers[:2]  # First two digits
    part2 = numbers[2:]  # Last two digits

    # Construct the new format with * and : separators
    return f"{gene}*{part1}:{part2}"


missing_alleles = []
def get_MHC_sequences_(alleles, PMGen_harmonized_df, specie="homo"):
    unique_alleles = alleles.unique()
    mapping_seq = {}
    mapping_pseudo = {}
    mapping_id = {}
    for allele in tqdm(unique_alleles, desc="Processing alleles"):
        mapping_alelle = allele
        if "-" in allele and specie == "homo":
            allele_a = get_simple_allele(allele.split("-")[0])
            allele_b = get_simple_allele(allele.split("-")[1])

            seq_a, pseudo_a, id_a = get_MHC_sequences_(pd.Series([allele_a]), PMGen_harmonized_df)
            seq_b, pseudo_b, id_b = get_MHC_sequences_(pd.Series([allele_b]), PMGen_harmonized_df)
            mapping_seq[mapping_alelle] = "/".join([seq_a.iloc[0], seq_b.iloc[0]])
            mapping_pseudo[mapping_alelle] = "/".join([pseudo_a.iloc[0], pseudo_b.iloc[0]])
            mapping_id[mapping_alelle] = ";;".join([id_a.iloc[0], id_b.iloc[0]])
            continue

        if len(allele) == 8:
            allele = get_simple_allele(allele)

        exact_matches = PMGen_harmonized_df[PMGen_harmonized_df['simple_allele'] == allele]
        if not exact_matches.empty:
            if len(exact_matches) > 1:
                idx = exact_matches['sequence'].str.len().idxmax()
                mapping_seq[mapping_alelle] = exact_matches.loc[idx, 'sequence']
                mapping_pseudo[mapping_alelle] = exact_matches.loc[idx, 'pseudo_sequence']
                mapping_id[mapping_alelle] = exact_matches.iloc[idx, 'allele']
            else:
                mapping_seq[mapping_alelle] = exact_matches.iloc[0]['sequence']
                mapping_pseudo[mapping_alelle] = exact_matches.iloc[0]['pseudo_sequence']
                mapping_id[mapping_alelle] = exact_matches.iloc[0]['allele']

            continue

        pattern = re.compile(re.escape(allele))
        matching_rows = PMGen_harmonized_df[PMGen_harmonized_df['simple_allele'].str.contains(pattern, na=False)]
        if not matching_rows.empty:
            allele_set = set(allele)
            matching_rows = matching_rows.copy()
            matching_rows['distance'] = matching_rows['simple_allele'].apply(lambda x: len(set(x) - allele_set))

            filtered_rows = matching_rows[matching_rows['sequence'].str.len() >= 75]

            if not filtered_rows.empty:
                idx = filtered_rows['distance'].idxmin()
                mapping_seq[mapping_alelle] = filtered_rows.loc[idx, 'sequence']
                mapping_pseudo[mapping_alelle] = filtered_rows.loc[idx, 'pseudo_sequence']
                mapping_id[mapping_alelle] = filtered_rows.loc[idx, 'allele']
            else:
                matching_rows['mhc_sequence_length'] = matching_rows['sequence'].str.len()
                longest_rows = matching_rows.sort_values(by='mhc_sequence_length', ascending=False).head(5).sort_values('distance')
                if not longest_rows.empty:
                    mapping_seq[mapping_alelle] = longest_rows.iloc[0]['sequence']
                    mapping_pseudo[mapping_alelle] = longest_rows.iloc[0]['pseudo_sequence']
                    mapping_id[mapping_alelle] = longest_rows.iloc[0]['allele']
                else:
                    mapping_seq[mapping_alelle] = np.nan
                    mapping_pseudo[mapping_alelle] = np.nan
        else:
            mapping_seq[mapping_alelle] = np.nan
            mapping_pseudo[mapping_alelle] = np.nan

    return alleles.map(mapping_seq), alleles.map(mapping_pseudo), alleles.map(mapping_id)


def update_dataset(dataset, PMGen_pseudoseq):
    # clean the dataset['allele'] column, remove : and *
    dataset['allele'] = dataset['allele'].str.replace(":", "", regex=False)
    dataset['allele'] = dataset['allele'].str.replace("*", "", regex=False)

    # select homo specie
    PMGen_harmonized_df_homo = PMGen_pseudoseq[PMGen_pseudoseq['species'] == 'homo']
    # select homo, they have D in allele name
    dataset_homo = dataset[dataset["allele"].str.contains("D")].copy()
    dataset_mice = dataset[~dataset["allele"].str.contains("D")].copy()  # Create a copy instead of a view

    # harmonize_dataset allele names
    # if allele_name contains _ then remove it
    dataset_homo['simple_allele'] = dataset_homo['allele'].str.replace("_", "", regex=False)
    dataset_homo['simple_allele'] = dataset_homo['simple_allele'].str.replace("HLA-", "", regex=False)

    # Vectorize mhc_class determination
    dataset_homo['mhc_sequence'], dataset_homo['PMGen_pseudo_sequence'], dataset_homo['PMGen_id'] = get_MHC_sequences_(dataset_homo['simple_allele'], PMGen_harmonized_df_homo)

    # do for mice
    PMGen_harmonized_df_mice = PMGen_pseudoseq[PMGen_pseudoseq['species'] == 'mice']

    # Vectorize mhc_class determination
    dataset_mice['mhc_sequence'], dataset_mice['PMGen_pseudo_sequence'],  dataset_mice['PMGen_id']= get_MHC_sequences_(dataset_mice['allele'], PMGen_harmonized_df_mice, specie="mice")

    # Concatenate the datasets
    dataset = pd.concat([dataset_homo, dataset_mice], ignore_index=True)

    dataset['mhc_class'] = "II"

    # print the number of missing values in the mhc_sequence column
    print("Number of missing values in mhc_sequence column: ", dataset['mhc_sequence'].isnull().sum())

    # print number of unique missing alleles
    print("Number of unique missing alleles: ", dataset[dataset['mhc_sequence'].isnull()]['allele'].nunique())

    print("Simple alleles with missing sequences: ", dataset[dataset['mhc_sequence'].isnull()]['allele'].unique())

    dataset.rename(columns={"peptide": "long_mer"}, inplace=True)
    # remove duplicates with the same mhc sequence and 9mer
    dataset = dataset.drop_duplicates(subset=['mhc_sequence', 'long_mer'])

    return dataset


# get mhc sequence data from PMGen
PMGen_mapping = pd.read_csv("../../database/PMGen/data/PMGen_pseudoseq.csv")

# Update the train and test_all datasets with MHC sequences
train = update_dataset(train, PMGen_mapping)
test_all = update_dataset(test_all, PMGen_mapping)

# Save the updated train and test_all datasets to CSV files
if not os.path.exists('../../database/PMGen_data'):
    os.makedirs('../../database/PMGen_data')
train.to_csv(dataset_path + "/processed/train.csv", index=False)
test_all.to_csv(dataset_path + "/processed/test_all.csv", index=False)

# Analysis of train and test_all datasets
analysis_results = []
analysis_results.append("Train dataset analysis:")
analysis_results.append(f"Total peptides: {len(train)}")
analysis_results.append(f"Unique alleles: {train['allele'].nunique()}")
analysis_results.append(f"Total allele entries: {len(train['allele'])}")
analysis_results.append("Peptide count per allele:")
analysis_results.append(train['allele'].value_counts().to_string())
analysis_results.append("")
analysis_results.append("Test_all dataset analysis:")
analysis_results.append(f"Total peptides: {len(test_all)}")
analysis_results.append(f"Unique alleles: {test_all['allele'].nunique()}")
analysis_results.append(f"Total allele entries: {len(test_all['allele'])}")
analysis_results.append("Peptide count per allele:")
analysis_results.append(test_all['allele'].value_counts().to_string())

unique_train_only = set(train['allele'].dropna().unique()) - set(test_all['allele'].dropna().unique())
unique_test_only = set(test_all['allele'].dropna().unique()) - set(train['allele'].dropna().unique())
analysis_results.append("")
analysis_results.append(f"Alleles only in Train: {len(unique_train_only)}")
analysis_results.append(f"Alleles only in Test_all: {len(unique_test_only)}")

analysis_text = "\n".join(analysis_results)

analysis_output_path = "../../database/Conbotnet/processed/dataset_analysis.txt"
with open(analysis_output_path, "w") as f:
    f.write(analysis_text)

print(f"Analysis saved to: {analysis_output_path}")