import os
import pandas as pd
import numpy as np
import re

from scipy.ndimage import label
from tqdm.auto import tqdm

# Define a column mapping dictionary to handle different input formats
COLUMN_MAPPING = {
    # Each key represents the target standardized column name
    # The value is a list of possible source column names
    'allele': [
        'allele', 'Allele', 'MHC', 'MHC_class_I', 'MHC_class_II', 'mhc',
        'MHC_I', 'MHC_II', 'MHC_Class_I', 'MHC_Class_II'
    ],
    '9mer': ['9mer', 'core', 'peptide_9mer'],
    'peptide_length': ['peptide_length', 'Peptide_Length', 'length'],
    'long_mer': [
        'long_mer', 'peptide', 'longmer', 'Long_Mer', 'Peptide_Longer',
        'Peptide', 'icore', 'core_sequence'
    ],
    'convnext_value': ['value', 'ConvNeXT_value', 'convnext_value', 'prediction', 'score'],
    'log_value': ['true_value', 'log50k'],
    'binder_state': ['NB', 'label'],

}

def map_columns(df):
    """
    Map input dataframe columns to standardized column names.
    Only specified columns are renamed. All other columns are retained as-is.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with potentially different column names

    Returns:
    --------
    pd.DataFrame
        Dataframe with standardized column names and all other columns retained
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Initialize a dictionary to store found column mappings
    rename_dict = {}

    # Iterate over the COLUMN_MAPPING to identify and rename columns
    for target_col, possible_sources in COLUMN_MAPPING.items():
        for source in possible_sources:
            if source in df.columns:
                if target_col not in rename_dict.values():
                    rename_dict[source] = target_col
                else:
                    # If the target column is already mapped, skip or handle duplicates
                    print(f"Warning: Multiple sources found for '{target_col}'. Using the first occurrence and skipping '{source}'.")
                break  # Stop searching once the first match is found

    # Perform the renaming
    df.rename(columns=rename_dict, inplace=True)

    # After renaming, add any missing required columns with NaN values
    required_columns = ['allele', 'peptide_length', 'long_mer', '9mer', 'log_value', 'binder_state']
    for col in required_columns:
        if col not in df.columns:
            print(f"Warning: No mapping found for '{col}'. Adding with NaN values.")
            df[col] = np.nan

    # Ensure no duplicate column names
    if df.columns.duplicated().any():
        duplicated_cols = df.columns[df.columns.duplicated()].tolist()
        print(f"Duplicate columns detected after renaming: {duplicated_cols}")
        # Drop duplicate columns, keeping the first occurrence
        df = df.loc[:, ~df.columns.duplicated()]

    # add a new column for binding score in binary form
    df['binding_label'] = np.where(df['log_value'] >= 0.428, 1, 0)

    return df

def process_and_combine_datasets(folders):
    """
    Process CSV files from multiple folders, map columns, retain other columns,
    combine them, and remove duplicates based on 'long_mer' and 'allele'.

    Parameters:
    -----------
    folders : dict
        Dictionary with folder names as keys and folder paths as values

    Returns:
    --------
    dict
        Dictionary of combined and de-duplicated datasets for each folder
    """
    # Initialize a dictionary to store dataframes for each folder
    datasets = {}

    # Process each folder
    for folder_name, folder_path in folders.items():
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found - {folder_path}")
            continue

        # Get all CSV files in the directory
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        print(f"\nProcessing '{folder_name}' folder:")
        print(f"Found {len(csv_files)} CSV file(s)")

        # Initialize a list to store dataframes for this folder
        folder_dataframes = []

        # Process each CSV file in the folder
        for file in csv_files:
            file_path = os.path.join(folder_path, file)
            try:
                # Read the CSV file
                data = pd.read_csv(file_path)

                # Display initial info about the file
                print(f"\nFile: {file}")
                print("Original Columns:", data.columns.tolist())

                # Map columns to standardized names
                mapped_data = map_columns(data)

                # Display mapped columns and a sample of the data
                print("Mapped Columns:", mapped_data.columns.tolist())
                print(mapped_data.head())

                # Append the dataframe to the list
                folder_dataframes.append(mapped_data)

            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

        # Combine all dataframes in this folder
        if folder_dataframes:
            try:
                # Reset index for each dataframe before concatenation
                combined_df = pd.concat(
                    [df.reset_index(drop=True) for df in folder_dataframes],
                    ignore_index=True,
                    sort=False  # To align columns in the order they appear
                )

                # Remove duplicates based on 'long_mer' and 'allele'
                if 'long_mer' in combined_df.columns and 'allele' in combined_df.columns:
                    combined_df_before = len(combined_df)
                    combined_df = combined_df.drop_duplicates(subset=['long_mer', 'allele'])
                    combined_df_after = len(combined_df)
                    duplicates_removed = combined_df_before - combined_df_after
                    print(f"Dropped {duplicates_removed} duplicate rows based on 'long_mer' and 'allele'.")
                else:
                    print("Columns 'long_mer' and/or 'allele' are missing. Skipping duplicate removal based on these columns.")

                # Store the combined and de-duplicated dataframe
                datasets[folder_name] = combined_df

                print(f"Combined '{folder_name}' dataset shape: {combined_df.shape}")

            except pd.errors.InvalidIndexError as ie:
                print(f"InvalidIndexError during concatenation in folder '{folder_name}': {ie}")
                # Additional debugging info
                for idx, df in enumerate(folder_dataframes):
                    print(f"\nDataFrame {idx} columns: {df.columns.tolist()}")
                raise  # Re-raise the exception after logging

    return datasets


def get_MHC_sequences_(alleles, PMGen_harmonized_df):
    """
    Get MHC sequences from PMGen_dataset dataset.
    Expects alleles is a pandas Series of allele names.
    """
    unique_alleles = alleles.unique()
    mapping = {}
    for allele in tqdm(unique_alleles, desc="Processing alleles"):
        pattern = re.compile(re.escape(allele))
        matching_rows = PMGen_harmonized_df[PMGen_harmonized_df['simple_allele'].str.contains(pattern, na=False)]
        if not matching_rows.empty:
            allele_set = set(allele)
            matching_rows = matching_rows.copy()
            matching_rows['distance'] = matching_rows['simple_allele'].apply(lambda x: len(set(x) - allele_set))
            mapping[allele] = matching_rows.loc[matching_rows['distance'].idxmin(), 'mhc_sequence']
        else:
            mapping[allele] = None
    return alleles.map(mapping)



def update_dataset(dataset, PMGen_harmonized_df):
    # Vectorize simple_allele transformation using pandas string methods
    mask = PMGen_harmonized_df['simple_allele'].str.len() < 15
    PMGen_harmonized_df.loc[mask, 'simple_allele'] = (
        'HLA-' +
        PMGen_harmonized_df.loc[mask, 'simple_allele']
        .str.replace(r'\*', '', regex=True)
        .str.replace(':', '', regex=False)
    )
    # Vectorize mhc_class determination
    dataset['mhc_sequence'] = get_MHC_sequences_(dataset['allele'], PMGen_harmonized_df)
    dataset['mhc_class'] = np.where(dataset['allele'].str.contains('DP|DQ|DR|DM'), 'II', 'I')
    return dataset


def get_MHC_pseudosequences_(alleles, NetMHCpan_df):
    """
    Get MHC sequences from PMGen_dataset dataset.
    Expects alleles is a pandas Series of allele names.
    """
    unique_alleles = alleles.unique()
    mapping = {}
    for allele in tqdm(unique_alleles, desc="Processing alleles"):
        pattern = re.compile(re.escape(allele))
        matching_rows = NetMHCpan_df[NetMHCpan_df[0].str.contains(pattern, na=False)]
        if not matching_rows.empty:
            allele_set = set(allele)
            matching_rows = matching_rows.copy()
            matching_rows['distance'] = matching_rows[0].apply(lambda x: len(set(x) - allele_set))
            mapping[allele] = matching_rows.loc[matching_rows['distance'].idxmin(), 1]
        else:
            mapping[allele] = None
    return alleles.map(mapping)



def update_dataset2(dataset, NetMHCpan_df):
    # Vectorize simple_allele transformation using pandas string methods
    mask = NetMHCpan_df[0].str.len() < 15
    NetMHCpan_df.loc[mask, 0] = (
        'HLA-' +
        NetMHCpan_df.loc[mask, 0]
        .str.replace(':', '', regex=False)
    )
    # Vectorize mhc_class determination
    dataset['netmhcpan_pseudosequence'] = get_MHC_pseudosequences_(dataset['allele'], NetMHCpan_df)
    dataset['mhc_class'] = np.where(dataset['allele'].str.contains('DP|DQ|DR|DM'), 'II', 'I')
    return dataset


def get_psuedosequence_PMGen(mhc_sequence, PMGen_psuedoseq_dict):
    """
    Get the pseudo sequence for the given MHC sequence from the PMGen pseudo sequence dictionary.
    """
    # Check if the MHC sequence is present in the dictionary
    if mhc_sequence in PMGen_psuedoseq_dict:
        return PMGen_psuedoseq_dict[PMGen_psuedoseq_dict['mhc_sequence'] == mhc_sequence]['pseudo_sequence'].values[0]
    else:
        return None


def update_dataset_with_PMGen_psuedoseq(dataset, PMGen_psuedoseq_dict):
    """
    Update the dataset with additional columns and values.
    """
    # Add a new column 'pseudo_sequence' and fill with the pseudo sequence from PMGen
    dataset['pseudo_sequence'] = dataset['mhc_sequence'].apply(get_psuedosequence_PMGen, args=(PMGen_psuedoseq_dict,))
    return dataset


def main():
    # Define folders to process
    folders = {
        'test': "../../database/ConvNeXt/test",
        'train': "../../database/ConvNeXt/ms_train_data"
    }

    # Output directory for the harmonized files
    output_dir = "../../database/ConvNeXt/harmonized_MHC_I"
    os.makedirs(output_dir, exist_ok=True)

    # Process and combine datasets
    harmonized_datasets = process_and_combine_datasets(folders)

    # Save each dataset separately
    for dataset_name, dataset in harmonized_datasets.items():
        # Output file path
        output_file = os.path.join(output_dir, f"harmonized_{dataset_name}_dataset.csv")

        # Save the harmonized dataset
        dataset.to_csv(output_file, index=False)
        print(f"\nHarmonized '{dataset_name}' dataset saved to: {output_file}")

        # save the analysis to a file
        analysis_file = os.path.join(output_dir, f"analysis_{dataset_name}.txt")
        with open(analysis_file, 'w') as f:
            f.write(f"Dataset Summary for '{dataset_name}' Dataset:\n")
            f.write(f"Total rows: {len(dataset)}\n\n")
            if 'allele' in dataset.columns:
                f.write("Alleles peptide count:\n")
                f.write(str(dataset['allele'].value_counts(dropna=False)))
            else:
                f.write("No 'allele' column present in the dataset.")

    if 'test' in harmonized_datasets and 'train' in harmonized_datasets:
        test_alleles = set(harmonized_datasets['test']['allele'].dropna().unique())
        train_alleles = set(harmonized_datasets['train']['allele'].dropna().unique())

        # Alleles exclusive to 'test'
        unique_test_alleles = test_alleles - train_alleles

        # Convert to a sorted list for better readability
        unique_test_alleles_sorted = sorted(unique_test_alleles)

        # save to a text file as well
        unique_alleles_txt_file = os.path.join(output_dir, "unique_test_alleles.txt")
        with open(unique_alleles_txt_file, 'w') as f:
            for allele in unique_test_alleles_sorted:
                f.write(f"{allele}\n")
        print(f"Unique alleles exclusive to 'test' saved to: {unique_alleles_txt_file}")

    else:
        print("Both 'test' and 'train' datasets must be processed to identify unique alleles.")

    # load the harmonized PMGen dataset
    PMGen_harmonized_df = pd.read_csv("../../database/PMGen/data/harmonized_PMgen_mapping.csv")

    # select the homo specie and MHC I
    PMGen_harmonized_df = PMGen_harmonized_df[PMGen_harmonized_df["species"] == "homo"]
    PMGen_harmonized_df = PMGen_harmonized_df[PMGen_harmonized_df["mhc_class"] == "I"]

    # # load the test and train datasets
    test = pd.read_csv("../../database/ConvNeXt/harmonized_MHC_I/harmonized_test_dataset.csv")
    train = pd.read_csv("../../database/ConvNeXt/harmonized_MHC_I/harmonized_train_dataset.csv")


    test=update_dataset(test, PMGen_harmonized_df)
    train=update_dataset(train, PMGen_harmonized_df)


    print(test.head())
    print(train.head())

    # # load the NetMHCpan dataset
    net_mhcpan_path1 = "../../database/NetMHCpan/MHC_pseudo.dat"
    net_mhcpan_path2 = "../../database/NetMHCpan/pseudosequence.2023.all.X.dat"

    # load the file and concatenate them
    net_mhcpan1 = pd.read_csv(net_mhcpan_path1, delim_whitespace=True, header=None)
    net_mhcpan2 = pd.read_csv(net_mhcpan_path2, delim_whitespace=True, header=None)
    net_mhcpan = pd.concat([net_mhcpan1, net_mhcpan2], ignore_index=True)


    test = update_dataset2(test, net_mhcpan)

    if 'binder_state' in test.columns:
        test['binding_label'] = test.apply(
            lambda row: row['binder_state'] if pd.notna(row['binder_state']) else row['binder_state'],
            axis=1
        )

    pseudo_sequences_dict_path = "../../database/PMGen/data/PMGen_pseudoseq.tsv"
    pseudo_sequences_dict = pd.read_csv(pseudo_sequences_dict_path)

    # get the pseudo sequence
    test = update_dataset_with_PMGen_psuedoseq(test, pseudo_sequences_dict)

    test.to_csv("../../database/ConvNeXt/harmonized_MHC_I/harmonized_test_dataset.csv", index=False)

    train = update_dataset2(train, net_mhcpan)

    if 'binder_state' in train.columns:
        train['binding_label'] = train.apply(
            lambda row: row['binder_state'] if pd.notna(row['binder_state']) else row['binder_state'],
            axis=1
        )

    # if log_value exists, fill null binding_labels with log_value >= 0.428 as 1 and 0 otherwise
    if 'log_value' in train.columns:
        train['binding_label'] = train['binding_label'].combine_first(train['log_value'].apply(lambda x: 1 if x >= 0.428 else 0))

    train = update_dataset_with_PMGen_psuedoseq(train, pseudo_sequences_dict)

    train.to_csv("../../database/ConvNeXt/harmonized_MHC_I/harmonized_train_dataset.csv", index=False)


if __name__ == "__main__":
    main()