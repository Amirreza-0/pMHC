import os
import pandas as pd
import numpy as np
import re

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
    'convnext_value': ['value', 'ConvNeXT_value', 'convnext_value', 'prediction', 'score']
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
    required_columns = ['allele', 'peptide_length', 'long_mer', '9mer', 'convnext_value']
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


# def get_allele_seq(files, mhc_type=1, species=''):
#
#     PMGen_path = "../../database/PMGen/data/HLA_alleles/processed/mmseq_clust"
#
#     # Verify that the path exists
#     if not os.path.exists(PMGen_path):
#         print(f"Error: The specified path does not exist: {PMGen_path}")
#         return pd.DataFrame()
#     if mhc_type==1:
#         all_files = [f for f in os.listdir(PMGen_path) if f.endswith('.fasta') and 'all_seqs' in f and f[0]!='D']
#     elif mhc_type==2:
#         all_files = [f for f in os.listdir(PMGen_path) if f.endswith('.fasta') and 'all_seqs' in f and f[0] == 'D']
#
#     if not all_files:
#         print(f"No matching '.fasta' files found in: {PMGen_path}")
#         return pd.DataFrame()
#
#     print(f"Found {len(all_files)} '.fasta' files to process.")
#
#     harmonized_df = pd.DataFrame(columns=['allele', 'mhc_sequence', 'class'])
#
#     # Updated regex pattern to capture both Class I and Class II alleles anywhere in the line
#     pattern = r'(DP|DQ|DR|A|B|C)\*\d+(?::\d+)*(?:[A-Z]?)'
#
#     for file in all_files:
#         file_path = os.path.join(PMGen_path, file)
#         print(f"Processing file: {file_path}")
#         with open(file_path, 'r') as f:
#             lines = f.readlines()
#             for i, line in enumerate(lines):
#                 if line.startswith(">"):
#                     allele_line = line.strip()
#                     # Search for the allele pattern anywhere in the line
#                     match = re.search(pattern, allele_line)
#                     if match:
#                         raw_allele = match.group()
#                         print(f"Found allele: {raw_allele} in line: {allele_line}")
#                         # Clean the allele name
#                         cleaned_allele = clean_allele(raw_allele)
#                         # Determine MHC class
#                         mhc_class = determine_mhc_class(raw_allele)
#                     else:
#                         # If no match is found, skip this entry with a debug message
#                         print(f"No allele match found in line: {allele_line}")
#                         continue
#
#                     # Ensure that the next line exists and is a sequence
#                     if i + 1 < len(lines):
#                         sequence = lines[i + 1].strip()
#                         if sequence.startswith(">"):
#                             print(f"Warning: Expected sequence after line {i}, but found another header.")
#                             sequence = ""  # Invalid sequence entry
#                         else:
#                             print(
#                                 f"Extracted sequence for allele {raw_allele}: {sequence[:30]}...")  # Show first 30 chars
#                     else:
#                         print(f"Warning: No sequence found for allele {raw_allele} at end of file.")
#                         sequence = ""
#
#                     harmonized_df = harmonized_df._append(
#                         {'allele': cleaned_allele, 'mhc_sequence': sequence, 'class': mhc_class},
#                         ignore_index=True
#                     )
#
#     # Remove rows without sequence
#     initial_count = len(harmonized_df)
#     harmonized_df = harmonized_df[harmonized_df['mhc_sequence'] != ""]
#     removed_count = initial_count - len(harmonized_df)
#     if removed_count > 0:
#         print(f"Removed {removed_count} rows without sequences.")
#
#     # Drop duplicates
#     before_dedup = len(harmonized_df)
#     harmonized_df = harmonized_df.drop_duplicates(subset=['allele', 'mhc_sequence'])
#     deduped_count = before_dedup - len(harmonized_df)
#     if deduped_count > 0:
#         print(f"Removed {deduped_count} duplicate rows.")
#
#     # Clean the allele column
#     harmonized_df['allele'] = harmonized_df['allele'].apply(clean_allele)
#
#     # Save the harmonized dataset
#     output_path = "../../database/PMGen/data/harmonized_PMgen_mapping.csv"
#     harmonized_df.to_csv(output_path, index=False)
#     print(f"Harmonized PMgen mapping saved to: {output_path}")
#
#     # Final count
#     print(f"Total entries in harmonized mapping: {len(harmonized_df)}")
#
#     return harmonized_df








def get_MHC_sequences_(alleles, PMGen_harmonized_df):
    """
    Get MHC sequences from PMGen_dataset dataset.
    Expects alleles is a pandas Series of allele names.
    """
    def get_seq(allele):
        matching_rows = PMGen_harmonized_df[PMGen_harmonized_df['simple_allele'].str.contains(allele, na=False)]
        if not matching_rows.empty:
            return matching_rows['mhc_sequence'].values[0]
        return None
    return alleles.apply(get_seq)



def update_dataset(dataset, PMGen_harmonized_df):
    """
    Update the dataset with additional columns and values.
    """
    # simplify the allele names from PMGen
    # if the allele length is longer than 10
    # remove * and : from the allele names and add HLA- prefix from the PMGen dataset if the length of the value is lower than 15
    PMGen_harmonized_df['simple_allele'] = PMGen_harmonized_df['simple_allele'].apply(lambda x: 'HLA-' + x.replace('*', '').replace(':', '') if len(x) < 15 else x)
    dataset['mhc_sequence'] = dataset['allele'].apply(get_MHC_sequences_(dataset['allele'], PMGen_harmonized_df))


    # Add a new column 'mhc_sequence' and fill with the MHC sequence from ParseMHCflold
    dataset['mhc_sequence'] = dataset['allele'].apply(get_MHC_sequences_(dataset['allele'], PMGen_harmonized_df))
    # Add a new column 'mhc_class' and fill with 'I' or 'II' based on the allele name
    dataset['mhc_class'] = dataset['allele'].apply(lambda x: 'II' if any(sub in x for sub in ['DP', 'DQ', 'DR']) else 'I')
    # dataset['matched_allele'] = dataset['allele'].apply(get_MHC_sequences_(dataset['allele'], PMGen_harmonized_df))[0]
    # dataset['multiple_matches'] = dataset['allele'].apply(get_MHC_sequences_(dataset['allele'], PMGen_harmonized_df))[1]

    return dataset



def main():
    # # Define folders to process
    # folders = {
    #     'test': "../../database/ConvNeXt/test",
    #     'train': "../../database/ConvNeXt/ms_train_data"
    # }
    #
    # # Output directory for the harmonized files
    # output_dir = "../../database/ConvNeXt/harmonized_MHC_I"
    # os.makedirs(output_dir, exist_ok=True)
    #
    # # Process and combine datasets
    # harmonized_datasets = process_and_combine_datasets(folders)
    #
    # # Save each dataset separately
    # for dataset_name, dataset in harmonized_datasets.items():
    #     # Output file path
    #     output_file = os.path.join(output_dir, f"harmonized_{dataset_name}_dataset.csv")
    #
    #     # Save the harmonized dataset
    #     dataset.to_csv(output_file, index=False)
    #     print(f"\nHarmonized '{dataset_name}' dataset saved to: {output_file}")
    #
    #     # save the analysis to a file
    #     analysis_file = os.path.join(output_dir, f"analysis_{dataset_name}.txt")
    #     with open(analysis_file, 'w') as f:
    #         f.write(f"Dataset Summary for '{dataset_name}' Dataset:\n")
    #         f.write(f"Total rows: {len(dataset)}\n\n")
    #         if 'allele' in dataset.columns:
    #             f.write("Alleles peptide count:\n")
    #             f.write(str(dataset['allele'].value_counts(dropna=False)))
    #         else:
    #             f.write("No 'allele' column present in the dataset.")
    #
    # if 'test' in harmonized_datasets and 'train' in harmonized_datasets:
    #     test_alleles = set(harmonized_datasets['test']['allele'].dropna().unique())
    #     train_alleles = set(harmonized_datasets['train']['allele'].dropna().unique())
    #
    #     # Alleles exclusive to 'test'
    #     unique_test_alleles = test_alleles - train_alleles
    #
    #     # Convert to a sorted list for better readability
    #     unique_test_alleles_sorted = sorted(unique_test_alleles)
    #
    #     # save to a text file as well
    #     unique_alleles_txt_file = os.path.join(output_dir, "unique_test_alleles.txt")
    #     with open(unique_alleles_txt_file, 'w') as f:
    #         for allele in unique_test_alleles_sorted:
    #             f.write(f"{allele}\n")
    #     print(f"Unique alleles exclusive to 'test' saved to: {unique_alleles_txt_file}")
    #
    # else:
    #     print("Both 'test' and 'train' datasets must be processed to identify unique alleles.")

    # load the harmonized PMGen dataset
    PMGen_harmonized_df = pd.read_csv("../../database/PMGen/data/harmonized_PMgen_mapping.csv")

    # load the test and train datasets
    test = pd.read_csv("../../database/ConvNeXt/harmonized_MHC_I/harmonized_test_dataset.csv")
    train = pd.read_csv("../../database/ConvNeXt/harmonized_MHC_I/harmonized_train_dataset.csv")


    test=update_dataset(test, PMGen_harmonized_df)
    train=update_dataset(train, PMGen_harmonized_df)

    # save the updated datasets
    test.to_csv("../../database/ConvNeXt/harmonized_MHC_I/harmonized_test_dataset.csv", index=False)
    train.to_csv("../../database/ConvNeXt/harmonized_MHC_I/harmonized_train_dataset.csv", index=False)

    print(test.head())
    print(train.head())

if __name__ == "__main__":
    main()