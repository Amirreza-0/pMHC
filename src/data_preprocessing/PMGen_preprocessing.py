import os
import pandas as pd
import numpy as np
import re

def get_allele_seq(file, mhc_class, specie, allele_character):
    """
    Get allele sequences from the PMGen dataset.
    :param file: A path to a FASTA file containing allele sequences
    :param mhc_class: The MHC type (I or II): 1 for MHC-I, 2 for MHC-II
    :param specie: The species of the allele sequences
    :param allele_character: The character to match the allele names for search, e.g., "A"
    :return: df: A DataFrame containing the allele_line, mhc_sequence, species, mhc_class, and simple_allele
    """
    # create a df to store the allele, sequence and class
    df = pd.DataFrame(columns=['allele_line', 'mhc_sequence', 'species', 'mhc_class', 'simple_allele'])

    with open(file, "r") as f:
        lines = f.readlines()

    current_header = ""
    current_sequence = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_header and current_sequence:
                # Extract simple allele from header using allele_characters
                pattern = fr"{allele_character}\*\d+(?::\d+)*[A-Za-z]?"
                match = re.search(pattern, current_header)
                simple_allele = match.group(0) if match else current_header[1:].split()[0]
                new_row = pd.DataFrame([{
                    'allele_line': current_header,
                    'mhc_sequence': current_sequence,
                    'species': specie,
                    'mhc_class': mhc_class,
                    'simple_allele': simple_allele
                }])
                df = pd.concat([df, new_row], ignore_index=True)
            current_header = line
            current_sequence = ""
        else:
            current_sequence += line
    # Append last record if available
    if current_header and current_sequence:
        pattern = fr"{allele_character}\*\d+(?::\d+)*[A-Za-z]?"
        match = re.search(pattern, current_header)
        simple_allele = match.group(0) if match else current_header[1:].split()[0]
        new_row = pd.DataFrame([{
            'allele_line': current_header,
            'mhc_sequence': current_sequence,
            'species': specie,
            'mhc_class': mhc_class,
            'simple_allele': simple_allele
        }])
        df = pd.concat([df, new_row], ignore_index=True)

    return df


def get_allele_seq_multi(file, mhc_class, specie, allele_characters):
    """
    Get allele sequences from the PMGen dataset that may contain multiple entries per header.
    This function uses a regex pattern to extract all simple alleles from each header.

    :param file: A path to a FASTA file containing allele sequences
    :param mhc_class: The MHC type (I or II)
    :param specie: The species of the allele sequences
    :param allele_characters: The allele character(s) to include in the regex pattern, e.g., ["DPA", "DPB"]
    :return: A DataFrame containing the allele_line, mhc_sequence, species, mhc_class, and simple_allele for each entry
    """
    df = pd.DataFrame(columns=['allele_line', 'mhc_sequence', 'species', 'mhc_class', 'simple_allele'])

    with open(file, "r") as f:
        lines = f.readlines()

    current_headers = ""
    current_sequences = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_headers and len(current_sequences) == len(allele_characters):
                # Extract simple alleles from header using allele_characters
                simple_alleles = []
                for allele_character in allele_characters:
                    pattern = fr"{allele_character}\*\d+(?::\d+)*[A-Za-z]?"
                    match = re.search(pattern, current_headers)
                    simple_allele = match.group(0) if match else current_headers[1:].split()[0]
                    simple_alleles.append(simple_allele)
                if not simple_alleles:
                    print(f"Warning: No simple allele found in header: {current_headers}")
                    continue

                new_row = pd.DataFrame([{
                    'allele_line': current_headers,
                    'mhc_sequence': "/".join(current_sequences),
                    'species': specie,
                    'mhc_class': mhc_class,
                    'simple_allele': "-".join(simple_alleles)
                }])
                df = pd.concat([df, new_row], ignore_index=True)
            current_headers = line
            current_sequences = []
        else:
            current_sequences = line.split('/')

    # Append last record if available
    if current_headers and len(current_sequences) == len(allele_characters):
        simple_alleles = []
        for allele_character in allele_characters:
            pattern = fr"{allele_character}\*\d+(?::\d+)*[A-Za-z]?"
            match = re.search(pattern, current_headers)
            simple_allele = match.group(0) if match else current_headers[1:].split()[0]
            simple_alleles.append(simple_allele)
        if not simple_alleles:
            print(f"Warning: No simple allele found in header: {current_headers}")
        for i, simple_allele in enumerate(simple_alleles):
            new_row = pd.DataFrame([{
                'allele_line': current_headers,
                'mhc_sequence': current_sequences[i],
                'species': specie,
                'mhc_class': mhc_class,
                'simple_allele': simple_allele
            }])
            df = pd.concat([df, new_row], ignore_index=True)

    return df


def create_harmonized_PMgen(PMGen_path_class_I="../../database/PMGen/data/HLA_alleles/processed/mmseq_clust/",
                            PMGen_path_class_II="../../database/PMGen/data/HLA_alleles/processed/mmseq_clust/mhc2_rep_combinations/",
                            review=False):
    """
    Iterate over all the PMGen files from both class I and class II. For each specie and for each MHC class, extract the allele name and the sequence using the get_allele_seq
    and the corresponding mapping to get the mhc_class. Save the results in a DataFrame.

    The file names are in the format: <mhc_type>-<specie>_all_seqs.fasta
    The mhc_type is the allele_character.

    class I has a one-letter allele_character eg. (A, B, C)
    class II has a three-letter allele_character eg. (DPA, DPB, DQA, DQB, DRA, DRB)

    :param PMGen_path_class_I: Path to the PMGen files for Class I
    :param PMGen_path_class_II: Path to the PMGen files for Class II
    :param review: If True, display detected mhc_type and specie, plus the file head, before processing each file for manual confirmation.
    :return: harmonized_df: A DataFrame containing the allele_line, mhc_sequence, species, mhc_class, and simple_allele
    """
    harmonized_df = pd.DataFrame(columns=['allele_line', 'mhc_sequence', 'species', 'mhc_class', 'simple_allele'])

    # Process Class I files
    if not os.path.exists(PMGen_path_class_I):
        print(f"Error: PMGen_path does not exist: {PMGen_path_class_I}")
    else:
        all_files = [f for f in os.listdir(PMGen_path_class_I) if f.endswith('.fasta') and '_all_seqs' in f]
        print(f"Found {len(all_files)} file(s) in {PMGen_path_class_I} for Class I")

        for file in all_files:
            parts = file.split('-')
            if len(parts) < 2:
                print(f"Skipping file with unexpected format: {file}")
                continue
            allele_character = parts[0]
            # Skip files indicating Class II (allele_character longer than one character)
            if len(allele_character) > 1:
                print(f"Skipping file with class II allele character in Class I folder: {file}")
                continue
            remainder = parts[1]
            specie = remainder.split('_')[0]
            mhc_class = "I"
            file_path = os.path.join(PMGen_path_class_I, file)

            if review:
                try:
                    with open(file_path, "r") as preview_file:
                        head_lines = preview_file.readlines()[:5]
                    print(f"\nDetected mhc_type: {allele_character}, specie: {specie} in file: {file_path}")
                    print("File head:")
                    print("".join(head_lines))
                    input("Press Enter to confirm processing this file...")
                except Exception as e:
                    print(f"Error reading file head for {file_path}: {e}")
                    continue

            print(f"Processing file: {file_path}")
            try:
                df_temp = get_allele_seq(file_path, mhc_class, specie, allele_character)
                harmonized_df = pd.concat([harmonized_df, df_temp], ignore_index=True)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # Process Class II files
    if not os.path.exists(PMGen_path_class_II):
        print(f"Error: PMGen_path does not exist: {PMGen_path_class_II}")
    else:
        all_files = [f for f in os.listdir(PMGen_path_class_II) if f.endswith('.fasta')]
        print(f"Found {len(all_files)} file(s) in {PMGen_path_class_II} for Class II")

        for file in all_files:
            parts = file.split('-')
            if len(parts) < 2:
                print(f"Skipping file with unexpected format: {file}")
                continue
            allele_characters = parts[0].split('_')
            remainder = parts[1]
            specie = remainder.split('_')[0].split('.')[0]
            mhc_class = "II"
            file_path = os.path.join(PMGen_path_class_II, file)

            if review:
                try:
                    with open(file_path, "r") as preview_file:
                        head_lines = preview_file.readlines()[:5]
                    print(f"\nDetected mhc_types: {allele_characters}, specie: {specie} in file: {file_path}")
                    print("File head:")
                    print("".join(head_lines))
                    input("Press Enter to confirm processing this file...")
                except Exception as e:
                    print(f"Error reading file head for {file_path}: {e}")
                    continue

            print(f"Processing file: {file_path}")
            try:
                df_temp = get_allele_seq_multi(file_path, mhc_class, specie, allele_characters)
                harmonized_df = pd.concat([harmonized_df, df_temp], ignore_index=True)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    harmonized_df.drop_duplicates(subset=['allele_line', 'mhc_sequence'], inplace=True)
    harmonized_df = harmonized_df[harmonized_df['mhc_sequence'] != ""]
    output_path = "../../database/PMGen/data/harmonized_PMgen_mapping.csv"
    harmonized_df.to_csv(output_path, index=False)
    print(f"Harmonized PMgen mapping saved to: {output_path}")
    print(f"Total entries in harmonized mapping: {len(harmonized_df)}")
    return harmonized_df

harmonized_df = create_harmonized_PMgen()
print(harmonized_df.head())