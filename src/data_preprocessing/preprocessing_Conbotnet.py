# load all data based on the .yaml file
import pandas as pd

dataset_path = "../../database/Conbotnet"

"""mhc_seq: data/pseudosequence.2016.all.X.dat
train: data/data.txt
pretrain: data/data_pretrain.txt
#test: data/test_binary_2023.txt
test: data/test_ic50_2023.txt
cv_id: data/cv_id.txt
binding: data/binding.txt
seq2logo: data/seq2logo.txt
"""

# Load the data
mhc_seq = pd.read_csv(dataset_path + "/data/pseudosequence.2016.all.X.dat", sep="\t")
train = pd.read_csv(dataset_path + "/data/data.txt", sep="\t")
pretrain = pd.read_csv(dataset_path + "/data/data_pretrain.txt", sep="\t")
test = pd.read_csv(dataset_path + "/data/test_ic50_2023.txt", sep="\t")
cv_id = pd.read_csv(dataset_path + "/data/cv_id.txt", sep="\t")
binding = pd.read_csv(dataset_path + "/data/binding.txt", sep="\t")
seq2logo = pd.read_csv(dataset_path + "/data/seq2logo.txt", sep="\t")


def get_mhc_name_seq(mhc_name_seq_file):
    mhc_name_seq = {}
    with open(mhc_name_seq_file) as fp:
        for line in fp:
            mhc_name, mhc_seq = line.split()
            mhc_name_seq[mhc_name] = mhc_seq
    return mhc_name_seq


def get_data(data_file, mhc_name_seq):
    data_list = []
    with open(data_file) as fp:
        for line in fp:
            peptide_seq, score, mhc_name = line.split()
            if len(peptide_seq) >= 9:
                data_list.append((mhc_name, peptide_seq, mhc_name_seq[mhc_name], float(score)))
    return data_list


def get_binding_data(data_file, mhc_name_seq, core_len=9):
    data_list = []
    with open(data_file) as fp:
        for line in fp:
            pdb, mhc_name, mhc_seq, peptide_seq, core = line.split()
            assert len(core) == core_len
            data_list.append(((pdb, mhc_name, core), peptide_seq, mhc_name_seq[mhc_name], 0.0))
    return data_list


def get_seq2logo_data(data_file, mhc_name, mhc_seq):
    with open(data_file) as fp:
        return [(mhc_name, line.strip(), mhc_seq, 0.0) for line in fp]


# Get the MHC name and sequence
mhc_name_seq = get_mhc_name_seq(dataset_path + "/data/pseudosequence.2016.all.X.dat")
print(mhc_name_seq)