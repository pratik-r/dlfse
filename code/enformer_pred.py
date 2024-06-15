# Script to get predictions from enformer
# Runs maternal and paternal seperately
# Set up to use three sets of genes from multiple tissue list

import numpy as np
np.bool = np.bool_ # Due to deprecation & dependency chains
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import os
from os import listdir
from os.path import isfile, join
import sys
from tqdm import tqdm

enformer_seq_len = 393_216
window = int(enformer_seq_len / 2) # window to create consensus sequence

# Each gene will be one numpy array in the output folder
output_folder_hap1 = "/home/panwei/shared/nucleotide_transformer/multi_tissue/enf_pred_hap1"
output_folder_hap2 = "/home/panwei/shared/nucleotide_transformer/multi_tissue/enf_pred_hap2"

# A path to bcftools/samtools to run it from Python
bcftools = "/home/panwei/pai00032/installations/bcftools-1.9/bcftools"
samtools = "/home/panwei/pai00032/installations/samtools-1.9/samtools"

# Variables to store filenames
# SNP data
vcf_file = "/home/panwei/shared/GTEx_v8/GTEx_Analysis_v8_WholeGenomeSeq/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.SHAPEIT2_phased.vcf.gz"

# Reference genome file (from https://console.cloud.google.com/storage/browser/genomics-public-data/resources/broad/hg38/v0)
ref_genome_file = "/home/panwei/shared/nucleotide_transformer/data/Homo_sapiens_assembly38.fasta"

# Data folder holding genes and donors of interest
data_path = "/home/panwei/shared/nucleotide_transformer/multi_tissue/data/"
# Donors path
test_donors_path = data_path + "donors_test.csv"
train_donors_path = data_path + "donors_train.csv"
val_donors_path = data_path + "donors_val.csv"
donors_paths = [train_donors_path, val_donors_path, test_donors_path]

# Helper function to adjust sequence to enformer required length; pads with 'N'
def adjust_string_to_length(input_string, target_length):
    if len(input_string) < target_length:
        # Pad the string with 'N' on both sides
        left_padding = (target_length - len(input_string)) // 2
        right_padding = target_length - len(input_string) - left_padding
        adjusted_string = 'N' * left_padding + input_string + 'N' * right_padding
    elif len(input_string) > target_length:
        # Trim the string from both sides to the target length
        excess_length = len(input_string) - target_length
        trim_start = excess_length // 2
        trim_end = excess_length - trim_start
        adjusted_string = input_string[trim_start:-trim_end]  # Trim from both sides
    else: # no trimming needed
        adjusted_string = input_string
    return adjusted_string

# Taken from enformer.py in deepmind-research/enformer repo
def one_hot_encode(sequence: str,
                   alphabet: str = 'ACGT',
                   neutral_alphabet: str = 'N',
                   neutral_value: int = 0, # Changed Any to int
                   dtype=np.float32) -> np.ndarray:
  """One-hot encode sequence."""
  def to_uint8(string):
    return np.frombuffer(string.encode('ascii'), dtype=np.uint8)
  hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
  hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
  hash_table[to_uint8(neutral_alphabet)] = neutral_value
  hash_table = hash_table.astype(dtype)
  return hash_table[to_uint8(sequence)]

################################################################################################

if __name__ == "__main__":
    # Get the list of donors
    donors_set_index = int(sys.argv[3]) # Train = 0, val = 1, test = 2
    donors_df = pd.read_csv(donors_paths[donors_set_index], header=0)
    donors = list(donors_df["0"])

    gene_index = int(sys.argv[1])
    haplotype = int(sys.argv[2]) # Either 1 or 2
    output_folder = output_folder_hap1 if haplotype == 1 else output_folder_hap2

    # Get the list of genes and gene in question
    short_genes_df = pd.read_csv(data_path + "genes_short.csv")
    long_genes_df = pd.read_csv(data_path + "genes_long.csv")
    nh_genes_df = pd.read_csv(data_path + "genes_nh.csv")
    gene_info = pd.concat([short_genes_df, long_genes_df, nh_genes_df])
    gene_info.set_index('gene_id', inplace = True)

    # Gene name
    gene = gene_info.index[gene_index] # Gene name
    print(gene)

    # Need chr and start location in order to create consensus sequences
    gene_chr = gene_info.loc[gene]["chr"]
    tss = gene_info.loc[gene]["start"] # transcription start site
    # Check first bound, faidx truncates end automatically if needed
    seq_start = max(tss - window, 0) 

    ##################################################################################################
    # Create consensus sequences for each donor
    conseqs = []
    for donor in donors:
        conseq_command = samtools + " faidx " + ref_genome_file + " " + gene_chr + ":" + \
        str(seq_start) + "-" + str(tss + window)  + " | " + bcftools + " consensus -H " + str(haplotype) + " -s " + donor \
        + " " + vcf_file
        messy_conseq = os.popen(conseq_command).read().split('\n')
        conseq = ["".join(messy_conseq[1:])][0] # Concatenate lines minus header
        adjusted = adjust_string_to_length(conseq, enformer_seq_len) # Adjust length to enformer-required
        encoded = one_hot_encode(adjusted) # One-hot encode ATCG
        conseqs.append(encoded)

    np_conseqs = np.array(conseqs)
    del conseqs 

    # Download model
    enformer = hub.load('https://tfhub.dev/deepmind/enformer/1').model

    batch_size = 5
    pred_list = []
    for i in tqdm(range(0, len(donors), batch_size)):
        conseqs_batch = np_conseqs[i:i + batch_size]
        inputs = tf.convert_to_tensor(conseqs_batch)
        # Get predictions from enformer
        predictions = enformer.predict_on_batch(inputs)
        pred = predictions['human'] # enformer also produces mouse track predictions
        # Only need to save bins 440-456
        np_pred = pred.numpy()[:, 440:457, :] # all donors in batch, 17 bins, all tracks
        pred_list.append(np_pred)
        print("Finished predicting i = ", i)

    # Save output as np matrix
    result = np.concatenate(pred_list, axis=0)
    with open(output_folder + "/donor_set_" + str(donors_set_index) +"/" + gene + ".npy", "wb") as f:
        np.save(f, result)
        
    print("Finished gene ", gene)

