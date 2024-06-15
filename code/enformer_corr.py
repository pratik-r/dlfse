# Take the output from Enformer and calculate pearson & spearman correlations
# Both across individuals (hard) and across genes (easier), on all tracks

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import sys

# Data folder holding genes + donors of interest, true gene exp values
data_path = "/home/panwei/shared/nucleotide_transformer/multi_tissue/data/"
# Testing donors path
test_donors_path = data_path + "donors_test.csv"
train_donors_path = data_path + "donors_train.csv"

# Output folder for Enformer predictions after aggregating over bins and haps
enf_pred_final_path = "/home/panwei/shared/nucleotide_transformer/multi_tissue/enformer_predictions/"

# Enformer output for each haplotype
enf_output_folder_hap1 = "/home/panwei/shared/nucleotide_transformer/multi_tissue/enf_pred_hap1/donor_set_0/"
enf_output_folder_hap2 = "/home/panwei/shared/nucleotide_transformer/multi_tissue/enf_pred_hap2/donor_set_0/"

# Output folder for correlations
corr_folder = "/home/panwei/shared/nucleotide_transformer/multi_tissue/enf_corr_train_avg3_400/" # Create before running

# Tissues
tissues = ["Muscle_Skeletal","Whole_Blood","Skin_Sun_Exposed_Lower_leg","Artery_Tibial","Adipose_Subcutaneous"]

# Raw data unnormalized paths
raw_data_names = {
    "Muscle_Skeletal": 'gene_tpm_2017-06-05_v8_muscle_skeletal.gct.gz', 
    "Whole_Blood": 'gene_tpm_2017-06-05_v8_whole_blood.gct.gz',
    "Skin_Sun_Exposed_Lower_leg": 'gene_tpm_2017-06-05_v8_skin_sun_exposed_lower_leg.gct.gz', 
    "Artery_Tibial": 'gene_tpm_2017-06-05_v8_artery_tibial.gct.gz',
    "Adipose_Subcutaneous": 'gene_tpm_2017-06-05_v8_adipose_subcutaneous.gct.gz'
}
gene_reads_names = {
    "Muscle_Skeletal": 'gene_reads_2017-06-05_v8_muscle_skeletal.gct.gz', 
    "Whole_Blood": 'gene_reads_2017-06-05_v8_whole_blood.gct.gz',
    "Skin_Sun_Exposed_Lower_leg": 'gene_reads_2017-06-05_v8_skin_sun_exposed_lower_leg.gct.gz', 
    "Artery_Tibial": 'gene_reads_2017-06-05_v8_artery_tibial.gct.gz',
    "Adipose_Subcutaneous": 'gene_reads_2017-06-05_v8_adipose_subcutaneous.gct.gz'
}

if __name__ == "__main__":
    # Get the list of genes
    short_genes_df = pd.read_csv(data_path + "genes_short.csv")
    long_genes_df = pd.read_csv(data_path + "genes_long.csv")
    nh_genes_df = pd.read_csv(data_path + "genes_nh.csv")
    gene_info = pd.concat([short_genes_df, long_genes_df])#, #nh_genes_df])
    genes = list(gene_info['gene_id'])
    # Has overlapping indels/SNPs so bad conseq, skipping
    #genes.remove('ENSG00000226210.3') # 444
    genes.remove('ENSG00000170122.5') # 195
    genes.remove('ENSG00000268836.1') # 37
    #genes.remove('ENSG00000269981.1') # 485
    #genes = genes[:198] # try 200 shortest first

    # Tissue index
    tissue_index = int(sys.argv[1])
    tissue = tissues[tissue_index]
    
    # Get the list of donors
    donors = list(pd.read_csv(train_donors_path, header=0)["0"]) # 58 test donors

    #Load in Enformer outputs for all genes and get avg of 2 haplotypes, middle 3 bins
    enformer_res = []
    for gene in genes:
        with open(enf_output_folder_hap1 + gene + ".npy", "rb") as f:
            enf_output_hap1 = np.load(f)
        # Take avg of middle 3/11 bins (only saved middle 17 bins of Enformer), all tracks
        gene_res_hap1 = np.mean(enf_output_hap1[:, 7:10, :], axis = 1) # num donors x num tracks
        #gene_res_hap1 = np.mean(enf_output_hap1[:, 3:14, :], axis = 1) # num donors x num tracks
        #gene_res_hap1 = np.sum(np.log(enf_output_hap1[:, 7:10, :]), axis = 1)
        with open(enf_output_folder_hap2 + gene + ".npy", "rb") as f:
            enf_output_hap2 = np.load(f)
        gene_res_hap2 = np.mean(enf_output_hap2[:, 7:10, :], axis = 1) # num donors x num tracks
        #gene_res_hap2 = np.mean(enf_output_hap2[:, 3:14, :], axis = 1) # num donors x num tracks
        #gene_res_hap2 = np.sum(np.log(enf_output_hap2[:, 7:10, :]), axis = 1)
        # Average haplotypes
        enformer_res.append((gene_res_hap1 + gene_res_hap2) / 2)
    enf = np.array(enformer_res) # genes x donors x tracks; all enformer output
    # with open(enf_pred_final_path + "enf_pred_final_all_tracks_avg3.npy", "wb") as f:
    #     np.save(f, enf)

    #If ran previously
    # with open(enf_pred_final_path + "enf_pred_final_all_tracks_avg3.npy", "rb") as f:
    #     enf = np.load(f)

    # # Gene expression truths (normalized data)
    # gene_exp_df = pd.read_csv(data_path + tissue + ".csv")
    # gene_exp_df = gene_exp_df.rename(columns={'Unnamed: 0':'donor'})
    # gene_exp_df = gene_exp_df.set_index('donor')
    
    # Raw exp version
    gene_exp_df = pd.read_csv(data_path + "raw_tpm/" + raw_data_names[tissue], sep='\t', header=2) # TPM
    #gene_exp_df = pd.read_csv(data_path + "read_counts/" + gene_reads_names[tissue], sep='\t', header=2) # Read counts
    gene_exp_df = gene_exp_df.drop(columns=["Description", "id"])
    gene_exp_df = gene_exp_df.rename(columns={"Name": "gene_id"})
    gene_exp_df.set_index('gene_id', inplace=True)
    gene_exp_df.columns = gene_exp_df.columns.str.extract(r'([^-\s]+-[^-\s]+)')[0]  # Cut down the ID from 5 parts to standard format GTEX-id
    gene_exp_df = gene_exp_df.T  # Now, colnames are ensg gene ids, and rownames are donors
    #gene_exp_df = (gene_exp_df - gene_exp_df.mean()) / gene_exp_df.std() # Normalize across all genes and donors

    # Sort exp df in the order of genes in genes and donors in donors list
    gene_exp_df = gene_exp_df.reindex(columns=genes) # also drop columns (rows) not in the genes (donors) list
    gene_exp_df = gene_exp_df.reindex(donors)

    ##############################################################################################################
    # Calculate correlation across indivduals/donors for a fixed gene, all tracks
    r_xdonor, rho_xdonor = [], []
    for g, gene in enumerate(genes):
        exp_values = gene_exp_df[gene] # Expression values for gene of interest
        r_gene, rho_gene = [], [] # for the gene
        for track in range(enf.shape[2]):
            track_r, track_r_pv = pearsonr(exp_values, enf[g, :, track]) # Fixed gene, track + all donors
            r_gene.append(track_r) # don't need p-value
            track_rho, track_rho_pv = spearmanr(exp_values, enf[g, :, track])
            rho_gene.append(track_rho)
        r_xdonor.append(r_gene)
        rho_xdonor.append(rho_gene)
    
    r_xdonor_np = np.array(r_xdonor) # Convert to np for saving
    rho_xdonor_np = np.array(rho_xdonor)

    with open(corr_folder + "r_xdonor_" + tissue + ".npy", "wb") as f:
        np.save(f, r_xdonor_np)
    with open(corr_folder + "rho_xdonor_" + tissue + ".npy", "wb") as f:
        np.save(f, rho_xdonor_np)

    del r_xdonor, rho_xdonor, r_xdonor_np, rho_xdonor_np

    #############################################################################################################
    # Calculate correlation across genes for a fixed individual
    r_xgene, rho_xgene = [], []
    for d, donor in enumerate(donors):
        # Get expression values for donor
        donor_exp = list(gene_exp_df.loc[donor])
        r_donor, rho_donor = [], []
        for track in range(0, enf.shape[2]): # Iterate over all tracks
            track_r, track_r_pv = pearsonr(donor_exp, enf[:, d, track]) # All genes, one donor, one track
            r_donor.append(track_r)
            track_rho, track_rho_pv = spearmanr(donor_exp, enf[:, d, track])
            rho_donor.append(track_rho)
        r_xgene.append(r_donor)
        rho_xgene.append(rho_donor)

    r_xgene_np = np.array(r_xgene) # Convert to np for saving (dim donors x tracks)
    rho_xgene_np = np.array(rho_xgene)

    with open(corr_folder + "r_xgene_" + tissue + ".npy", "wb") as f:
        np.save(f, r_xgene_np)
    with open(corr_folder + "rho_xgene_" + tissue + ".npy", "wb") as f:
        np.save(f, rho_xgene_np)
