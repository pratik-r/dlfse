import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from gtfparse import read_gtf
import itertools
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression

gtex_dir = '/home/panwei/shared/GTEx_v8/'
exp_dir = os.path.join(gtex_dir, "GTEx_Analysis_v8_expression/expression_matrices/")
cov_dir = cov_dir = os.path.join(gtex_dir, "GTEx_Analysis_v8_expression/expression_covariates/")
data_dir = "/home/panwei/rampr009/gtex_nt_mult/data/"
num_genes = 200

SEED = 42

def largest_intersection_subset_dict(dict_of_sets, n):
    max_intersection = set()
    max_subset_keys = []
    
    keys = list(dict_of_sets.keys())
    
    # Iterate over all possible subsets of keys with at least n elements
    for r in range(n, len(keys) + 1):
        for subset_keys in itertools.combinations(keys, r):
            subset_sets = [dict_of_sets[key] for key in subset_keys]
            intersection = set.intersection(*subset_sets)
            if len(intersection) > len(max_intersection):
                max_intersection = intersection
                max_subset_keys = subset_keys
    
    return max_subset_keys, max_intersection

if __name__ == "__main__":
    # expression matrices
    files = list(filter(lambda x: x.endswith(".gz"), os.listdir(exp_dir)))
    dfs_exp = {file.replace(".v8.normalized_expression.bed.gz",""): pd.read_csv(os.path.join(exp_dir, file), sep="\t").iloc[:,3:].set_index('gene_id') for file in tqdm(files)}

    # drop gender-specific tissues
    drop = ["Breast_Mammary_Tissue","Prostate","Testis","Ovary","Vagina","Uterus"]
    for k in drop:
        del dfs_exp[k]

    # select tissues (and corresponding donors) as the 5 tissues with the largest intersection subset of donors from the 20 tissues with largest sample sizes
    num_donors = pd.Series({k: df.shape[1] for k,df in dfs_exp.items()})
    tissues = num_donors.sort_values(ascending=False).iloc[:10].index.tolist()
    donors = {k: set(dfs_exp[k].columns) for k in tissues}
    tissues_sel, donors_sel = largest_intersection_subset_dict(donors, n=5)
    
    # donors
    donors_series = pd.Series(list(donors_sel))
    donors_series.to_csv(os.path.join(data_dir, "donors.csv"))

    donors_train0, donors_test = train_test_split(donors_series, test_size = 0.2, random_state = SEED)
    donors_train, donors_val = train_test_split(donors_train0, test_size = 0.2, random_state = SEED)

    donors_train.to_csv(os.path.join(data_dir, "donors_train.csv"))
    donors_test.to_csv(os.path.join(data_dir, "donors_test.csv"))
    donors_val.to_csv(os.path.join(data_dir, "donors_val.csv"))
    
    # get list of all genes as intersection of available genes in expression matrices
    genes_all = reduce(set.intersection, [set(dfs_exp[tissue].index) for tissue in tissues_sel])

    # select list of genes as the intersection of locally heritable genes from the 5 selected tissues
    genes_h = {tissue: set(pd.read_csv(os.path.join(data_dir, f"GTExv8.ALL.{tissue}.hsq"), sep="\t").ID) for tissue in tissues_sel}
    genes_h_sel = set.intersection(*genes_h.values())

    # gtf file
    gtf_path = os.path.join(gtex_dir, "gencode.v26.GRCh38.genes.gtf")
    df_gtf = read_gtf(gtf_path, result_type="pandas", usecols=["gene_id","seqname","start","end","gene_name"], features={"gene"}).set_index('gene_id').rename(columns={"seqname": "chr"})
    df_gtf['length'] = df_gtf['end'] - df_gtf['start']

    # save all selected genes to file
    df_genes_h = df_gtf.loc[list(genes_h_sel)]
    df_genes_h.to_csv(os.path.join(data_dir, "genes_h_all.csv"))
    
    # save n shortest genes
    df_genes_h_short = df_genes_h.sort_values('length', ascending=True).iloc[:num_genes]
    df_genes_h_short.to_csv(os.path.join(data_dir, "genes_short.csv"))

    # save n longest genes
    df_genes_h_long = df_genes_h.sort_values('length', ascending=False).iloc[:num_genes]
    df_genes_h_long.to_csv(os.path.join(data_dir, "genes_long.csv"))
    
    # select list of "non-locally heritable" genes as the intersection of non-locally heritable genes from the 5 selected tissues
    genes_nh = {tissue: set(genes_all).difference(set(genes_h[tissue])) for tissue in tissues_sel}
    genes_nh_sel = set.intersection(*genes_nh.values())
    df_genes_nh = df_gtf.loc[list(genes_nh_sel)]

    # save all non-heritable genes to file
    df_genes_nh.to_csv(os.path.join(data_dir, "genes_nh_all.csv"))

    # randomly sample n genes from set of non-locally heritable genes and save to file
    np.random.seed(SEED)
    genes_nh_subset = np.random.choice(df_genes_nh.index, size=1000, replace=False)
    df_genes_nh_subset = df_genes_nh.loc[genes_nh_subset]
    df_genes_nh_subset.to_csv(os.path.join(data_dir, "genes_nh.csv"))
    
    # get clean gene expression matrices (regress out covariates and normalize)
    tissues = ["Muscle_Skeletal","Whole_Blood","Skin_Sun_Exposed_Lower_leg","Artery_Tibial","Adipose_Subcutaneous"]
    df_genes = pd.concat([df_genes_h_short, df_genes_h_long, df_genes_nh_subset], axis=0)

    for tissue in tissues:
        df_cov = pd.read_csv(os.path.join(cov_dir, f"{tissue}.v8.covariates.txt"), sep="\t", index_col=0).transpose()
        df_exp_raw = dfs_exp[tissue]

        exp_dict = {}
        for gene in tqdm(df_genes.index):
            y = df_exp_raw.loc[gene]
            X = df_cov.loc[y.index,:]
            X1 = scale(X)
            reg = LinearRegression()
            reg.fit(X1,y)
            y_res = y - reg.predict(X1)
            exp_dict[gene] = pd.Series(scale(y_res), index=y_res.index)
        df_exp = pd.concat(exp_dict, axis=1)
        df_exp.to_csv(os.path.join(data_dir, f"{tissue}.csv"))