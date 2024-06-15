import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pandas_plink import read_plink1_bin
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import PredefinedSplit
from collections import defaultdict
from functools import reduce

import warnings
warnings.simplefilter(action='ignore')

data_dir = "/home/panwei/rampr009/gtex_nt_mult/data/"
gtex_plink_dir = os.path.join(data_dir, "plink")
tissues = ["Muscle_Skeletal","Whole_Blood","Skin_Sun_Exposed_Lower_leg","Artery_Tibial","Adipose_Subcutaneous"]
genes_files = ["genes_short.csv", "genes_long.csv", "genes_nh.csv"]

if __name__ == "__main__":
    gtex_exp_paths = {tissue: os.path.join(data_dir, f"{tissue}.csv") for tissue in tissues}
    exp_dfs = {tissue: pd.read_csv(path, index_col=0) for tissue,path in gtex_exp_paths.items()}

    donors_dict = {k: pd.read_csv(os.path.join(data_dir, f"donors_{k}.csv"), index_col=0).squeeze() for k in ["train","val","test"]}
    donors_train = pd.concat([donors_dict['train'], donors_dict['val']])
    donors_test = donors_dict['test']

    # use the training and validation split defined in donors_train and donors_val
    ps = PredefinedSplit(test_fold=[0 if ix in donors_dict['val'].index else 1 for ix in donors_train.index])

    df_genes = pd.concat([pd.read_csv(os.path.join(data_dir, file)).set_index('gene_id') for file in genes_files], axis=0)

    gtex_plink_files = os.listdir(gtex_plink_dir)
    extensions = [os.path.splitext(file) for file in gtex_plink_files]
    ext_dict = defaultdict(list)
    for k,v in extensions:
        ext_dict[k].append(v)

    genes = ext_dict.keys()

    corrs_dict = dict()
    for tissue in tissues:
        exp_df = exp_dfs[tissue]
        exp_train = exp_df.loc[donors_train]
        exp_test = exp_df.loc[donors_test]
        corrs_dict[tissue] = dict()
        for gene in tqdm(genes):
            gtex_gene_files = {file.split('.')[-1]: file for file in gtex_plink_files if f'{gene}.' in file}
            gtex_gene_files = {k: os.path.join(gtex_plink_dir,v) for k,v in gtex_gene_files.items() if k in ['bed','bim','fam']}
            snps_df = read_plink1_bin(**gtex_gene_files, verbose=False).to_pandas()
            X_train = snps_df.loc[donors_train]
            X_test = snps_df.loc[donors_test]
            y_train = exp_train[gene]
            y_test = exp_test[gene]
            model = ElasticNetCV(cv = ps)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            corrs_dict[tissue][gene] = np.corrcoef(y_pred, y_test)[0,1]

    corrs_df = pd.DataFrame(corrs_dict)
    corrs_df.loc[corrs_df.index.isin(df_genes.index)].to_csv(os.path.join(data_dir, "corrs_enet.csv"))