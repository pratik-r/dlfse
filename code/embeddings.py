import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from safetensors.torch import save_file

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM

bcftools = "/home/panwei/rampr009/install/bcftools-1.9/bcftools"
samtools = "/home/panwei/rampr009/install/samtools-1.9/samtools"

vcf_path = "/home/panwei/shared/GTEx_v8/GTEx_Analysis_v8_WholeGenomeSeq/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.SHAPEIT2_phased.vcf.gz"
ref_genome_path = "/home/panwei/rampr009/data/Homo_sapiens_assembly38.fasta"
data_dir = "/home/panwei/rampr009/gtex_nt_mult/data/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g','--genes_file', required=True)
    parser.add_argument('-s','--save_dir', required=True)
    parser.add_argument('-i','--ix', required=True)
    args = vars(parser.parse_args())
    
    genes_path = os.path.join(data_dir, args["genes_file"])
    save_dir = os.path.join(data_dir, args["save_dir"])
    ix = int(args["ix"])
    
    device = "cuda"
    
    df_genes = pd.read_csv(genes_path).set_index('gene_id')

    num_splits = 20
    ixs_splits = np.array_split(np.arange(len(df_genes)), num_splits)
    window = 2000
    layer = 22
    
    haplotypes = ["1", "2"]
    modes = ["train", "test", "val"]
    dirs_dict = {h: {mode: os.path.join(save_dir, h, mode) for mode in modes} for h in haplotypes}
    for h in haplotypes:
        for mode in modes:
            os.makedirs(dirs_dict[h][mode], exist_ok=True)

    donors = pd.read_csv(os.path.join(data_dir, "donors.csv"), index_col=0).squeeze()
    donors_dict = {mode: pd.read_csv(os.path.join(data_dir, f"donors_{mode}.csv"), index_col=0).squeeze() for mode in modes}

    tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", trust_remote_code=True).to(device)
    
    for j in tqdm(ixs_splits[ix]):
        gene = df_genes.iloc[j]
        if f"{gene.name}.st" in os.listdir(dirs_dict["1"]["train"]):
            continue
        embed_dict = {}
        for donor in donors:
            embed_dict[donor] = {}
            for h in haplotypes:
                conseq_command = f"""{samtools} faidx {ref_genome_path} {gene.chr}:{gene.start-window}-{gene.start+window} | {bcftools} consensus -s {donor} -H {h} {vcf_path}"""
                conseq_raw = os.popen(conseq_command).read().split('\n')
                conseq = ["".join(conseq_raw[1:])]
                token_ids = tokenizer.batch_encode_plus(conseq, return_tensors="pt")["input_ids"].to(device)
                model.eval()
                with torch.no_grad():
                    model_out = model(input_ids = token_ids, output_hidden_states=True)
                embeddings = model_out["hidden_states"][layer].squeeze()
                embed_dict[donor][h] = embeddings.to(torch.bfloat16)

        for h in haplotypes:
            for mode in modes:
                save_file({k: embed_dict[k][h] for k in donors_dict[mode].values}, os.path.join(dirs_dict[h][mode], f"{gene.name}.st"))