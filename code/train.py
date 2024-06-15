import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools

import torch
from torch import nn, optim, Tensor
from torch.utils.data import Dataset, DataLoader

from models import Transformer
from dataloader import EmbeddingsDataset, collate_fn

root_dir = "/home/panwei/rampr009/gtex_nt_mult"
tissues = ["Muscle_Skeletal","Whole_Blood","Skin_Sun_Exposed_Lower_leg","Artery_Tibial","Adipose_Subcutaneous"]
gtex_exp_paths = {tissue: os.path.join(root_dir, f"data/{tissue}.csv") for tissue in tissues}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--data_dir', required=True)
    parser.add_argument('-n','--num_genes', required=False)
    args = vars(parser.parse_args())

    SEED = 42
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision('high')
        
    data_dir = os.path.join(root_dir, "data", args["data_dir"])
    
    files = os.listdir(os.path.join(data_dir, "1", "train"))
    genes_all = [file.replace(".st","") for file in files]
    num_genes = int(getattr(args, "num_genes", len(genes_all)))
    genes = genes_all[:num_genes]
    
    save_dir = os.path.join(root_dir, "models", args["data_dir"], str(num_genes))
    os.makedirs(save_dir, exist_ok=True)
    
    donors_dict = {k: pd.read_csv(os.path.join(root_dir, f"data/donors_{k}.csv"), index_col=0).squeeze() for k in ["train","val"]}

    dfs_exp = {k: pd.read_csv(path, index_col=0) for k, path in gtex_exp_paths.items()}

    train_params = {
        "lr_start": 2e-4,
        "lr_end": 0.,
        "num_epochs": 100,
        "batch_size": 64,
    }

    hparams = {
        "dim": 512,
        "dim_out": len(tissues),
        "depth": 2,
        "heads": 8,
        "emb_dropout": 0.1,
        "hidden_dropout": 0.1,
        "attn_dropout": 0.1,
        "head_dropout": 0.1,
    }
    
    device = 'cuda'
    haplotypes = ["1","2"]
    criterion = nn.MSELoss()
    
    train_tuples = list(itertools.product(*[genes, donors_dict["train"]]))
    val_tuples = list(itertools.product(*[genes, donors_dict["val"]]))

    train_dataset = EmbeddingsDataset(train_tuples, data_dir, "train", dfs_exp)
    train_loader = DataLoader(train_dataset, batch_size=train_params["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=1, persistent_workers=True, pin_memory=True)

    val_dataset = EmbeddingsDataset(val_tuples, data_dir, "val", dfs_exp)
    val_loader = DataLoader(val_dataset, batch_size=train_params["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=1, persistent_workers=True, pin_memory=True)

    model = Transformer(**hparams).to(device=device, dtype=torch.bfloat16)

    optimizer = optim.AdamW(params=model.parameters(), lr=train_params["lr_start"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = train_params["num_epochs"], eta_min=train_params["lr_end"])    
    
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(train_params["num_epochs"])):
        train_loss = 0.
        val_loss = 0.

        model.train()
        for xs_dict,y in train_loader:
            optimizer.zero_grad()
            for h in xs_dict.keys():
                for k in xs_dict[h].keys():
                    if k != "max_seqlen":
                        xs_dict[h][k] = xs_dict[h][k].to(device)
            y = y.to(device)
            y_pred = model(xs_dict).squeeze()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.size(0)
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            for xs_dict,y in val_loader:
                for h in xs_dict.keys():
                    for k in xs_dict[h].keys():
                        if k != "max_seqlen":
                            xs_dict[h][k] = xs_dict[h][k].to(device)
                y = y.to(device)
                y_pred = model(xs_dict).squeeze()
                loss = criterion(y_pred, y)
                val_loss += loss.item() * y.size(0)
            val_loss /= len(val_dataset)
            val_losses.append(val_loss)

        scheduler.step()

        pd.DataFrame({
            'train': pd.Series(train_losses),
            'val': pd.Series(val_losses),
        }).to_csv(os.path.join(save_dir, "losses.csv"), index=False)
        if (epoch+1) % 10 == 0:
            torch.save(model, os.path.join(save_dir, f"model_{epoch+1}.pt"))

if __name__ == "__main__":
    main()