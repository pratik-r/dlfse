import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from safetensors import safe_open

class EmbeddingsDataset(Dataset):
    def __init__(self, tuples, data_dir, mode, dfs_exp):
        self.tuples = tuples
        self.haplotypes = ["1","2"]
        self.data_dirs = {h: os.path.join(data_dir, h, mode) for h in self.haplotypes}
        self.dfs_exp = dfs_exp
        
    def __len__(self):
        return len(self.tuples)
    
    def __getitem__(self, ix):
        gene, donor = self.tuples[ix]
        filepaths = {h: os.path.join(self.data_dirs[h], f"{gene}.st") for h in self.haplotypes}
        xs = {}
        for h in self.haplotypes:
            with safe_open(filepaths[h], framework="pt") as f:
                xs[h] = f.get_tensor(donor)
        y = torch.tensor([df.loc[donor, gene] for df in self.dfs_exp.values()])
        return xs["1"], xs["2"], y

def collate_fn(batch):
    xs_raw = {}
    xs_raw["1"], xs_raw["2"], ys = zip(*batch)
    xs_dict = {}
    for h in xs_raw.keys():
        seqlens = torch.tensor([x.size(0) for x in xs_raw[h]], dtype=torch.int32)
        max_seqlen = seqlens.max().item()
        mask = torch.tensor([[True] * x.size(0) + [False] * (max_seqlen - x.size(0)) for x in xs_raw[h]])        
        xs_dict[h] = {
            "max_seqlen": max_seqlen,
            "indices": torch.nonzero(mask.flatten(), as_tuple=False).flatten(),
            "cu_seqlens": nn.functional.pad(torch.cumsum(seqlens, dim=0, dtype=torch.torch.int32), (1, 0)),
            "x": torch.cat(xs_raw[h], dim=0),
        }
    
    y = torch.vstack(ys).to(torch.bfloat16)
    
    return xs_dict, y