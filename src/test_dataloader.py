import torch
from option import args
import data

torch.manual_seed(args.seed)
loader = data.Data(args)

print(len(loader.loader_train))
