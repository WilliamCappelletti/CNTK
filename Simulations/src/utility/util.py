import torch

def one_hot(batch,depth):
    device = batch.device
    ones = torch.sparse.torch.eye(depth)
    ones = ones.to(device)
    return ones.index_select(0,batch)