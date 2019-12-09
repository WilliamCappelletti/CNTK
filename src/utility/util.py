import torch

def one_hot(batch,depth):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ones = torch.sparse.torch.eye(depth)
    ones = ones.to(device)
    return ones.index_select(0,batch)