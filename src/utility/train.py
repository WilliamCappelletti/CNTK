import torch
from tqdm.autonotebook import tqdm

def train(model, input_train, target_train, criterion, optimizer, nb_epochs=1000, batch_size=None, device=None):

    if batch_size is None:
        batch_size = input_train.shape[0]

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    criterion = criterion.to(device)

    input_train, target_train = input_train.to(device), target_train.to(device)
    # input_test, target_test = input_test.to(device), target_test.to(device)

    for e in tqdm(range(nb_epochs)):
        for input, targets in zip(input_train.split(batch_size), target_train.split(batch_size)):

            #the standart training
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
    
    return model