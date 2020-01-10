import torch
from tqdm.autonotebook import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def train(model, input_train, target_train, criterion, optimizer, nb_epochs=1000, batch_size=None, device=None, **kwargs):

    plot_evolution = kwargs.get('plot_evolution', None)
    score_evolution = []

    if batch_size is None:
        batch_size = input_train.shape[0]

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    criterion = criterion.to(device)

    input_train, target_train = input_train.to(device), target_train.to(device)

    for e in tqdm(range(nb_epochs)):
        for input, targets in zip(input_train.split(batch_size), target_train.split(batch_size)):

            #the standart training
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

        if plot_evolution:
            with torch.no_grad():
                scorer = kwargs.get('scorer', criterion)
                score_evolution.append(scorer(output, targets))

    scorer = kwargs.get('scorer', criterion)
    if scorer:
        print("Train score :", scorer(output, targets))
    
    if plot_evolution:
        fig, ax = plt.subplots(figsize=(8,5), dpi=120)
        ax.plot(list(range(nb_epochs)), score_evolution)
        ax.set(title='Score evolution during training',
               xlabel='nb of epochs',
               ylim=(0,3.4))
        plt.tight_layout()
        plt.savefig(plot_evolution, bbox='tight')

    model = model.to('cpu')
    criterion = criterion.to('cpu')
    input_train, target_train = input_train.to('cpu'), target_train.to('cpu')
    return model