import torch

def nn_accuracy_score(output, y_test):
    with torch.no_grad():
        return (output.argmax(dim=1) == y_test).sum() / float(len(y_test))

def print_score(model, criterion, input_train, target_train, input_test, target_test):
    with torch.no_grad():
        this_CELoss_tr = criterion(model(input_train), target_train).item()
        this_CELoss_te = criterion(model(input_test), target_test).item()
        print('{} on TRAIN :\t'.format(criterion), this_CELoss_tr,'\n\
{} on TEST :\t'.format(criterion), this_CELoss_te)