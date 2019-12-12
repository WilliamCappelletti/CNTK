import torch

def nn_accuracy_score(output, y_test):
    with torch.no_grad():
        return (output.argmax(dim=1) == y_test).sum() / float(len(y_test))

def print_score(model, criterion, input_train, target_train, input_test, target_test, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    try:
        criterion = criterion.to(device)
    except AttributeError:
        pass

    input_test, input_train = input_test.to(device), input_train.to(device)
    target_test, target_train = target_test.to(device), target_train.to(device)

    this_CELoss_tr = criterion(model(input_train), target_train).item()
    this_CELoss_te = criterion(model(input_test), target_test).item()
    print('{} on TRAIN :\t'.format(criterion), this_CELoss_tr,'\n\
{} on TEST :\t'.format(criterion), this_CELoss_te)