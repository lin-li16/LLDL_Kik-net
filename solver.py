import copy
import numpy as np
import torch
import time

def train(model, criterion, optimizer, train_dataset, valid_dataset, max_epoch, disp_freq):
    avg_train_loss = 0
    avg_val_loss = 0
    avg_train_loss_set = []
    avg_val_loss_set = []
    train_start = np.inf
    valid_start = np.inf

    for epoch in range(max_epoch):
        starttime = time.time()
        batch_train_loss = train_one_epoch(model, criterion, optimizer, train_dataset, max_epoch, disp_freq, epoch)
        batch_val_loss = validate(model, criterion, valid_dataset)
        avg_train_loss += batch_train_loss
        avg_val_loss += batch_val_loss
        avg_train_loss_set.append(batch_train_loss.cpu().detach().numpy())
        avg_val_loss_set.append(batch_val_loss.cpu().detach().numpy())
        if batch_train_loss < train_start:
            train_start = batch_train_loss
            train_best_model = copy.deepcopy(model)
        if batch_val_loss < valid_start:
            valid_start = batch_val_loss
            valid_best_model = copy.deepcopy(model)
        print('Epoch [{}/{}]\t Average training and validation loss: {:.4E} {:.4E}\tTime: {:.2f}s'.format(epoch + 1, max_epoch, batch_train_loss, batch_val_loss, time.time() - starttime))
    return train_best_model, valid_best_model, model, avg_train_loss_set, avg_val_loss_set


def train_one_epoch(model, criterion, optimizer, dataset, max_epoch, disp_freq, epoch):
    model.train()
    train_loss = 0
    iteration = 0
    for data, target in dataset:
        iteration += 1
        # GPU加速
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        y = model(data)
        loss = criterion(y, target)
        # loss = loss.to(torch.double)
        loss.backward()
        optimizer.step()
        train_loss += loss

        if iteration % disp_freq == 0 and disp_freq > 0:
            print("Epoch [{}][{}]\t Batch [{}][{}]\t Training Loss {:.4f}".format(
                epoch + 1, max_epoch, iteration, len(dataset),
                train_loss / iteration))
    return train_loss / len(dataset)


def validate(model, criterion, dataset):
    model.eval()
    val_loss = 0
    for data, target in dataset:
        # GPU加速
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        with torch.no_grad():
            y = model(data)
        loss = criterion(y, target)
        val_loss += loss
    return val_loss / len(dataset)


def test(model, criterion, dataset):
    print('Testing...')
    test_loss = 0
    model.eval()
    prediction = []
    for data, target in dataset:
        # GPU加速
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        y = model(data)
        prediction.append(y.cpu().detach().numpy())
        loss = criterion(y, target)
        test_loss += loss

    print("Test Loss {:.4f}\n".format(test_loss / len(dataset)))
    prediction = torch.tensor(prediction)
    return prediction, test_loss / len(dataset)