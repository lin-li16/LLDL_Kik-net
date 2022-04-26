import matplotlib.pyplot as plt
import numpy as np


def plot_loss(train_loss, valid_loss):
    fig = plt.figure(figsize=(8, 6))
    maxEpoch = len(train_loss)
    maxLoss = 1.1 * float(max(max(train_loss), max(valid_loss)))
    minLoss = max(0, 0.9 * float(min(min(train_loss), min(valid_loss))))

    plt.plot(range(1, 1 + maxEpoch), train_loss, label='train', marker='o', markevery=int(maxEpoch / 10))
    plt.plot(range(1, 1 + maxEpoch), valid_loss, label='valid', marker='s', markevery=int(maxEpoch / 10))

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.xticks(range(0, maxEpoch + 1, int(maxEpoch / 5)))
    plt.axis([0, maxEpoch, minLoss, maxLoss])
    # plt.yscale('log')
    plt.show()


# def plot_loss(loss_dict):
#     fig = plt.figure()
#     tmp = list(loss_dict.values())
#     maxEpoch = len(tmp[0])
#     stride = np.ceil(maxEpoch / 10)

#     maxLoss = float(max(tmp[0])) + 0.1
#     minLoss = max(0, float(min(tmp[0])) - 0.1)

#     for name, loss in loss_dict.items():
#         plt.plot(range(1, 1 + maxEpoch), loss, '-s', label=name)

#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.xticks(range(0, maxEpoch + 1, 2))
#     plt.axis([0, maxEpoch, minLoss, maxLoss])
#     plt.show()


if __name__ == '__main__':
    loss = [x for x in range(10, 0, -1)]
    acc = [x / 10. for x in range(0, 10)]
    plot_loss({'as': [loss, acc]})
