# # 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
import torch.nn as nn
import warnings
import time
import os
import sys
import scipy.io
import argparse
from tqdm import tqdm
from solver import train, test
from plot import plot_loss
from eventDataset import eqkDataset
from net import *
from get_performance import *
warnings.filterwarnings("ignore")
sns.set_style('ticks')
sns.set_context("poster")
plt.rcParams['font.sans-serif'] = 'Times New Roman'


class Logger(object):
    '''
    log文件记录对象，将所有print信息记录在log文件中
    '''
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass


def main():
    # 添加命令行输入参数
    parser = argparse.ArgumentParser(description='LSTM Model for Time Series Forecasting in KiK-Net Downhole Array Dataset')
    parser.add_argument('--path', type=str, default='IBRH13', help='Parent file path of the dataset and the results')
    parser.add_argument('--batch', type=int, default=64, help='Batch size of training data')
    parser.add_argument('--validratio', type=float, default=0.1, help='Ratio of validation data in all data, 0-1.0')
    parser.add_argument('--testratio', type=float, default=0.1, help='Ratio of test data in all data, 0-1.0')
    parser.add_argument('--fixedorder', type=int, default=1, help='Whether to use the former data order')
    parser.add_argument('--epochs', type=int, default=1000, help='Maximum training epochs')
    parser.add_argument('--printfreq', type=int, default=-1, help='Training message print frequency in each epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model', type=str, default='LSTM', help='Type of model used in this dataset')
    parser.add_argument('--hiddensize', type=int, default=32, help='Hidden size in LSTM layers')
    parser.add_argument('--numlayers', type=int, default=3, help='Number of layers in LSTM or CNN')
    parser.add_argument('--kernel', type=int, default=101, help='Kernel size used in CNN layers')
    args = parser.parse_args()

    # 创建结果文件夹
    if args.model == 'CNN':
        rstpath = '%s_ker%d_layers%d_batch%d_lr%.4f' % (args.model, args.kernel, args.numlayers, args.batch, args.lr)
    elif args.model == 'LSTM' or args.model == 'RNN':
        rstpath = '%s_hidden%d_layers%d_batch%d_lr%.4f' % (args.model, args.hiddensize, args.numlayers, args.batch, args.lr)
    elif args.model == 'MLP':
        rstpath = '%s_batch%d_lr%.4f' % (args.model, args.batch, args.lr)
    elif args.model == 'FC':
        rstpath = '%s_batch%d_lr%.4f' % (args.model, args.batch, args.lr)
    results_path = os.path.join(args.path, rstpath)
    if not os.path.exists(results_path):
        os.mkdir(results_path)
        os.mkdir(os.path.join(results_path, 'figures'))
    sys.stdout = Logger(os.path.join(results_path, 'message.log'))      # 创建log文件对象
    print('The path of the results is %s' % results_path)

    # # 数据预处理
    # ## 导入数据
    station = args.path
    print('Load data from %s' % station)
    dhacc = 981 * np.load(os.path.join(station, 'dhacc.npy'))
    upacc = 981 * np.load(os.path.join(station, 'upacc.npy'))
    dhacc = dhacc.astype(np.float32)
    upacc = upacc.astype(np.float32)
    dt = 0.02
    t = np.linspace(dt, dt * dhacc.shape[1], dhacc.shape[1])

    # ## 数据标准化
    # 用井下地震动的PGA进行数据标准化
    PGA_dh = np.max(np.abs(dhacc), axis=1)
    for i in range(dhacc.shape[0]):
        dhacc[i, :] = dhacc[i, :] / PGA_dh[i]
        upacc[i, :] = upacc[i, :] / PGA_dh[i]


    # ## 构造训练集、验证集和测试集
    batch_size = args.batch
    valid_size = args.validratio
    test_size = args.testratio
    numdata = dhacc.shape[0]

    if args.fixedorder:
        print('The training data is fixed!')
        data = scipy.io.loadmat(os.path.join(station, 'idx.mat'))
        index = list(range(numdata))
        train_idx = data['train_idx'].ravel().tolist()
        test_idx = data['test_idx'].ravel().tolist()
        valid_idx = list(set(index) - set(train_idx) - set(test_idx))
        print('Train data size: %d, Valid data size: %d, test data size: %d, batch size: %d' % (len(train_idx), len(valid_idx), len(test_idx), batch_size))
    else:
        num_valid = int(valid_size * numdata)
        num_test = int(test_size * numdata)
        num_train = numdata - num_valid - num_test
        index = list(range(numdata))
        np.random.shuffle(index)
        train_idx, valid_idx, test_idx = index[:num_train], index[num_train : num_train + num_valid], index[num_train + num_valid:]
        print('Train data size: %d, Valid data size: %d, test data size: %d, batch size: %d' % (num_train, num_valid, num_test, batch_size))

    train_data, train_label = dhacc[train_idx, :, None], upacc[train_idx, :, None]
    valid_data, valid_label = dhacc[valid_idx, :, None], upacc[valid_idx, :, None]
    test_data, test_label = dhacc[test_idx, :, None], upacc[test_idx, :, None]

    train_dataset = eqkDataset(train_data, train_label)
    valid_dataset = eqkDataset(valid_data, valid_label)
    test_dataset = eqkDataset(test_data, test_label)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset)


    # ## 进行训练
    max_epoch = args.epochs
    disp_freq = args.printfreq
    learning_rate = args.lr
    print('%s model is applied' % args.model)
    print('Learning rate is %f' % learning_rate)
    if args.model == 'LSTM':
        Net = LSTM_basic(train_data.shape[-1], args.hiddensize, args.numlayers)
    elif args.model == 'CNN':
        Net = CNN_basic(args.kernel, args.numlayers)
    elif args.model =='RNN':
        Net = RNN_basic(train_data.shape[-1], args.hiddensize, args.numlayers)
    elif args.model == 'MLP':
        Net = MLP_basic(train_data.shape[1])
    elif args.model == 'FC':
        Net = FC_basic()

    # GPU加速
    if torch.cuda.is_available():
        Net = Net.cuda()

    optimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    starttime = time.time()
    train_best_model ,valid_best_model, last_model, train_loss, valid_loss = train(Net, criterion, optimizer, train_loader, valid_loader, max_epoch, disp_freq)
    train_time = time.time()-starttime
    print('Training Time {:.4f}'.format(train_time))
    validation = test(valid_best_model, criterion, valid_loader)
    prediction = test(valid_best_model, criterion, test_loader)
    torch.cuda.empty_cache()


    # ## 绘制loss变化曲线
    plot_loss(train_loss, valid_loss, yscale='log')
    plt.savefig(os.path.join(results_path, 'loss.svg'), bbox_inches='tight')
    print('Training best epoch: %d\tTraining minimum loss: %.3E' % (np.argmin(train_loss) + 1, np.min(train_loss)))
    print('Validate best epoch: %d\tValidate minimum loss: %.3E' % (np.argmin(valid_loss) + 1, np.min(valid_loss)))
    torch.save(train_best_model, os.path.join(results_path, 'trainbest.pt'))
    torch.save(valid_best_model, os.path.join(results_path, 'validbest.pt'))
    torch.save(last_model, os.path.join(results_path, 'last.pt'))


    # 结果处理
    ## 训练集上的结果
    ### 训练集结果计算
    train_pred = np.zeros(train_label.shape)
    pbar = tqdm(range(train_data.shape[0]), desc='Calculating', ncols=100)
    for i in pbar:
        with torch.no_grad():
            y = valid_best_model(torch.tensor(train_data[i:i+1, :, :]).cuda())
            train_pred[i, :, :] = y.cpu().detach().numpy()
            train_data[i, :, 0] = train_data[i, :, 0] * PGA_dh[train_idx[i]]
            train_label[i, :, 0] = train_label[i, :, 0] * PGA_dh[train_idx[i]]
            train_pred[i, :, 0] = train_pred[i, :, 0] * PGA_dh[train_idx[i]]

    MSE_t, RMSE_t, MAE_t, RMAE_t, r_t = time_performance(train_label, train_pred)
    MSE_f, RMSE_f, MAE_f, RMAE_f, r_f = freq_performance(train_label, train_pred, dt=0.02, damp=0.05, Period=np.linspace(0.05, 2, 100))
    train_performance_t = [MSE_t, RMSE_t, MAE_t, RMAE_t, r_t]
    train_performance_f = [MSE_f, RMSE_f, MAE_f, RMAE_f, r_f]
    print('Time domain performance: MSE: %.3E, RMSE: %.3E, MAE: %.3E, RMAE: %.3E, r: %.1f' % (MSE_t, RMSE_t, MAE_t, RMAE_t, 100 * r_t))
    print('Freq domain performance: MSE: %.3E, RMSE: %.3E, MAE: %.3E, RMAE: %.3E, r: %.1f' % (MSE_f, RMSE_f, MAE_f, RMAE_f, 100 * r_f))


    ## 验证集上的结果
    with torch.no_grad():
        valid_pred = valid_best_model(torch.tensor(valid_data).cuda())
        valid_pred = valid_pred.cpu().detach().numpy()
    for i in range(valid_data.shape[0]):
        valid_data[i, :, 0] = valid_data[i, :, 0] * PGA_dh[valid_idx[i]]
        valid_label[i, :, 0] = valid_label[i, :, 0] * PGA_dh[valid_idx[i]]
        valid_pred[i, :, 0] = valid_pred[i, :, 0] * PGA_dh[valid_idx[i]]
    torch.cuda.empty_cache()


    MSE_t, RMSE_t, MAE_t, RMAE_t, r_t = time_performance(valid_label, valid_pred)
    MSE_f, RMSE_f, MAE_f, RMAE_f, r_f = freq_performance(valid_label, valid_pred, dt=0.02, damp=0.05, Period=np.linspace(0.05, 2, 100))
    valid_performance_t = [MSE_t, RMSE_t, MAE_t, RMAE_t, r_t]
    valid_performance_f = [MSE_f, RMSE_f, MAE_f, RMAE_f, r_f]
    print('Time domain performance: MSE: %.3E, RMSE: %.3E, MAE: %.3E, RMAE: %.3E, r: %.1f' % (MSE_t, RMSE_t, MAE_t, RMAE_t, 100 * r_t))
    print('Freq domain performance: MSE: %.3E, RMSE: %.3E, MAE: %.3E, RMAE: %.3E, r: %.1f' % (MSE_f, RMSE_f, MAE_f, RMAE_f, 100 * r_f))

    ## 测试集上的结果
    ### 测试集结果计算
    with torch.no_grad():
        test_pred = valid_best_model(torch.tensor(test_data).cuda())
        test_pred = test_pred.cpu().detach().numpy()
    for i in range(test_data.shape[0]):
        test_data[i, :, 0] = test_data[i, :, 0] * PGA_dh[test_idx[i]]
        test_label[i, :, 0] = test_label[i, :, 0] * PGA_dh[test_idx[i]]
        test_pred[i, :, 0] = test_pred[i, :, 0] * PGA_dh[test_idx[i]]
    torch.cuda.empty_cache()


    MSE_t, RMSE_t, MAE_t, RMAE_t, r_t = time_performance(test_label, test_pred)
    MSE_f, RMSE_f, MAE_f, RMAE_f, r_f = freq_performance(test_label, test_pred, dt=0.02, damp=0.05, Period=np.linspace(0.05, 2, 100))
    test_performance_t = [MSE_t, RMSE_t, MAE_t, RMAE_t, r_t]
    test_performance_f = [MSE_f, RMSE_f, MAE_f, RMAE_f, r_f]
    print('Time domain performance: MSE: %.3E, RMSE: %.3E, MAE: %.3E, RMAE: %.3E, r: %.1f' % (MSE_t, RMSE_t, MAE_t, RMAE_t, 100 * r_t))
    print('Freq domain performance: MSE: %.3E, RMSE: %.3E, MAE: %.3E, RMAE: %.3E, r: %.1f' % (MSE_f, RMSE_f, MAE_f, RMAE_f, 100 * r_f))


    ## 输出performance数据
    perfile = open(os.path.join(results_path, 'performance.out'), 'w')
    datatype = ['Train', 'Valid', 'Test']
    time_freq = ['t', 'f']
    pertype = ['MSE', 'RMSE', 'MAE', 'RMAE', 'r']
    allperformance = [[train_performance_t, valid_performance_t, test_performance_t], [train_performance_f, valid_performance_f, test_performance_f]]
    perfile.write('训练总次数:\t\t%d\n' % args.epochs)
    perfile.write('训练总时间:\t\t%.2f\n' % train_time)
    perfile.write('训练最好次数:\t\t%d\n' % np.argmin(train_loss))
    perfile.write('验证最好次数:\t\t%d\n' % np.argmin(valid_loss))
    for i, tf in enumerate(time_freq):
        for j, dset in enumerate(datatype):
            for k, per in enumerate(pertype):
                if per == 'r':
                    perfile.write(dset + '-' + per + '-' + tf + ':\t\t%.1f\n' % (100 * allperformance[i][j][k]))
                else:
                    perfile.write(dset + '-' + per + '-' + tf + ':\t\t%.3E\n' % allperformance[i][j][k])
    perfile.close()

    ## 保存结果数据
    scipy.io.savemat(os.path.join(results_path, 'result.mat'),
                    {'train_data': train_data, 'train_idx': train_idx, 'valid_idx': valid_idx, 'test_idx': test_idx, 'train_label': train_label, 'train_pred': train_pred, 'test_data': test_data, 'test_label': test_label, 'test_pred': test_pred, 'valid_data': valid_data, 'valid_label': valid_label, 'valid_pred': valid_pred, 'train_loss': train_loss, 'valid_loss': valid_loss, 'train_performance_t': train_performance_t, 'train_performance_f': train_performance_f, 'valid_performance_t': valid_performance_t, 'valid_performance_f': valid_performance_f, 'test_performance_t': test_performance_t, 'test_performance_f': test_performance_f, 'time': t, 'dt': dt})

    # plotResults(results_path)


if __name__ == "__main__":
    main()