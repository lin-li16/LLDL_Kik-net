import numpy as np
import matlab.engine
import matplotlib.pyplot as plt
import scipy.io
import seaborn as sns
import os
from tqdm import tqdm
matlabeng = matlab.engine.start_matlab()   #启动matlab
sns.set_style('ticks')
sns.set_context("poster")
plt.rcParams['font.sans-serif'] = 'Times New Roman'


def time_performance(label, pred):
    MSE = np.mean((pred - label) ** 2)
    std_label = np.sqrt(np.mean((label - np.mean(label, axis=1)[:, None]) ** 2, axis=1))
    RMSE = np.mean(np.sqrt(np.mean((pred - label)**2, axis=1)) / std_label)
    MAE = np.mean(np.abs(pred - label))
    RMAE = np.mean(np.max(np.abs(pred - label), axis=1) / std_label)
    r_all = []
    for i in range(label.shape[0]):
        r_all.append(np.corrcoef(label[i, :, :].ravel(), pred[i, :, :].ravel())[0, 1])
    r = np.mean(r_all)
    return np.sqrt(MSE), RMSE, MAE, RMAE, r


def freq_performance(label, pred, dt, damp, Period):
    Sa_label_all, Sa_pred_all = [], []
    for i in range(label.shape[0]):
        Sa_label = matlabeng.getResponseSpectrum(matlab.double(label[i, :, :].ravel().tolist()), dt, matlab.double(Period.tolist()), damp)
        Sa_pred = matlabeng.getResponseSpectrum(matlab.double(pred[i, :, :].ravel().tolist()), dt, matlab.double(Period.tolist()), damp)
        Sa_label = np.array(Sa_label).ravel() / np.max(np.abs(label[i, :, :]))
        Sa_pred = np.array(Sa_pred).ravel() / np.max(np.abs(pred[i, :, :]))
        Sa_label_all.append(list(Sa_label))
        Sa_pred_all.append(list(Sa_pred))
    Sa_label_all = np.array(Sa_label_all)
    Sa_pred_all = np.array(Sa_pred_all)

    MSE = np.mean((Sa_pred_all - Sa_label_all) ** 2)
    std_label = np.sqrt(np.mean((Sa_label_all - np.mean(Sa_label_all, axis=1)[:, None]) ** 2, axis=1))
    RMSE = np.mean(np.sqrt(np.mean((Sa_pred_all - Sa_label_all)**2, axis=1)) / std_label)
    MAE = np.mean(np.abs(Sa_pred_all - Sa_label_all))
    RMAE = np.mean(np.max(np.abs(Sa_pred_all - Sa_label_all), axis=1) / std_label)
    r_all = []
    for i in range(label.shape[0]):
        r_all.append(np.corrcoef(Sa_pred_all[i, :], Sa_label_all[i, :])[0, 1])
    r = np.mean(r_all)
    return np.sqrt(MSE), np.sqrt(RMSE), MAE, RMAE, r


def plotResults(results_path, damp=0.05, Period=np.logspace(-1.5, 0.5, 300)):
    results = scipy.io.loadmat(os.path.join(results_path, 'result.mat'))
    dt = float(results['dt'].ravel()[0])

    train_r_t, train_r_f = [], []
    train_Sa_input, train_Sa_label, train_Sa_pred = [], [], []
    train_input = results['train_data']
    train_label = results['train_label']
    train_pred = results['train_pred']
    t = np.arange(dt, dt * train_input.shape[1] + dt, dt)
    for i in range(train_input.shape[0]):
        Sa_input = matlabeng.getResponseSpectrum(matlab.double(train_input[i, :, :].ravel().tolist()), dt, matlab.double(Period.tolist()), damp)
        Sa_label = matlabeng.getResponseSpectrum(matlab.double(train_label[i, :, :].ravel().tolist()), dt, matlab.double(Period.tolist()), damp)
        Sa_pred = matlabeng.getResponseSpectrum(matlab.double(train_pred[i, :, :].ravel().tolist()), dt, matlab.double(Period.tolist()), damp)
        Sa_input = np.array(Sa_input).ravel() / np.max(np.abs(train_input[i, :, :]))
        Sa_label = np.array(Sa_label).ravel() / np.max(np.abs(train_label[i, :, :]))
        Sa_pred = np.array(Sa_pred).ravel() / np.max(np.abs(train_pred[i, :, :]))
        train_Sa_input.append(list(Sa_input))
        train_Sa_label.append(list(Sa_label))
        train_Sa_pred.append(list(Sa_pred))
    train_Sa_input = np.array(train_Sa_input)
    train_Sa_label = np.array(train_Sa_label)
    train_Sa_pred = np.array(train_Sa_pred)

    std_train_label = np.sqrt(np.mean((train_label - np.mean(train_label, axis=1)[:, None]) ** 2, axis=1))
    train_MSE_t = np.sqrt(np.mean((train_pred - train_label) ** 2, axis=1)).ravel()
    train_RMSE_t = train_MSE_t / std_train_label.ravel()
    train_MAE_t = np.mean(np.abs(train_pred - train_label), axis=1).ravel()
    train_RMAE_t = np.max(np.abs(train_pred - train_label), axis=1).ravel() / std_train_label.ravel()

    std_train_Sa_label = np.sqrt(np.mean((train_Sa_label - np.mean(train_Sa_label, axis=1)[:, None]) ** 2, axis=1))
    train_MSE_f = np.sqrt(np.mean((train_Sa_pred - train_Sa_label) ** 2, axis=1)).ravel()
    train_RMSE_f = train_MSE_f / std_train_Sa_label.ravel()
    train_MAE_f = np.mean(np.abs(train_Sa_pred - train_Sa_label), axis=1).ravel()
    train_RMAE_f = np.max(np.abs(train_Sa_pred - train_Sa_label), axis=1).ravel() / std_train_Sa_label.ravel()
    for i in range(train_input.shape[0]):
        train_r_t.append(np.corrcoef(train_pred[i, :, :].ravel(), train_label[i, :, :].ravel())[0, 1])
        train_r_f.append(np.corrcoef(train_Sa_pred[i, :], train_Sa_label[i, :])[0, 1])
    train_r_t = np.array(train_r_t)
    train_r_f = np.array(train_r_f)

    pbar = tqdm(range(0, train_input.shape[0]), desc='Plotting', ncols=100)
    for i in pbar:
        PGA = np.max(np.abs(train_label[i, :, :]))
        plt.figure(figsize=(8, 12))
        plt.subplots_adjust(hspace=0.3, wspace=0.2)
        ax1 = plt.subplot2grid((4, 1), (0, 0))
        ax1.plot(t, train_input[i, :, :], 'k', linewidth=1, label='input')
        ax1.set_ylabel('acc (gal)')
        # ax1.legend(loc='upper right')
        ax2 = plt.subplot2grid((4, 1), (1, 0))
        ax2.plot(t, train_label[i, :, :], linewidth=1, label='label')
        ax2.plot(t, train_pred[i, :, :], '--', linewidth=1, label='predict')
        ax2.set_xlabel('t (s)')
        ax2.set_ylabel('acc (gal)')
        # ax2.legend(loc='upper right')
        ax3 = plt.subplot2grid((4, 1), (2, 0), rowspan=2)
        ax3.semilogx(Period, train_Sa_input[i, :], 'k', label='input')
        ax3.semilogx(Period, train_Sa_label[i, :], label='label')
        ax3.semilogx(Period, train_Sa_pred[i, :], label='predict')
        ax3.set_xlabel('T (s)')
        ax3.set_ylabel('$\\beta$')
        ax3.legend(loc='upper right')
        plt.savefig(os.path.join(results_path, 'figures', 'train%d_PGA%d_et%.3f_rt%.1f_ef%.3f_rf%.1f.svg' % (i, PGA, train_MSE_t[i] / PGA, 100 * train_r_t[i], train_MSE_f[i], 100 * train_r_f[i])), bbox_inches='tight')
    
    valid_r_t, valid_r_f = [], []
    valid_Sa_input, valid_Sa_label, valid_Sa_pred = [], [], []
    valid_input = results['valid_data']
    valid_label = results['valid_label']
    valid_pred = results['valid_pred']
    t = np.arange(dt, dt * valid_input.shape[1] + dt, dt)
    for i in range(valid_input.shape[0]):
        Sa_input = matlabeng.getResponseSpectrum(matlab.double(valid_input[i, :, :].ravel().tolist()), dt, matlab.double(Period.tolist()), damp)
        Sa_label = matlabeng.getResponseSpectrum(matlab.double(valid_label[i, :, :].ravel().tolist()), dt, matlab.double(Period.tolist()), damp)
        Sa_pred = matlabeng.getResponseSpectrum(matlab.double(valid_pred[i, :, :].ravel().tolist()), dt, matlab.double(Period.tolist()), damp)
        Sa_input = np.array(Sa_input).ravel() / np.max(np.abs(valid_input[i, :, :]))
        Sa_label = np.array(Sa_label).ravel() / np.max(np.abs(valid_label[i, :, :]))
        Sa_pred = np.array(Sa_pred).ravel() / np.max(np.abs(valid_pred[i, :, :]))
        valid_Sa_input.append(list(Sa_input))
        valid_Sa_label.append(list(Sa_label))
        valid_Sa_pred.append(list(Sa_pred))
    valid_Sa_input = np.array(valid_Sa_input)
    valid_Sa_label = np.array(valid_Sa_label)
    valid_Sa_pred = np.array(valid_Sa_pred)

    std_valid_label = np.sqrt(np.mean((valid_label - np.mean(valid_label, axis=1)[:, None]) ** 2, axis=1))
    valid_MSE_t = np.sqrt(np.mean((valid_pred - valid_label) ** 2, axis=1)).ravel()
    valid_RMSE_t = valid_MSE_t / std_valid_label.ravel()
    valid_MAE_t = np.mean(np.abs(valid_pred - valid_label), axis=1).ravel()
    valid_RMAE_t = np.max(np.abs(valid_pred - valid_label), axis=1).ravel() / std_valid_label.ravel()

    std_valid_Sa_label = np.sqrt(np.mean((valid_Sa_label - np.mean(valid_Sa_label, axis=1)[:, None]) ** 2, axis=1))
    valid_MSE_f = np.sqrt(np.mean((valid_Sa_pred - valid_Sa_label) ** 2, axis=1)).ravel()
    valid_RMSE_f = valid_MSE_f / std_valid_Sa_label.ravel()
    valid_MAE_f = np.mean(np.abs(valid_Sa_pred - valid_Sa_label), axis=1).ravel()
    valid_RMAE_f = np.max(np.abs(valid_Sa_pred - valid_Sa_label), axis=1).ravel() / std_valid_Sa_label.ravel()
    for i in range(valid_input.shape[0]):
        valid_r_t.append(np.corrcoef(valid_pred[i, :, :].ravel(), valid_label[i, :, :].ravel())[0, 1])
        valid_r_f.append(np.corrcoef(valid_Sa_pred[i, :], valid_Sa_label[i, :])[0, 1])
    valid_r_t = np.array(valid_r_t)
    valid_r_f = np.array(valid_r_f)

    pbar = tqdm(range(0, valid_input.shape[0]), desc='Plotting', ncols=100)
    for i in pbar:
        PGA = np.max(np.abs(valid_label[i, :, :]))
        plt.figure(figsize=(8, 12))
        plt.subplots_adjust(hspace=0.3, wspace=0.2)
        ax1 = plt.subplot2grid((4, 1), (0, 0))
        ax1.plot(t, valid_input[i, :, :], 'k', linewidth=1, label='input')
        ax1.set_ylabel('acc (gal)')
        # ax1.legend(loc='upper right')
        ax2 = plt.subplot2grid((4, 1), (1, 0))
        ax2.plot(t, valid_label[i, :, :], linewidth=1, label='label')
        ax2.plot(t, valid_pred[i, :, :], '--', linewidth=1, label='predict')
        ax2.set_xlabel('t (s)')
        ax2.set_ylabel('acc (gal)')
        # ax2.legend(loc='upper right')
        ax3 = plt.subplot2grid((4, 1), (2, 0), rowspan=2)
        ax3.semilogx(Period, valid_Sa_input[i, :], 'k', label='input')
        ax3.semilogx(Period, valid_Sa_label[i, :], label='label')
        ax3.semilogx(Period, valid_Sa_pred[i, :], label='predict')
        ax3.set_xlabel('T (s)')
        ax3.set_ylabel('$\\beta$')
        ax3.legend(loc='upper right')
        plt.savefig(os.path.join(results_path, 'figures', 'valid%d_PGA%d_et%.3f_rt%.1f_ef%.3f_rf%.1f.svg' % (i, PGA, valid_MSE_t[i] / PGA, 100 * valid_r_t[i], valid_MSE_f[i], 100 * valid_r_f[i])), bbox_inches='tight')

    test_r_t, test_r_f = [], []
    test_Sa_input, test_Sa_label, test_Sa_pred = [], [], []
    test_input = results['test_data']
    test_label = results['test_label']
    test_pred = results['test_pred']
    t = np.arange(dt, dt * test_input.shape[1] + dt, dt)
    for i in range(test_input.shape[0]):
        Sa_input = matlabeng.getResponseSpectrum(matlab.double(test_input[i, :, :].ravel().tolist()), dt, matlab.double(Period.tolist()), damp)
        Sa_label = matlabeng.getResponseSpectrum(matlab.double(test_label[i, :, :].ravel().tolist()), dt, matlab.double(Period.tolist()), damp)
        Sa_pred = matlabeng.getResponseSpectrum(matlab.double(test_pred[i, :, :].ravel().tolist()), dt, matlab.double(Period.tolist()), damp)
        Sa_input = np.array(Sa_input).ravel() / np.max(np.abs(test_input[i, :, :]))
        Sa_label = np.array(Sa_label).ravel() / np.max(np.abs(test_label[i, :, :]))
        Sa_pred = np.array(Sa_pred).ravel() / np.max(np.abs(test_pred[i, :, :]))
        test_Sa_input.append(list(Sa_input))
        test_Sa_label.append(list(Sa_label))
        test_Sa_pred.append(list(Sa_pred))
    test_Sa_input = np.array(test_Sa_input)
    test_Sa_label = np.array(test_Sa_label)
    test_Sa_pred = np.array(test_Sa_pred)

    std_test_label = np.sqrt(np.mean((test_label - np.mean(test_label, axis=1)[:, None]) ** 2, axis=1))
    test_MSE_t = np.sqrt(np.mean((test_pred - test_label) ** 2, axis=1)).ravel()
    test_RMSE_t = test_MSE_t / std_test_label.ravel()
    test_MAE_t = np.mean(np.abs(test_pred - test_label), axis=1).ravel()
    test_RMAE_t = np.max(np.abs(test_pred - test_label), axis=1).ravel() / std_test_label.ravel()

    std_test_Sa_label = np.sqrt(np.mean((test_Sa_label - np.mean(test_Sa_label, axis=1)[:, None]) ** 2, axis=1))
    test_MSE_f = np.sqrt(np.mean((test_Sa_pred - test_Sa_label) ** 2, axis=1)).ravel()
    test_RMSE_f = test_MSE_f / std_test_Sa_label.ravel()
    test_MAE_f = np.mean(np.abs(test_Sa_pred - test_Sa_label), axis=1).ravel()
    test_RMAE_f = np.max(np.abs(test_Sa_pred - test_Sa_label), axis=1).ravel() / std_test_Sa_label.ravel()
    for i in range(test_input.shape[0]):
        test_r_t.append(np.corrcoef(test_pred[i, :, :].ravel(), test_label[i, :, :].ravel())[0, 1])
        test_r_f.append(np.corrcoef(test_Sa_pred[i, :], test_Sa_label[i, :])[0, 1])
    test_r_t = np.array(test_r_t)
    test_r_f = np.array(test_r_f)

    pbar = tqdm(range(0, test_input.shape[0]), desc='Plotting', ncols=100)
    for i in pbar:
        PGA = np.max(np.abs(test_label[i, :, :]))
        plt.figure(figsize=(8, 12))
        plt.subplots_adjust(hspace=0.3, wspace=0.2)
        ax1 = plt.subplot2grid((4, 1), (0, 0))
        ax1.plot(t, test_input[i, :, :], 'k', linewidth=1, label='input')
        ax1.set_ylabel('acc (gal)')
        # ax1.legend(loc='upper right')
        ax2 = plt.subplot2grid((4, 1), (1, 0))
        ax2.plot(t, test_label[i, :, :], linewidth=1, label='label')
        ax2.plot(t, test_pred[i, :, :], '--', linewidth=1, label='predict')
        ax2.set_xlabel('t (s)')
        ax2.set_ylabel('acc (gal)')
        # ax2.legend(loc='upper right')
        ax3 = plt.subplot2grid((4, 1), (2, 0), rowspan=2)
        ax3.semilogx(Period, test_Sa_input[i, :], 'k', label='input')
        ax3.semilogx(Period, test_Sa_label[i, :], label='label')
        ax3.semilogx(Period, test_Sa_pred[i, :], label='predict')
        ax3.set_xlabel('T (s)')
        ax3.set_ylabel('$\\beta$')
        ax3.legend(loc='upper right')
        plt.savefig(os.path.join(results_path, 'figures', 'test%d_PGA%d_et%.3f_rt%.1f_ef%.3f_rf%.1f.svg' % (i, PGA, test_MSE_t[i] / PGA, 100 * test_r_t[i], test_MSE_f[i], 100 * test_r_f[i])), bbox_inches='tight')

    scipy.io.savemat(os.path.join(results_path, 'performance.mat'),
                    {'train-MSE-t': train_MSE_t, 'train-RMSE-t': train_RMSE_t, 'train-MAE-t': train_MAE_t, 'train-RMAE': train_RMAE_t, 'train-r-t': train_r_t, 'valid-MSE-t': valid_MSE_t, 'valid-RMSE-t': valid_RMSE_t, 'valid-MAE-t': valid_MAE_t, 'valid-RMAE': valid_RMAE_t, 'valid-r-t': valid_r_t, 'test-MSE-t': test_MSE_t, 'test-RMSE-t': test_RMSE_t, 'test-MAE-t': test_MAE_t, 'test-RMAE': test_RMAE_t, 'test-r-t': test_r_t, 'train-MSE-f': train_MSE_f, 'train-RMSE-f': train_RMSE_f, 'train-MAE-f': train_MAE_f, 'train-RMAE': train_RMAE_f, 'train-r-f': train_r_f, 'valid-MSE-f': valid_MSE_f, 'valid-RMSE-f': valid_RMSE_f, 'valid-MAE-f': valid_MAE_f, 'valid-RMAE': valid_RMAE_f, 'valid-r-f': valid_r_f, 'test-MSE-f': test_MSE_f, 'test-RMSE-f': test_RMSE_f, 'test-MAE-f': test_MAE_f, 'test-RMAE': test_RMAE_f, 'test-r-f': test_r_f})