import shutil
import tarfile
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def getacc(filename):
    file = open(filename, 'r')
    acc = file.read()
    file.close()
    acc = acc.split()
    acc = [float(a) for a in acc]
    acc = np.array(acc).ravel()
    return acc


def getvel_dsp(acc, dt):
    t = np.linspace(dt, dt * acc.shape[0], acc.shape[0])
    acc1 = acc[:-1]
    acc1 = np.insert(acc1, 0, 0)
    accAvg = (acc1 + acc) / 2
    vel = np.cumsum(accAvg) * dt
    vel1 = vel[:-1]
    vel1 = np.insert(vel1, 0, 0)
    velAvg = (vel1 + vel) / 2
    dsp = np.cumsum(velAvg) * dt
    return vel, dsp


def baselineCorrection(acc, dt, M):
    t = np.linspace(dt, dt * len(acc), len(acc))
    acc1 = acc
    vel, dsp = getvel_dsp(acc, dt)
    Gv = np.zeros(shape=(acc.shape[0], M + 1))
    for i in range(M + 1):
        Gv[:, i] = t ** (M + 1 - i)
    polyv = np.dot(np.dot(np.linalg.inv(Gv.transpose().dot(Gv)), Gv.transpose()), vel)
    for i in range(M + 1):
        acc1 -= (M + 1 - i) * polyv[i] * t ** (M - i)
        
    acc_new = acc1
    vel1, dsp1 = getvel_dsp(acc1, dt)
    Gd = np.zeros(shape=(acc.shape[0], M + 1))
    for i in range(M + 1):
        Gd[:, i] = t ** (M + 2 - i)
    polyd = np.dot(np.dot(np.linalg.inv(Gd.transpose().dot(Gd)), Gd.transpose()), dsp1)
    for i in range(M + 1):
        acc_new -= (M + 2 - i) * (M + 1 - i) * polyd[i] * t ** (M - i)
    return acc_new


def cvt2col(filename):
    f = open(filename, 'r')
    for i in range(10):
        line = f.readline()
    
    line = f.readline()
    freq = line.split()[-1]
    freq = int(freq[:-2])
    dt = 1 / freq
    for i in range(2):
        line = f.readline()

    line = f.readline()
    line = line.split()[-1]
    line = line.split('/')
    scft = float(line[0][:-5]) / float(line[1])
    line = f.readline()
    maxa = float(line.split()[-1])
    for i in range(2):
        line = f.readline()

    acc = f.read()
    acc = acc.split()
    acc = [scft * float(a) for a in acc]
    acc = np.array(acc)
    acc = acc - np.mean(acc)
    acc = acc / 981
    f.close()

    if filename.endswith('.EW1'):
        id1 = 'EW_dh'
    elif filename.endswith('.EW2'):
        id1 = 'EW_up'
    elif filename.endswith('.NS1'):
        id1 = 'NS_dh'
    elif filename.endswith('.NS2'):
        id1 = 'NS_up'
    elif filename.endswith('.UD1'):
        id1 = 'UD_dh'
    elif filename.endswith('.UD2'):
        id1 = 'UD_up'
    else:
        id1 = 'other'
    
    if abs(dt - 0.01) < 1e-5:
        id2 = '_010.acc'
    elif abs(dt - 0.005) < 1e-5:
        id2 = '_005.acc'
    else:
        id2 = '_dt.acc'
    
    cvtfilename = filename[:-4] + id1 + id2
    f = open(cvtfilename, 'w')
    for a in acc:
        f.write('{:7.6E}\n'.format(a))

    f.close()
    return acc, dt


def getazimuth(station, source):
    '''
    station: 测站的经纬度[纬度, 经度]
    source: 震源的经纬度[纬度, 经度]
    '''
    direc = np.zeros(2)
    direc[0] = source[0] - station[0]
    direc[1] = (source[1] - station[1]) * np.cos(source[0] * np.pi / 180)
    direc = direc / np.linalg.norm(direc)
    return direc

def getSH(filename, station, source, dt):
    direc = getazimuth(station, source)
    if np.abs(dt - 0.005) < 1e-5:
        dh_EW = getacc(filename + 'EW_dh_005.acc')
        dh_NS = getacc(filename + 'NS_dh_005.acc')
        dh_file = open(filename + '_dh_005.acc', 'w')
        up_EW = getacc(filename + 'EW_up_005.acc')
        up_NS = getacc(filename + 'NS_up_005.acc')
        up_file = open(filename + '_up_005.acc', 'w')
    elif np.abs(dt - 0.01) < 1e-5:
        dh_EW = getacc(filename + 'EW_dh_010.acc')
        dh_NS = getacc(filename + 'NS_dh_010.acc')
        dh_file = open(filename + '_dh_010.acc', 'w')
        up_EW = getacc(filename + 'EW_up_010.acc')
        up_NS = getacc(filename + 'NS_up_010.acc')
        up_file = open(filename + '_up_010.acc', 'w')
    else:
        dh_EW = getacc(filename + 'EW_dh_dt.acc')
        dh_NS = getacc(filename + 'NS_dh_dt.acc')
        dh_file = open(filename + '_dh_dt.acc', 'w')
        up_EW = getacc(filename + 'EW_up_dt.acc')
        up_NS = getacc(filename + 'NS_up_dt.acc')
        up_file = open(filename + '_up_dt.acc', 'w')

    dh_acc = np.zeros_like(dh_EW)
    for i in range(len(dh_acc)):
        dh_acc[i] = dh_EW[i] * direc[0] - dh_NS[i] * direc[1]

    for a in dh_acc:
        dh_file.write('{:7.6E}\n'.format(a))
    dh_file.close()

    up_acc = np.zeros_like(up_EW)
    for i in range(len(dh_acc)):
        up_acc[i] = up_EW[i] * direc[0] - up_NS[i] * direc[1]

    for a in up_acc:
        up_file.write('{:7.6E}\n'.format(a))
    up_file.close()



# # 解压文件
# flist = os.listdir()
# for file in flist:
#     if file.endswith('.tar.gz'):
#         name = file.split('.')[0]
#         t = tarfile.open(file)
#         t.extractall(name)
#         t.close()

# # 将KiK-net格式的文件转换为列
# label = ['.EW1', '.EW2', '.NS1', '.NS2', '.UD1', '.UD2']
# flist = os.listdir()
# pbar = tqdm(flist, desc='转换Kik-net格式文件', ncols=100)
# for filedir in pbar:
#     if os.path.isdir(filedir):
#         for f in os.listdir(filedir):
#             if f[-4:] in label:
#                 cvt2col(os.path.join(filedir, f))

# # 旋转EW和NS分量
# print()
# filelist = os.listdir()
# pbar = tqdm(filelist, desc='旋转EW和NS分量', ncols=100)
# for filedir in pbar:
#     if os.path.isdir(filedir) and filedir != '.vscode':
#         EWfile = os.path.join(filedir, filedir + '.EW1')
#         sor = []
#         sta = []
#         f = open(EWfile, 'r')
#         for line in f.readlines():
#             if 'Lat' in line and 'Station' not in line:
#                 sor.append(float(line.split()[-1]))
#             if 'Long' in line and 'Station' not in line:
#                 sor.append(float(line.split()[-1]))
#             if 'Station Lat' in line:
#                 sta.append(float(line.split()[-1]))
#             if 'Station Long' in line:
#                 sta.append(float(line.split()[-1]))
#             if 'Sampling Freq' in line:
#                 freq = line.split()[-1]
#                 dt = 1 / float(freq[:-2])
#                 break
#         f.close()
#         shfile = os.path.join(filedir, filedir)
#         getSH(shfile, sta, sor, dt)

# # 绘图
# print()
# filelist = os.listdir()
# pbar = tqdm(filelist, desc='绘图', ncols=100)
# for filedir in pbar:
#     if os.path.isdir(filedir) and filedir != '.vscode':
#         flist = os.listdir(filedir)
#         for f in flist:
#             if f.endswith('.acc') and 'filter' not in f:
#                 name = f.split('.')[0]
#                 name = name.split('_')
#                 if name[-1] == '005':
#                     dt = 0.005
#                 elif name[-1] == '010':
#                     dt = 0.01
#                 else:
#                     dt = 10
#                 if name[0][-2:] == 'EW':
#                     if name[1] == 'dh':
#                         EW_dh = getacc(os.path.join(filedir, f))
#                     else:
#                         EW_up = getacc(os.path.join(filedir, f))
#                 elif name[0][-2:] == 'NS':
#                     if name[1] == 'dh':
#                         NS_dh = getacc(os.path.join(filedir, f))
#                     else:
#                         NS_up = getacc(os.path.join(filedir, f))
#                 elif name[0][-2:] == 'UD':
#                     if name[1] == 'dh':
#                         UD_dh = getacc(os.path.join(filedir, f))
#                     else:
#                         UD_up = getacc(os.path.join(filedir, f))
#                 else:
#                     if name[1] == 'dh':
#                         dh = getacc(os.path.join(filedir, f))
#                     else:
#                         up = getacc(os.path.join(filedir, f))

#         T = np.linspace(dt, dt * len(EW_dh), len(EW_dh))
#         plt.plot(T, EW_up, T, EW_dh, linewidth=0.5)
#         plt.xlabel('t(s)')
#         plt.ylabel('acc(g)')
#         plt.legend(['surface', 'downhole'])
#         plt.savefig(os.path.join(filedir, filedir + 'EW.png'), dpi=300)
#         plt.close()

#         plt.plot(T, NS_up, T, NS_dh, linewidth=0.5)
#         plt.xlabel('t(s)')
#         plt.ylabel('acc(g)')
#         plt.legend(['surface', 'downhole'])
#         plt.savefig(os.path.join(filedir, filedir + 'NS.png'), dpi=300)
#         plt.close()

#         plt.plot(T, UD_up, T, UD_dh, linewidth=0.5)
#         plt.xlabel('t(s)')
#         plt.ylabel('acc(g)')
#         plt.legend(['surface', 'downhole'])
#         plt.savefig(os.path.join(filedir, filedir + 'UD.png'), dpi=300)
#         plt.close()

#         plt.plot(T, up, T, dh, linewidth=0.5)
#         plt.xlabel('t(s)')
#         plt.ylabel('acc(g)')
#         plt.legend(['surface', 'downhole'])
#         plt.savefig(os.path.join(filedir, filedir + '.png'), dpi=300)
#         plt.close()