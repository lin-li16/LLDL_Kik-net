import argparse
import numpy as np
import time
import subprocess
import math
import matlab.engine
import sys
import os
import shutil
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool   #并行计算的库


def change_analysis_time(tclfile, numstep, dt):
    '''
    功能：此函数用于修改analysis.tcl（OpenSees输入文件中分析部分）中的分析时间步数

    --Input
    -tclfile: string, tcl文件路径
    -numstep: int, 分析时间步数
    -dt: float, 分析时间步长

    --Return
    无
    '''
    analysisfile = open(tclfile, 'r')
    content = analysisfile.read()
    analysisfile.close()
    content = content.split('\n')
    analysisfile = open(tclfile, 'w')
    for line in content:
        if 'set numstep' in line:
            analysisfile.write('set numstep %d\n' % numstep)
        elif 'set dt' in line:
            analysisfile.write('set dt %.3f\n' % dt)
        else:
            analysisfile.write(line)
            analysisfile.write('\n')
    analysisfile.close()


def getacc(filename):
    '''
    功能：此函数可从加速度时间序列文件中获取数据，转换成numpy格式的一维数组

    --Input
    -filename: string, 加速度时间序列的文件名，文件格式为一列数据

    --Return
    -acc: numpy.array(npts), 加速度时间序列数组，npts为数据个数
    '''
    file = open(filename, 'r')
    acc = file.read()
    file.close()
    acc = acc.split()
    acc = [float(a) for a in acc]
    acc = np.array(acc).ravel()
    return acc


def runOpenSees(filepath):
    '''
    功能：此函数用于控制OpenSees进行场地响应计算

    --Input
    -tclfile:string, 输入tcl文件名

    --Return
    无
    '''
    command = 'source main.tcl'
    # s = subprocess.Popen(os.path.join(filepath, 'OpenSees.exe'), shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    s = subprocess.Popen('OpenSees.exe', shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=filepath)
    s.stdin.write(command.encode())
    s.communicate()
    s.terminate()


initialtime = time.time()
stationlist = []
alllist = os.listdir()
for f in alllist:
    if len(f) == 6:
        stationlist.append(f)

station = 'IBRH13'
tclpath = station + '_tclfile_Kalman'
tclfile = os.listdir(tclpath)
path1 = station + '_data_Kalman'
motiondir = os.listdir(path1)
pbar = tqdm(motiondir, desc=station, ncols=100)
mainfilelist = []
for motion in pbar:
    motionpath = os.path.join(path1, motion)
    if os.path.isdir(motionpath):
        for tcl in tclfile:
            shutil.copyfile(os.path.join(tclpath, tcl), os.path.join(motionpath, tcl))
        accdir = os.listdir(motionpath)
        for accfile in accdir:
            if motion + 'EW_dh' in accfile:
                dt = accfile[-7 : -4]
                if dt == '005':
                    dt = 0.005
                else:
                    dt = 0.01
                acc = getacc(os.path.join(motionpath, accfile))
                numstep = len(acc)
                change_analysis_time(os.path.join(motionpath, 'analysis.tcl'), numstep, dt)
                break
        mainfilelist.append(motionpath)
start = time.time()
print('OpenSees Running ...')
# 并行运行OpenSees
pool = ThreadPool(8)
pool.map(runOpenSees, mainfilelist)
pool.close()
pool.join()
print('OpenSees Running Time: {tt:.2f}'.format(tt=time.time() - start))


print('All Running Time: {tt:.2f}'.format(tt=time.time() - initialtime))


# # 删除OpenSees.exe文件
# # stationlist = os.listdir()
# for station in stationlist:
#     if os.path.isdir(station):
#         path1 = os.path.join(station, station + '_small')
#         motiondir = os.listdir(path1)
#         pbar = tqdm(motiondir, desc=station, ncols=100)
#         mainfilelist = []
#         for motion in pbar:
#             motionpath = os.path.join(path1, motion)
#             if os.path.isdir(motionpath):
#                 if os.path.exists(os.path.join(motionpath, 'OpenSees.exe')):
#                     os.remove(os.path.join(motionpath, 'OpenSees.exe'))


# 删除多余文件
stationlist = ['IBRH13']
for station in stationlist:
    path1 = os.path.join(station + '_data_Kalman')
    motiondir = os.listdir(path1)
    pbar = tqdm(motiondir, desc=station, ncols=100)
    mainfilelist = []
    for motion in pbar:
        motionpath = os.path.join(path1, motion)
        if os.path.isdir(motionpath):
            filelist = os.listdir(motionpath)
            for f in filelist:
                if not (f.endswith('.acc') or f.endswith('.out') or f.endswith('.fig')):
                    fpath = os.path.join(motionpath, f)
                    os.remove(fpath)
                # else:
                #     if 'EW_' in f or 'NS_' in f or 'UD_' in f:
                #         fpath = os.path.join(motionpath, f)
                #         os.remove(fpath)



# matlabeng = matlab.engine.start_matlab()   #启动matlab
# ll = matlabeng.TRYLL()