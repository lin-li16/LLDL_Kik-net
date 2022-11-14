import geopy.distance as dist
import os
from tqdm import tqdm


def getMessage(filename):
    msg = []
    f = open(filename, 'r')
    for line in f.readlines():
        if 'Lat.' in line and 'Station' not in line:
            line = line.split()
            Lat = float(line[-1])
            msg.append(Lat)
        if 'Long.' in line and 'Station' not in line:
            line = line.split()
            Long = float(line[-1])
            msg.append(Long)
        if 'Depth.' in line:
            line = line.split()
            Depth = float(line[-1])
            msg.append(Depth)
        if 'Mag.' in line:
            line = line.split()
            Mag = float(line[-1])
            msg.append(Mag)
        if 'Station Lat.' in line:
            line = line.split()
            SLat = float(line[-1])
            msg.append(SLat)
        if 'Station Long.' in line:
            line = line.split()
            SLong = float(line[-1])
            msg.append(SLong)
        if 'Record Time' in line:
            line = line.split()
            recdate = line[-2]
            rectime = line[-1]
            msg.append(recdate)
            msg.append(rectime)

    source = (msg[0], msg[1])
    station = (msg[4], msg[5])
    Distance = dist.geodesic(source, station).kilometers
    msg.append(Distance)
    return msg


msg = open('message.out', 'w')
msg.write('Event\tLat.\tLong.\tDepth\tMag.\tSLat.\tSLong.\tRec_data\tRec_time\tDistance\n')

filelist = os.listdir()
pbar = tqdm(filelist, desc='Get message', ncols=100)
for filedir in pbar:
    if os.path.isdir(filedir) and filedir != '.vscode':
        EWfile = os.path.join(filedir, filedir + '.EW1')
        message = getMessage(EWfile)
        msg.write('%s\t' % filedir)
        for m in message:
            if isinstance(m, float):
                msg.write('%f\t' % m)
            else:
                msg.write('%s\t' % m)
        msg.write('\n')
msg.close()