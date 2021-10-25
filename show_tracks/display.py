import pandas as pd
import numpy as np
from scipy import interpolate
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import sys

# 预先填充0，对应初始数据中的数据缺失值。
def fillzero():
    for file in os.listdir(os.path.join('input', 'digiMc')):
        path = os.path.join(os.path.join('input', 'digiMc'), file)
        data_names = ["run","event","trackIndex","gid","layer","cell","rawDriftTime","centerX","centerY","radius","chargeParticle","currentTrackPID","isScondary","posx","posy","posz","momx","momy","momz","driftdistance","depositeEnergy","positionFlag","flightLength","creatorProcessturnID"]
        # 读取文件，处理，并输出。这一步主要是防止出现读入时候的错误。
        df = pd.read_csv(path, index_col='gid', sep=',', names=data_names)
        df = df.iloc[1:, :]
        df.replace(np.nan, 0, True)
        df.to_csv(os.path.join(os.path.join('input', 'fillzero'), file))
        
    
# 合并，将实验数据信息与每个斜丝的坐标进行对应。
def premerge(path=None):
    # path对应实验数据的路径
    if path == None:
        data_file = 'test.csv'
    else:
        data_file = path
    # gid_file对应斜丝的位置信息路径。
    gid_file = os.path.join('input', 'MdcWirePosition_191003.csv')
    data_Dtypes = {"run":np.int64,"event":np.int64,"trackIndex":np.int16,"gid":np.uint32,\
                    "layer":np.uint8,"cell":np.uint16,"rawDriftTime":np.float32,"centerX":np.float32,\
                     "centerY":np.float32,"radius":np.float32,"chargeParticle":np.int16,"currentTrackPID":np.int16,\
                     "isScondary":np.int32,"posx":np.float32,"posy":np.float32,"posz":np.float32,"momx":np.float32,\
                     "momy":np.float32,"momz":np.float32,"driftdistance":np.float32,"depositeEnergy":np.float32,\
                     "positionFlag":np.int8,"flightLength":np.float32,"creatorProcessturnID":str}
    gid_Dtypes={'gid':np.uint16,'layer':np.uint16,'cell':np.uint16,'ex':np.float32,'ey':np.float32,'ez':np.float32,'wx':np.float32,'wy':np.float32,'wz':np.float32}
    data_names = ["gid","run","event","trackId","layer","cell","rawDriftTime","centerX","centerY","radius","chargeParticle","currentTrackPID","isScondary","posx","posy","posz","momx","momy","momz","driftdistance","depositeEnergy","positionFlag","flightLength","creatorProcessturnID"]
    gid_names = ["gid","layer","cell","ex","ey","ez","wx","wy","wz"]
    data_df = pd.read_csv(data_file, sep=',', header=0, dtype=data_Dtypes, names=data_names)
    gid_df = pd.read_csv(gid_file, sep=',', header=0, dtype=gid_Dtypes, names=gid_names)
    data_df.set_index(['gid'], inplace=True)
    gid_df.set_index(['gid'], inplace=True)
    # 根据gid这个关键之，合并每个实验数据的粒子所在位置。
    out = pd.merge(data_df, gid_df, on='gid')
    # 以腔室中心为圆心，计算极坐标系下的半径和偏移角。
    out['e_rho'] = np.sqrt(out.ex**2 + out.ey**2)
    out['e_phi'] = np.arctan2(out.ey, out.ex)
    out.to_csv(os.path.join(os.path.join('input', 'premerge'), os.path.basename(data_file)), index=None)


# 将>=1000的径迹值减去1000，并重新编码。
def dealtrack(path=None):
    if path == None:
        data_file = os.path.join('input', 'premerge.csv')
    else:
        data_file = path
    data_Dtypes={'run':np.int64,'event':np.int64,'trackId':np.int16,'gid':np.uint32,'layer':np.uint8,'cell':np.uint16,'rawTime':np.float32,'charge':np.uint64,'rawDriftTime':np.float32, 'ex':np.float32,'ey':np.float32,'ez':np.float32,'wx':np.float32,'wy':np.float32,'wz':np.float32, 'e_phi':np.float32, 'e_rho':np.float32}    
    data_df = pd.read_csv(data_file, sep=',', header=0, dtype=data_Dtypes)
    data_df['trackId'] = data_df['trackId'].apply(lambda x: x-1000 if x >= 1000 else x)
    data_df['trackId'] = data_df['trackId'].apply(lambda x: -1 if x < 0 else x)
    groups = data_df.groupby(['run', 'event'])
    dfs = []
    # run和event确定一组径迹值，重新编码。
    for key, group in groups:
        trackId = sorted(list(set(group['trackId'])))
        maps = [(trackId[i], i) for i in range(0, len(trackId))]
        maps = dict(maps)
        maps[-1] = -1
        group['trackId'] = group['trackId'].map(maps)
        # groups['trackId'] = group['trackId'].apply(lambda x: iTrack)
        dfs.append(group)
    data_df = pd.concat(dfs, axis=0)
    data_df.to_csv(os.path.join(os.path.join('input', 'pretracked'), os.path.basename(data_file)), index=None)



# 绘制图像。
def drawpic(method, n, path=None, outputdir='output'):
    # n为绘制多少张图，目前最多有189个图。超过189，等于全部绘制。
    if path == None:
        data_file = os.path.join('input', 'pretracked.csv')
    else:
        data_file = path
    data_df = pd.read_csv(data_file, header=0)
    groups = data_df.groupby(['run', 'event'])
    colorMap = {-1:'grey', 0:'brown', 1:'orange', 2:'green', 3:'red', 4:'yellow', 5:'pink'} 
    with PdfPages(os.path.join(outputdir, os.path.basename(data_file[:-4]+'.pdf'))) as pdf:
        for key, group in groups:
            fig = plt.figure(figsize=(17, 8))
            fig.suptitle('East'+str(key))
            ax1 = fig.add_subplot(121)
            plt.title('trackId x & y')
            plt.xlabel('east_x')
            plt.ylabel('east_y')
            outCircle = plt.Circle((0,0), 81, fill=False)
            innerCircle = plt.Circle((0, 0), 6, fill=False)
            ax1.add_artist(outCircle)
            ax1.add_artist(innerCircle)
            for k,v in group.groupby('trackId'):
                # 忽略噪声点的轨迹
                if k == -1:
                    continue
                minRow = v[v['flightLength']==v['flightLength'].min()].iloc[0]
                cX, cY, r = v['centerX'].iloc[0], v['centerY'].iloc[0], v['radius'].iloc[0]
                cX, cY, r = float(cX), float(cY), float(r)
                if method=='scatter':
                    tx = group[group.trackId==k].ex - cX
                    ty = group[group.trackId==k].ey - cY
                    thetas = np.arctan2(ty, tx)
                    thetas = np.array(thetas)
                    up = thetas[thetas > 0]
                    down = thetas[thetas <= 0]
                    # 上半部分图像进行散点线绘制
                    if len(up) > 0:
                        uptheta = np.arange(up.min(), up.max(), 0.001)
                        thetas = uptheta
                    # 下半部分图像进行散点线绘制
                    if len(down) > 0:
                        downtheta = np.arange(down.min(), down.max(), 0.001)
                        thetas = downtheta
                    # 补充0和pi位置的小圆缺口
                    if len(up) > 0 and len(down) > 0:
                        thetas = np.concatenate([uptheta, downtheta])
                        if len(up[up>(np.pi-0.1)]) > 0 and len(down[down<(np.pi+0.1)]) > 0:
                            thetas = np.concatenate([thetas, np.arange(up.max(), down.min()+np.pi*2, 0.001)])
                        if len(up[up<0.15]) > 0 and len(down[down>-0.15]) > 0:
                            thetas = np.concatenate([thetas, np.arange(down.max(), up.min(), 0.001)])

                    x = cX + r * np.cos(thetas)
                    y = cY + r * np.sin(thetas)
                    idx = ((x**2+y**2)>(6**2)) & ((x**2+y**2)<(81**2))
                    x = x[idx]
                    y = y[idx]
                    # 绘制散点的圆弧。
                    ax1.scatter(x,y,color=colorMap[k],s=0.3)
                elif method == 'line':
                    ax1.add_artist(plt.Circle((cX,cY), r,fill=False, color=colorMap[k]))
            # 限制方形范围。
            ax1.set_xlim((-82, 82))
            ax1.set_ylim((-82, 82))
            ax1.set_aspect(1)
            sc1 = ax1.scatter(group.ex, group.ey, s=8, c=group.trackId.map(colorMap))
            ax2 = fig.add_subplot(122)
            sc2 = ax2.scatter(group.e_rho, group.e_phi, s=8, c=group.trackId.map(colorMap))
            plt.title('trackId rho & phi')
            plt.xlabel('rho')
            plt.ylabel('phi')
            pdf.savefig()
            plt.close()
            n -= 1
            if n <= 0:
                break

if __name__ == '__main__':
    # 设置命令行参数。
    # Line method is quicker, but without track limited. 
    # Scatter method use scattered points to simulate the arc, which is slower but precisely.
    parser = argparse.ArgumentParser(description='Draw track and particles. Two ways to implement: line and scatter.')
    parser.add_argument('--method', help='--method line, or --method scatter', default='scatter')
    parser.add_argument('--n', help='--n 10, the number of pictures to be printed', default=1000)
    args = parser.parse_args()

    # 用到的路径名，如果文件夹不存在，就创建这些文件夹。
    paths = [os.path.join('input', 'fillzero'), os.path.join('input', 'premerge'), os.path.join('input', 'pretracked')]
    path_dict = {'line':os.path.join('output_line'), 'scatter':os.path.join('output_scatter')}
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)
    if not os.path.exists(path_dict[args.method]):
        os.mkdir(path_dict[args.method])

    # 填充初始数据中的缺失数据。
    fillzero()
    for file in os.listdir(os.path.join('input', 'fillzero')):
        # 预先合并个细丝对应的投影位置。
        out = premerge(os.path.join(os.path.join('input', 'fillzero'), file))
        # 合并径迹，并为径迹编码。
        out = dealtrack(os.path.join(os.path.join('input', 'premerge'), file))
        # 根据信息，绘制径迹图。
        oo = drawpic(args.method, int(args.n), os.path.join(os.path.join('input', 'pretracked'), file), path_dict[args.method])
    