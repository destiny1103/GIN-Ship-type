import sys
from tqdm import tqdm
import pandas as pd
from geopy.distance import geodesic as GD
import math
import os

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 100)
# pd.set_option('display.width', 1000)
'''
四种船舶轨迹特征
    渔船(通常只在近岸活动)
    客船(不会转圈,路线较为笔直,)
    货船(会转圈,路线曲折)
    油船(会转圈,路线曲折,但大小与货船有区别)
'''
'''为每条船添加 轨迹属性（1.位移        2.主航向      3.时间差    4.长宽比   5.是否转圈  ，
                       9.平均速度   10.总路程     11总时间）
              船舶点属性（ 船舶状态 )   '''


def dtime(data):  # 后点与前点间隔时间
    data['BaseDateTime'] = pd.to_datetime(data['BaseDateTime'], format='%Y-%m-%d %H:%M:%S')
    dt = data['BaseDateTime'] - data['BaseDateTime'].shift(1)
    dt = dt.dt.total_seconds()
    dt = dt.copy()
    dt[0] = dt[1]
    return dt


def dlenth(data):  # 后点与前点移动位移
    data['next_LAT'] = data['LAT'].shift(1)
    data['next_LON'] = data['LON'].shift(1)
    data.loc[0, 'next_LAT'] = data.iloc[1]['next_LAT']
    data.loc[0, 'next_LON'] = data.iloc[1]['next_LON']
    gds = data.apply(lambda row: GD((row['LAT'], row['LON']), (row['next_LAT'], row['next_LON'])).m, axis=1)
    gds[0] = gds[1]
    return gds


def calculate_bearing(row):  # 求航向角（与正北方向夹角）
    lat1 = math.radians(row['next_LAT'])
    lon1 = math.radians(row['next_LON'])
    lat2 = math.radians(row['LAT'])
    lon2 = math.radians(row['LON'])
    # 计算夹角
    delta_lon = lon2 - lon1
    y = math.sin(delta_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    bearing_rad = math.atan2(y, x)
    # 转换为角度
    bearing_deg = math.degrees(bearing_rad)

    # 确保角度在0到360度之间
    return (bearing_deg + 360) % 360


def dangle(data):  # 转向角 后点航向减前点航向
    data['LAT'] = pd.to_numeric(data['LAT'])
    data['LON'] = pd.to_numeric(data['LON'])
    data['next_LAT'] = data['LAT'].shift(1)
    data['next_LON'] = data['LON'].shift(1)

    COG = data.apply(calculate_bearing, axis=1)

    da = COG.diff()
    da = (da + 180) % 360 - 180
    try:
        da[0], da[1] = da[2], da[2]
    except KeyError:
        print(da)
    da = da.abs()
    return da


def sog_lmh(data):  # SOG高中低速占比
    low_medium = 3
    medium_high = 12
    total_count = len(data)
    data['SOG'] = pd.to_numeric(data['SOG'], errors='coerce')
    count_less_than_n = len(data[data['SOG'] <= low_medium])
    count_between_n_and_m = len(data[(data['SOG'] > low_medium) & (data['SOG'] <= medium_high)])
    count_greater_than_m = len(data[data['SOG'] > medium_high])
    per_l = (count_less_than_n / total_count) * 100
    per_m = (count_between_n_and_m / total_count) * 100
    per_h = (count_greater_than_m / total_count) * 100
    return per_l, per_m, per_h


def main(group):
    group.reset_index(inplace=True,drop=True)
    '自身衍生特征'
    try:
        group['len_wids'] = float(group.iloc[0]['Length1']) / float(group.iloc[0]['Beam'])  # 长宽比
    except  ZeroDivisionError:
        print(group)
    '个体运动特征'
    dt = dtime(group)  # 前后时间差 单位：s
    dl = dlenth(group)  # 前后距离 单位：m
    da = dangle(group)  # 转向角 单位：°
    dt.name = 'dt'
    dl.name = 'dl'
    da.name = 'da'

    # group['sum_dt'] = dt.sum()  # 轨迹片段-航行时长 单位：s
    # group['sum_dl'] = dl.sum()  # 轨迹片段-航行路程 单位：m
    # group['sum_da'] = da.sum()  # 轨迹片段-累计转向角 单位：°

    # group['mean_S'] = float(group.iloc[0]['sum_dl']) / float(group.iloc[0]['sum_dt'])  # 长宽比
    #
    # group['Delta'] = GD([group.iloc[0]['LAT'], group.iloc[0]['LON']],  # 位移
    #                     [[group.iloc[-1]['LAT'], group.iloc[-1]['LON']]]).m
    # try:
    #     group['S/D'] = float(group.iloc[0]['mean_S']) / float(group.iloc[0]['Delta'])  # 距离位移比
    # except ZeroDivisionError:
    #     group['S/D'] = 400000

    # group['LS'], group['MS'], group['HS'] = sog_lmh(group)

    group = group.drop(columns=['next_LAT', 'next_LON'])
    out_df = pd.concat([group, dt, dl, da],axis=1)
    out_df['count'] = out_df.groupby(['LAT', 'LON'])['LAT'].transform('count')
    out_df['BaseDateTime'] = pd.to_datetime(out_df['BaseDateTime'], format='%Y-%m-%d %H:%M:%S')
    out_df['time'] = out_df.groupby('SegmentID')['BaseDateTime'].transform(lambda x: x - x.min()).dt.total_seconds()
    out_df = out_df.drop_duplicates(subset=['LAT', 'LON'])
    out_df = out_df.reset_index(drop=True)
    return out_df


if __name__ == '__main__':

    input_folder_path = r'E:\项目\model_practice\python\GNN\New_York_data\2_cleaned'
    outpath = r'E:\项目\model_practice\python\GNN\New_York_data\4_feature'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    file_list = os.listdir(input_folder_path)
    for filename in file_list:
        outfile = os.path.join(outpath, filename)
        if filename.endswith('.csv'):
            inputfile = os.path.join(input_folder_path, filename)
            df = pd.read_csv(inputfile)
            for SegmentID, group in tqdm(df.groupby('SegmentID'), desc="Processing groups"):
                group = group.reset_index(drop=True)
                out_df = main(group)
                if len(out_df) < 100:
                    pop_file='pop_list.txt'
                    filepath=os.path.join(outpath,pop_file)
                    file = os.path.splitext(filename)[0]
                    with open(filepath,mode='a') as f:
                        f.write(file+','+SegmentID+ '\n')
                    print(f'跳过{SegmentID}')
                    continue
                if os.path.exists(outfile):
                    out_df.to_csv(outfile, mode='a', index=False, header=False)
                else:
                    out_df.to_csv(outfile, mode='w', index=False)
        print(f'完成{filename}文件处理')
    # inputfile=r'D:\项目\编程\python\model_practice\AIS\Olean_compression_only\Tug.csv'
    # outfile=r'D:\项目\编程\python\model_practice\AIS\Olean_feature\Tug.csv'
    # df = pd.read_csv(inputfile)
    # for SegmentID, group in tqdm(df.groupby('SegmentID'), desc="Processing groups"):
    #     out_df = main(group)
    #     if os.path.exists(outfile):
    #         out_df.to_csv(outfile, mode='a', index=False, header=False)
    #     else:
    #         out_df.to_csv(outfile, mode='w', index=False)

