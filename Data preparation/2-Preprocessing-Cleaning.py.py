import os
import pandas as pd
from geopy.distance import geodesic as GD
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

####################数据分组######################
'''
group:按照MMSI进行分组的数据

'''


def data_groups(df):

    df_groups = df.groupby(pd.Grouper(key='BaseDateTime', freq='D'))
    i=1
    df_all = pd.DataFrame()
    for index,group in df_groups:
        if group.empty:
            continue
        df=group
        df = df.sort_values('BaseDateTime').reset_index(drop=True)
        df['time'] = df['BaseDateTime'].diff(periods=1)
        df['time'] = df['time'].dt.total_seconds()
        df.loc[0, 'time'] = df['time'].min()

        threshold = 3600
        df_list = [group for _, group in df.groupby((df['time'] > threshold).cumsum())]
        for df in df_list:
            df['SegmentID'] = str(int(group.iloc[0]['MMSI'])) + '-' + str(i)
            df_all = pd.concat([df_all, df], ignore_index=True)
            i += 1
        df_all.drop('time', axis=1, inplace=True)
    return df_all


def plot(line_segments):
    # 创建Matplotlib图形
    plt.figure(figsize=(10, 6))
    # 循环绘制每个线段，以点连线的形式
    for segment in line_segments:
        plt.plot(segment['LON'], segment['LAT'], marker='o', linestyle='-')
        # 添加标题和标签
        plt.title('Line Segments Based on Time Difference with Point-to-Point Lines')
        plt.xlabel('LON')
        plt.ylabel('LAT')
        # 显示图形
        plt.show()


####################数据剔除按照数据量######################
'''
group:MMSI分组
'''


def data_culling_bycounts(group):
    df = group
    mmsi_counts = df['SegmentID'].value_counts()
    # print(mmsi_counts)
    mmsi_greater_than_ = mmsi_counts[mmsi_counts > 150].index
    df_all = pd.DataFrame()
    for mmsi in mmsi_greater_than_:
        # 根据MMSI值筛选数据
        filtered_data = df[df['SegmentID'] == mmsi]
        df_all = pd.concat([df_all, filtered_data], ignore_index=True)
    return df_all


####################数据剔除按照面积######################
'''
剔除面积过小得数据
group:传入的每组的数据，group是指轨迹
'''


def data_culling_byarea(group):
    df = group[1]
    mmsi = group[0]
    latmax_value = df['LAT'].max()
    latmin_value = df['LAT'].min()
    lonmax_value = df['LON'].max()
    lommin_value = df['LON'].min()
    area = (latmax_value - latmin_value) * (lonmax_value - lommin_value)
    return mmsi, area


####################数据剔除按照低速占比######################
'''
剔除占比过大的数据
group:传入的每组的数据，group是指轨迹
'''


def data_culling_bysogl(group):
    tracks = group.groupby('SegmentID')
    df_all = pd.DataFrame()
    for index,track in tracks:
        if len(track[track['SOG'] > 2]) / len(track) < 0.3:
            df = pd.DataFrame()
        else:
            df = track
        df_all = pd.concat([df_all, df], ignore_index=True)
    return df_all


####################数据清洗######################
'''
按照加速度来剔除异常数据
'''
# def cleaning(filename):
#     df=pd.read_csv(filename)
#     df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
#     df=df.drop_duplicates(subset=['MMSI', 'BaseDateTime'])
#     df.sort_values(by='BaseDateTime', inplace=False)
#
#     groups = df.groupby('MMSI')
#     size_list=groups.size()
#
#
#     for group,size in zip(groups,size_list):
#         print(size)
#         # print(group[1].index[0])
#         # print(group[1].index[0], group[1])
#         # print(df.iloc[group[1].index[0]+1])
#
#         for i in range(1,size-1):
#
#             c=group[1].index
#             print(c[i])
#             b = df[['LON', 'LAT']]
#             gd1 = GD([b.iloc[c[i-1]]['LAT'], b.iloc[c[i-1]]['LON']],[[b.iloc[c[i]]['LAT'], b.iloc[c[i]]['LON']]]).kilometers*1000
#             gd2 = GD([b.iloc[c[i]]['LAT'], b.iloc[c[i]]['LON']], [[b.iloc[c[i+1]]['LAT'], b.iloc[c[i+1]]['LON']]]).kilometers*1000
#             print(gd1)
#             print(gd2)
#
#             t1=(group[1]['BaseDateTime'][c[i]]-group[1]['BaseDateTime'][c[i-1]]).total_seconds()
#             t2 = (group[1]['BaseDateTime'][c[i+1]] - group[1]['BaseDateTime'][c[i]]).total_seconds()
#             print(t1)
#             print(t2)
#             try:
#                 v1=gd1/t1
#                 v2=gd2/t2
#             except ZeroDivisionError:
#                 print(group[1]['BaseDateTime'][c[i-1]],group[1]['BaseDateTime'][c[i]],group[1]['BaseDateTime'][c[i+1]])
#             try:
#                 a=(v2-v1)/(t2/2+t1/2)
#             except ZeroDivisionError:
#                 print('报错了！！！')
#             print(a)
#             if a>15:
#                 df.drop(index=c[i])
#                 print(a,f'剔除数据{c[i]}')


########################数据按照速度异常值清洗####################
'''
group:可以是MMSI分组，也可以是每个轨迹段
'''


def cleaning_sog(group):
    df = group.copy()
    # df['SOG'] = df['SOG'] * 1.852 * 1000 / 3600
    #
    # threshold_value = df['SOG'].describe(percentiles=[0.1, 0.99983])['99.98%']
    # print(df['SOG'].describe(percentiles=[0.1, 0.99983]))
    threshold_value=35
    for index in df.index:
        if df.iloc[index]['SOG'] > threshold_value:
            group.drop(index=index)
            df.drop(index=index)
            print(f'删除第{index + 1}行')

    group = group.reset_index(drop=True)
    return group


####################按照数据距离清洗数据##################
'''
group:是指每个轨迹段
'''


def cleaning_lenth(group):
    df = group.reset_index(drop=True)
    gds = []
    gds.append(0)
    for i in range(len(group)):
        if i != len(df) - 1:
            gd = GD([group.iloc[i]['LAT'], group.iloc[i]['LON']],
                    [[group.iloc[i + 1]['LAT'], group.iloc[i + 1]['LON']]]).km
            gds.append(gd)
        else:
            break

    threshold_value = 3
    index = 0
    i = 0
    while index < len(gds):
        gd = gds[index]
        if gd > threshold_value and index < len(gds) - 1 and gds[index + 1] > threshold_value:
            df.drop(index=i, inplace=True)
            gds.pop(index)
        elif gd > threshold_value and index == len(gds) - 1:
            df.drop(index=i, inplace=True)
            gds.pop(index)
        else:
            index += 1
        i += 1

    df = df.reset_index(drop=True)
    return df

def main(inputfile,outpath):
    df=pd.read_csv(inputfile)
    df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
    df.sort_values(by='BaseDateTime', inplace=True)
    groups = df.groupby('MMSI')
    filename = os.path.basename(inputfile)
    for index,group in groups:
        mmsitracks=data_groups(group)

        mmsitracks=data_culling_bycounts(mmsitracks)

        try:
            mmsitracks = data_culling_bysogl(mmsitracks)
        except KeyError:
            print(f'跳过{int(index)}')
            continue
        try:
            tracks=mmsitracks.groupby('SegmentID')
        except KeyError:
            print(f'跳过{int(index)}')
            continue
        df_all = pd.DataFrame()
        for id,track in tracks:
            track=cleaning_lenth(track)
            track=cleaning_sog(track)
            df_all = pd.concat([df_all, track], ignore_index=True)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        outfile = os.path.join(outpath, filename)
        if os.path.exists(outfile):
            mmsitracks.to_csv(outfile, mode='a', index=False, header=False)
        else:
            mmsitracks.to_csv(outfile, mode='w', index=False)
        print(f'完成{int(index)}')






if __name__ == '__main__':
    input_folder_path = r'E:\项目\model_practice\python\GNN\New_York_data\1_data_contact'
    outpath = r'E:\项目\model_practice\python\GNN\New_York_data\2_cleaned'


    file_list = os.listdir(input_folder_path)
    for filename in file_list:
        if filename.endswith('.csv'):
            inputfile = os.path.join(input_folder_path, filename)
            main(inputfile, outpath)
        print(f'完成{filename}文件处理')


