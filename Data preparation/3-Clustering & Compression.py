from __future__ import division
import pandas as pd
import numpy as np
import sklearn.cluster as skc
from math import sqrt, pow
import matplotlib.pyplot as plt
import os


# 两点之间距离
def point2LineDistance(point_a, point_b, point_c):
    """-------计算点a到点b c所在直线的距离-------"""
    # 首先计算 b c 所在直线的斜率和截距
    if point_b[3] == point_c[3]:
        return 9999999
    slope = (point_b[2] - point_c[2]) / (point_b[3] - point_c[3])  # 计算斜率
    intercept = point_b[2] - slope * point_b[3]  # 计算截距
    # 计算点a到 b c所在直线的距离
    distance = abs(slope * point_a[3] - point_a[2] + intercept) / sqrt(1 + pow(slope, 2))
    return distance


# 改进dp算法
class Improved_DP_algorithm(object, ):
    def __init__(self):
        self.shape_threshold = 0.0009  # 形状阈值
        self.speed_threshold = 0.1  # 速度阈值
        self.heading_threshold = 30  # 航向阈值

        self.qualify_list = list()  # 保留点列表
        self.disqualify_list = list()  # 剩余点列表

    def method_shape_speed_heading(self, point_list):
        if len(point_list) < 2:
            self.qualify_list.extend(point_list[::-1])  # 若点少于2，则将点加入qualify_list
        else:
            # ---------1.找到与首尾两点连线距离最大的点---------#
            max_distance_index, max_distance = 0, 0
            for index, point in enumerate(point_list):
                if index in [0, len(point_list) - 1]:
                    continue
                distance = point2LineDistance(point, point_list[0], point_list[-1])
                if distance > max_distance:
                    max_distance_index = index
                    max_distance = distance
            # ---------------2.找到速度差值均值的最大值---------------#
            previous_velocity_difference = 0
            latter_velocity_difference = 0
            ave_velocity_difference = 0
            for index, point in enumerate(point_list):
                if index in [0, len(point_list) - 1]:
                    continue
                previous_velocity_difference = point[4] - point_list[index - 1][4]
                latter_velocity_difference = point_list[index + 1][4] - point[4]
                # 计算点与前一点的速度的差值
                # 计算点与后一点的速度的差值
                if abs(previous_velocity_difference) >= self.speed_threshold and abs(latter_velocity_difference) \
                        >= self.speed_threshold:
                    if (abs(previous_velocity_difference) + abs(
                            latter_velocity_difference)) / 2 > self.speed_threshold:
                        ave_velocity_difference = (abs(previous_velocity_difference) + abs(
                            latter_velocity_difference)) / 2
                        max_speed_index = index

            # ---------------3.找到航向的最大值---------------#
            heading_max = 0
            for index, point in enumerate(point_list):
                if index in [0, len(point_list) - 1]:
                    continue
                if abs(point_list[index + 1][5] - point[5]) > heading_max:
                    heading_max = abs(point_list[index + 1][5] - point[5])
                    max_heading_index = index

            # ---------若最大距离小于阈值，则去掉所有中间点。 反之，则将曲线按最大距离点分割---------#
            if max_distance > self.shape_threshold:
                # ---------1.将曲线按最大距离的点分割成两段---------#
                sequence_a = point_list[:max_distance_index]
                sequence_b = point_list[max_distance_index:]
                for sequence in [sequence_a, sequence_b]:
                    if len(sequence) < 2 and sequence == sequence_b:
                        self.qualify_list.extend(sequence[::-1])
                    else:
                        self.disqualify_list.append(sequence)
            elif ave_velocity_difference > self.speed_threshold:
                # ---------2.将曲线按最大平均速度差值的点分割成两段---------#
                sequence_a = point_list[:max_speed_index]
                sequence_b = point_list[max_speed_index:]
                for sequence in [sequence_a, sequence_b]:
                    if len(sequence) < 2 and sequence == sequence_b:
                        self.qualify_list.extend(sequence[::-1])
                    else:
                        self.disqualify_list.append(sequence)
            elif heading_max > self.heading_threshold:
                # ---------3.将曲线按最大航向角的点分割成两段---------#
                sequence_a = point_list[:max_heading_index]
                sequence_b = point_list[max_heading_index:]
                for sequence in [sequence_a, sequence_b]:
                    if len(sequence) < 2 and sequence == sequence_b:
                        self.qualify_list.extend(sequence[::-1])
                    else:
                        self.disqualify_list.append(sequence)
            else:
                # ---------都不满足, 去掉所有中间点---------#
                self.qualify_list.append(point_list[-1])
                self.qualify_list.append(point_list[0])

    def improved_dp_main(self, point_list):
        self.method_shape_speed_heading(point_list)
        while len(self.disqualify_list) > 0:
            self.method_shape_speed_heading(self.disqualify_list.pop())
        # print('压缩之后的数据长度:%s' % len(self.qualify_list))
        # print('压缩之后的数据:\n %s' %      self.qualify_list)


def clustering(trajectory, eps, min_samples):
    trajectory = trajectory.copy()
    db = skc.DBSCAN(eps=eps, min_samples=min_samples).fit(np.array(trajectory[['LON', 'LAT']]))
    labels = db.labels_
    trajectory['labels_'] = labels
    # 遍历、处理所有簇
    ship_labeled = pd.DataFrame()
    for label, group in trajectory.groupby('labels_'):
        if label == -1:
            ship_labeled = pd.concat([ship_labeled, group], axis=0)
        else:
            first_row = group.iloc[0].copy()
            first_row['LON'] = group['LON'].mean()
            first_row['LAT'] = group['LAT'].mean()
            ship_labeled = pd.concat([ship_labeled, first_row.to_frame().T], axis=0)

    # ship.to_csv('afg.csv')
    # ship_labeled.columns = trajectory.columns

    return ship_labeled


def visualization(uncompressed, compressed, index):
    fig = plt.figure()
    show_original_line_x = []
    show_original_line_y = []
    for show_point in uncompressed:
        show_original_line_x.append(show_point[3])
        show_original_line_y.append(show_point[2])

    show_compression_complete_line_x = []
    show_compression_complete_line_y = []
    for show_point in compressed:
        show_compression_complete_line_x.append(show_point[3])
        show_compression_complete_line_y.append(show_point[2])

    ax1 = fig.add_subplot(211)
    plt.plot(show_original_line_x, show_original_line_y, color='green', lw=1)  # , linestyle='--'
    plt.scatter(show_original_line_x, show_original_line_y, color='red', s=5)
    plt.title(index)
    plt.sca(ax1)

    ax2 = fig.add_subplot(212)
    plt.plot(show_compression_complete_line_x, show_compression_complete_line_y, color='green', linestyle='--')
    plt.scatter(show_compression_complete_line_x, show_compression_complete_line_y, color='red')
    plt.title(index)
    plt.sca(ax2)

    plt.show()


def main(intput_file, output_file, identifier='SegmentID'):
    df = pd.read_csv(intput_file)

    'berth: 0.0001 100/300; anchorage: 0.0015 100'
    eps, min_samples = 0.0003, 200

    for index, trajectory in df.groupby(identifier):
        if len(trajectory) > 1000:

            # print(trajectory.iloc[0])
            "-------cluster-------"
            # ship = clustering(trajectory, eps, min_samples)  # dbscan
            # del ship['labels_']
            "-------compression-------"
            dp = Improved_DP_algorithm()
            dp.improved_dp_main(trajectory.values.tolist())
            compression_complete_data = dp.qualify_list
            ship = pd.DataFrame(compression_complete_data, columns=trajectory.columns)
            "-------sort by time-------"
            ship['BaseDateTime'] = pd.to_datetime(ship['BaseDateTime'])
            ship = ship.sort_values(by='BaseDateTime')
            "-------visualization-------"
            # visualization(trajectory.values.tolist(), compression_complete_data, index)
            print('clustered:' + str(len(ship)), 'uncompressed:' + str(len(trajectory))
                  , 'compression rate:' + str(round((len(trajectory) - len(ship)) * 100 / len(trajectory), 2)) + '%')
            # print('compressed:' + str(len(compression_complete_data)), 'clustered:' + str(len(ship)), 'uncompressed:' + str(len(trajectory))
            #       , 'compression rate:' + str(round((len(trajectory) - len(compression_complete_data)) * 100 / len(trajectory), 2)) + '%')
        else:
            ship = trajectory
        '-------write-------'
        if len(ship)<150:
            continue

        if os.path.exists(output_file):
            ship.to_csv(output_file, mode='a', index=False, header=False)
        else:
            ship.to_csv(output_file, mode='w', index=False)


if __name__ == '__main__':
    input_folder_path = r'D:\项目\编程\python\model_practice\AIS\Olean_pretreatment'
    out_folder_path = r'D:\项目\编程\python\model_practice\AIS\Olean_compression_only'
    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)
    file_list = os.listdir(input_folder_path)
    for filename in file_list:
        if filename.endswith('.csv'):
            inputfile = os.path.join(input_folder_path, filename)
            outputfile = os.path.join(out_folder_path, filename)
            main(inputfile, outputfile)
        print(f'完成{filename}文件处理')
