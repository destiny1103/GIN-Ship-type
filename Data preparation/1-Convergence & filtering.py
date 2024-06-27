import requests
import random
import time
import pandas as pd
from bs4 import BeautifulSoup
import csv
import os

########################数据爬取#####################
'''
inputfile:原始数据文件
outfile:爬虫得到的数据文件
读取数据文件，按照MMSI爬取Vessel Name、SHIP TYPE、Flag、Length(m)、Beam (m)、Year of Built
数据输人0，可以从第一个爬取
输入上一次程序断开连接的MMSI可以继续爬取
'''


def data_get(inputfile, outfile):
    # 浏览器标识
    user_agents = [
        'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:54.0) Gecko/20100101 Firefox/54.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'
    ]
    # 请求头
    headers = {
        'authority': 'www.vesselfinder.com',
        'method': 'GET',
        # 'path': '/api/pub/weather/at/567561000',
        'scheme': 'https',
        'accept': '*/*',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
        'cookie': '_ga=GA1.1.286039417.1681912285; _gcl_au=1.1.1859430801.1681912287; __gads=ID=5ecf6240ea617f6c-222459c740df00b7:T=1681912287:RT=1681912287:S=ALNI_MZE6X3SEDN0QTmS2XiNJhC2eCoDNQ; __gpi=UID=00000bfa1610a3eb:T=1681912287:RT=1681912287:S=ALNI_MaCSiW2ccYvQrLduaGCb98qYahagQ; _ga_0MB1EVE8B7=GS1.1.1681912285.1.1.1681913590.0.0.0',
        # 'referer': 'https://www.vesselfinder.com/vessels/details/9474113',
        'sec-ch-ua': '"Chromium";v="112", "Microsoft Edge";v="112", "Not:A-Brand";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': random.choice(user_agents)}

    # 读取MMSI，IMO进行检索
    df = pd.read_csv(inputfile)
    MMSI_IMO = df[['MMSI', 'IMO']].drop_duplicates()
    All_MMSI = MMSI_IMO['MMSI'].tolist()
    All_IMO = MMSI_IMO['IMO'].tolist()
    print("MMSI和IMO的个数分别为:", len(All_MMSI), len(All_IMO))
    # 建立csv文件
    if not os.path.exists(outfile):
        with open(outfile, "w", newline='') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(
                ["MMSI", "IMO", "Vessel Name", "SHIP TYPE", "Flag", "Length(m)", "Beam (m)", "Year of Built"])

    num = input("请输入开始MMSI:")
    # 程序运行控制
    statu = False
    i = 0
    # for循环实现数据获取、解析、写入
    for MMSI, IMO in zip(All_MMSI, All_IMO):

        print(MMSI, IMO)
        i += 1
        print(f'已完成{i}/{len(All_MMSI)}')
        if IMO == "IMO0000000" or isinstance(IMO, float) or len(str(IMO[3:])) > 7:
            if str(MMSI) == num or num == "0":
                statu = True
            if statu == True:
                url = r'https://www.vesselfinder.com/vessels/details/' + str(MMSI)
                try:
                    resp = requests.get(url=url, headers=headers)
                except requests.exceptions.SSLError:
                    time.sleep(2)
                    resp = requests.get(url=url, headers=headers)
                except requests.exceptions.ProxyError:
                    time.sleep(2)
                    resp = requests.get(url=url, headers=headers)
                except requests.exceptions.ChunkedEncodingError:
                    time.sleep(2)
                    resp = requests.get(url=url, headers=headers)
                print(resp.status_code)
                ####################解析数据#####################

                page = BeautifulSoup(resp.text, "html.parser")
                # 标题处数据
                IMO = 0
                try:
                    Vessel_Name = page.find('h1', class_='title').text
                    Vessel_type = page.find('h2', class_='vst').text.split(',')[0]
                except AttributeError:
                    print(f"跳过MMSI:{MMSI}")
                    continue
                # 表格处数据
                table_aparams = page.find('table', {'class': 'aparams'})
                if table_aparams is not None:
                    Length_Beam = table_aparams.find('td', string='Length / Beam').find_next_sibling('td').text.strip()
                    Flag = table_aparams.find('td', string='Flag').find_next_sibling('td').text.strip()
                    Length, Beam, Year = -1, -1, -1
                    try:
                        Length = int(Length_Beam.split('/')[0].strip())
                    except (ValueError, IndexError):
                        pass
                    try:
                        Beam = int(Length_Beam.split('/')[1].strip().split(' ')[0])
                    except (ValueError, IndexError):
                        pass
                else:
                    MMSI, IMO, Flag, Length, Beam, Year = MMSI, -1, 'None', -1, -1, -1

                # try:
                #     tds = table.find_all("td")
                # except AttributeError:
                #     print(f"跳过的MMSI:{MMSI}")
                #     continue
                print(MMSI, IMO, Vessel_Name, Vessel_type, Flag, Length, Beam, Year)
                ####################数据写入表格#####################
                with open("IMOdetail.csv", "a", newline='') as f:
                    csvwriter = csv.writer(f)
                    # 写入一行数据，包括两个单元格
                    csvwriter.writerow([MMSI, IMO, Vessel_Name, Vessel_type, Flag, Length, Beam, Year])
                time.sleep(random.uniform(0.5, 1))

                print(i)

            else:
                continue
        else:
            if str(MMSI) == num or num == "0":
                statu = True
            if statu == True:
                url = r'https://www.vesselfinder.com/vessels/details/' + str(IMO[3:])
                try:
                    resp = requests.get(url=url, headers=headers)
                except requests.exceptions.SSLError:
                    time.sleep(2)
                    resp = requests.get(url=url, headers=headers)
                except requests.exceptions.ProxyError:
                    time.sleep(2)
                    resp = requests.get(url=url, headers=headers)
                except requests.exceptions.ChunkedEncodingError:
                    time.sleep(2)
                    resp = requests.get(url=url, headers=headers)
                print(resp.status_code)
                ####################解析数据#####################
                page = BeautifulSoup(resp.text, "html.parser")
                table = page.find("div", class_="col npr vfix-top vfix-bottom")

                try:
                    tds = table.find_all("td")
                except AttributeError:
                    print(f"跳过的MMSI:{MMSI}")
                    continue
                print(MMSI, IMO, tds[3].text, tds[5].text, tds[7].text, tds[15].text, tds[17].text, tds[21].text)
                ####################数据写入#####################
                with open("IMOdetail.csv", "a", newline='') as f:
                    csvwriter = csv.writer(f)
                    # 写入一行数据，包括两个单元格
                    csvwriter.writerow(
                        [MMSI, IMO[3:], tds[3].text, tds[5].text, tds[7].text, tds[15].text, tds[17].text,
                         tds[21].text])
                time.sleep(random.uniform(0.5, 1))

                print(i)

            else:
                continue


########################数据融合#####################
'''
getdata:爬取得到的数据文件
datafile:原始数据
outfile:合并文件
根据原始数据中的MMSI号匹配爬取数据中的'SHIP TYPE', "Length(m)", "Beam (m)"，将这些加到原始数据的右边，形成合并文件
'''


def contact_data(getdata, datafile, outfile):
    get_df = pd.read_csv(getdata).drop_duplicates(subset='MMSI')
    for chunk in pd.read_csv(datafile, chunksize=10000):
        datafile_df = chunk
        merged_df = datafile_df.merge(get_df[['MMSI', 'Ship type', "Length1", "Beam"]], on='MMSI', how='left',
                                      suffixes=('_1', '_2'))
        columns_to_replace = ['Length1', 'Beam']

        merged_df[columns_to_replace] = merged_df[columns_to_replace].replace([0, '0', '-1', '-'], float('nan'))
        merged_df["Length1"].fillna(merged_df['Length'], inplace=True)
        merged_df["Beam"].fillna(merged_df['Width'], inplace=True)
        merged_df[columns_to_replace] = merged_df[columns_to_replace].replace([0, '0', '-1', '-'], float('nan'))
        columns_to_check = ['Length1', 'Beam']  # 要检查的列的名称列表
        merged_df.dropna(subset=columns_to_check, how='any', inplace=True)
        df = merged_df.reset_index(drop=True)
        columns_to_drop = ['Heading', 'VesselName', 'IMO', 'CallSign', 'VesselType', 'Length', 'Width', 'Draft',
                           'Cargo', 'TransceiverClass']
        df.drop(columns=columns_to_drop, inplace=True)
        df.to_csv(outfile, mode='a', index=False)


####################数据裁剪######################
'''
根据
inputfile:
outfile:
LAT1:维度范围1，小
LAT2:维度范围2，大
LON1:jingdu
LON2:

'''


def cut(inputfile, outfile, LAT1, LAT2, LON1, LON2):
    df = pd.read_csv(inputfile)
    filtered_data = df[(df['LAT'] >= LAT1) & (df['LAT'] <= LAT2) & (df['LON'] >= -LON1) & (df['LON'] <= -LON2)]
    filtered_data.to_csv(outfile)


####################多个文件合并######################
'''
多个格式相同的数据文件合到一个大文件
input_folder:数据文件夹
outfile
'''


def contact_files(input_folder, outfile):
    # 获取输入文件夹中所有的CSV文件
    csv_files = [file for file in os.listdir(input_folder) if file.endswith('.csv')]
    # 使用read_csv函数读取文件的标题行（列名）
    df = pd.read_csv(os.path.join(input_folder, csv_files[0]), nrows=0)  # 仅读取0行，即标题行
    column_names = df.columns.tolist()  # 获取列名列表

    # 检查输出文件是否存在，如果不存在则先保存一个包含列名的DataFrame作为标题行
    if not os.path.exists(outfile):
        initial_df = pd.DataFrame(columns=column_names)
        initial_df.to_csv(outfile, index=False)

    # 遍历每个CSV文件
    for file in csv_files:
        # 构建输入文件的完整路径
        input_file = os.path.join(input_folder, file)
        # 使用pandas读取CSV文件
        df = pd.read_csv(input_file)
        # 将筛选后的数据添加到filtered_data中
        df.to_csv(outfile, mode='a', header=False, index=False)
        print(f"完成{file}")
    # 将筛选后的数据保存为CSV文件

    print("数据文件合并完成！")


########################数据提取#####################
'''
对原始数据按照船舶类型提取分类成各个文件夹
inputfile:读入文件
'''


def data_extract(inputfile, outpath):
    # 判断路径是否存在
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    ship_dic = {
        'Cargo_Ship': ['Bulk Carrier', 'Container Ship', 'General Cargo Ship', 'Cargo ship', 'Deck Cargo Ship',
                       'Cargo ship (HAZ-D)', 'Cargo ship (HAZ-B)',
                       'Vehicles Carrier', 'Refrigerated Cargo Ship', 'Passenger/General Cargo Ship',
                       'Self Discharging Bulk Carrier', 'Fish Carrier',
                       'Cargo ship (HAZ-B)', 'Wood Chips Carrier', 'Cargo ship (HAZ-A)',
                       'Bulk/Oil Carrier', "Livestock Carrier", 'Cement Carrier'],

        'Tanker': ['Chemical/Oil Products Tanker', 'Crude Oil Tanker', 'LPG Tanker', 'Tanker', 'LNG Tanker',
                   'Oil Products Tanker', 'Fruit Juice Tanker', 'Chemical/Oil Products Tanker', 'Chemical Tanker',
                   'Bitumen Tanker', 'Bunkering Tanker'],

        'Passenger_ship': ['Passenger/Ro-Ro Cargo Ship', 'Ro-Ro Cargo', 'Ro-Ro Cargo Ship', 'Passenger ship',
                           'Passenger Ship', 'Passenger (Cruise) Ship', 'Passenger ship (HAZ-C)',
                           'Passenger ship (HAZ-D)', 'Passenger ship (HAZ-B)'],

        'Tug': ['Tug', 'Pusher Tug', 'Towing vessel', 'Towing vessel (tow>200)', 'Offshore Tug/Supply Ship'],

        'Fishing_vessel': ['Fishing vessel', 'Fishing Vessel', 'Fish Factory Ship', 'Fishing Support Vessel'],

        'Pleasure_craft': ['Pleasure craft', 'Sailing vessel', 'Yacht']}
    df = pd.read_csv(inputfile, nrows=0)  # 仅读取0行，即标题行
    column_names = df.columns.tolist()  # 获取列名列表
    column_names.append('GNN-label')
    for ship, ship_types in ship_dic.items():
        if not os.path.exists(os.path.join(outpath, f'{ship}.csv')):
            initial_df = pd.DataFrame(columns=column_names)

            initial_df.to_csv(os.path.join(outpath, f'{ship}.csv'), index=False)
    for chunk in pd.read_csv(inputfile, chunksize=100000):
        data = chunk
        i = 1
        for ship, ship_types in ship_dic.items():
            outfile = os.path.join(outpath, f'{ship}.csv')
            filtered_data = data[data['Ship type'].isin(ship_types)]
            filtered_data = filtered_data.copy()
            filtered_data['GNN-label'] = i
            filtered_data.to_csv(outfile, mode='a', header=False, index=False)
            print(f"已完成{ship}")
            i += 1


def main(place, path):
    inputfile = fr'{path}\cut_{place}.csv'
    outfile = fr'{path}\contact_{place}.csv'
    getdata = fr'{path}\Data_rawling.csv'
    outpath = fr'{path}\{place}'
    contact_data(getdata, inputfile, outfile)
    data_extract(outfile, outpath)


if __name__ == '__main__':
    place = 'Olean'
    path = 'D:\项目\编程\python\model_practice\AIS'
    main(place, path)
