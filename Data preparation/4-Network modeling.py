import sys
import scipy
import numpy as np
import pandas as pd
from tqdm import tqdm
import igraph as ig
from igraph import plot
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from geopy.distance import geodesic as GD  # 由经纬度计算距离
# from itertools import chain
import itertools
import networkx as nx
import os
import plotly.graph_objs as go

##########################deluanay#################################
def network_deluanay(MMSI_id, group, node_index_map, current_index, SubGraph_number):
    g = ig.Graph()
    processing_ship = group[['LAT', 'LON', 'GNN-label']].to_numpy()

    'Delaunay三角剖分'
    try:
        delaunay = Delaunay(processing_ship[:, :2], qhull_options="QJ0.003")
        # 进行你的操作...
    except scipy.spatial._qhull.QhullError as qhull_error:
        with open('error.txt',mode='a') as f:
            f.write(MMSI_id+'\n')




    '添加节点并映射节点索引'
    for i in range(1, len(group) + 1):
        node_id = str(current_index)
        attributes = {'MMSI': MMSI_id, 'LAT': processing_ship[i - 1][0], 'LON': processing_ship[i - 1][1],
                      'SubGraphnumber': SubGraph_number, 'GNN-label': processing_ship[i - 1][2]}
        g.add_vertex(node_id, **attributes)

        node_index_map[(MMSI_id, i)] = current_index
        current_index += 1

    '添加边'
    edge_ = 0

    for tri_index in delaunay.simplices:
        tri_edges = itertools.combinations(tri_index, 2)
        for edge in tri_edges:
            # 使用映射后的节点索引添加边
            source_node = edge[0]
            target_node = edge[1]

            gd = GD([g.vs[source_node]['LAT'], g.vs[source_node]['LON']],
                    [[g.vs[target_node]['LAT'], g.vs[target_node]['LON']]]).kilometers


            # print(gd.kilometers)
            edge_attrs = {'SubGraphnumber': SubGraph_number, 'GNN-label': g.vs[source_node]['GNN-label']
                , 'distance': gd}
            g.add_edge(source_node, target_node, **edge_attrs)
            edge_ += 1

    g.simplify(combine_edges="first")
    '删除长边'
    threshold = 3
    edges_to_delete = [edge.index for edge in g.es.select(lambda edge: edge['distance'] > threshold)]
    g.delete_edges(edges_to_delete)

    '可视化当前网络'
    # visual_graph(g)

    # isolated_vertices = [v.index for v in g.vs.select(_degree=0)]
    # # 打印孤立的点
    # if len(isolated_vertices) > 0:
    #     print("孤立的点：", isolated_vertices)

    return g


################################创建网络结构######################
def network_MST(MMSI_id, group, node_index_map, current_index, SubGraph_number):
    g = ig.Graph()
    processing_ship = group[['LAT', 'LON', 'GNN-label']].to_numpy()

    'Delaunay三角剖分'
    try:
        delaunay = Delaunay(processing_ship[:, :2], qhull_options="QJ0.003")
        # 进行你的操作...
    except scipy.spatial._qhull.QhullError as qhull_error:
        print(MMSI_id)




    '添加节点并映射节点索引'
    for i in range(1, len(group) + 1):
        node_id = str(current_index)
        attributes = {'MMSI': MMSI_id, 'LAT': processing_ship[i - 1][0], 'LON': processing_ship[i - 1][1],
                      'SubGraphnumber': SubGraph_number, 'GNN-label': processing_ship[i - 1][2]}
        g.add_vertex(node_id, **attributes)

        node_index_map[(MMSI_id, i)] = current_index
        current_index += 1

    '添加边'
    edge_ = 0
    for tri_index in delaunay.simplices:
        tri_edges = itertools.combinations(tri_index, 2)
        for edge in tri_edges:
            # 使用映射后的节点索引添加边
            source_node = edge[0]
            target_node = edge[1]

            gd = GD([g.vs[source_node]['LAT'], g.vs[source_node]['LON']],
                    [[g.vs[target_node]['LAT'], g.vs[target_node]['LON']]]).kilometers

            # print(gd.kilometers)
            edge_attrs = {'SubGraphnumber': SubGraph_number, 'GNN-label': g.vs[source_node]['GNN-label']
                , 'distance': gd}
            g.add_edge(source_node, target_node, **edge_attrs)
            edge_ += 1

    g.simplify(combine_edges="first")

    '删除长边'
    threshold = 3
    edges_to_delete = [edge.index for edge in g.es.select(lambda edge: edge['distance'] > threshold)]
    g.delete_edges(edges_to_delete)
    g = g.spanning_tree(weights=g.es["distance"])
    '可视化当前网络'
    # visual_graph(g)

    # isolated_vertices = [v.index for v in g.vs.select(_degree=0)]
    # # 打印孤立的点
    # if len(isolated_vertices) > 0:
    #     print("孤立的点：", isolated_vertices)

    return g


#############################按照时间创建轨迹##########################
def creat_track(MMSI_id, group, node_index_map, current_index, SubGraph_number):
    g = ig.Graph()
    processing_ship = group[['LAT', 'LON', 'GNN-label']].to_numpy()

    # 添加节点并映射节点索引
    for i in range(1, len(processing_ship) + 1):
        node_id = str(current_index)
        attributes = {'MMSI': MMSI_id, 'LAT': processing_ship[i - 1][0], 'LON': processing_ship[i - 1][1],
                      'SubGraphnumber': SubGraph_number, 'GNN-label': processing_ship[i - 1][2]}
        g.add_vertex(node_id, **attributes)

        node_index_map[(MMSI_id, i)] = current_index
        current_index += 1

    # 添加边
    edge_ = 0
    for i in range(1, len(processing_ship)):
        # 使用映射后的节点索引添加边
        source_node = str(node_index_map[(MMSI_id, i)])
        target_node = str(node_index_map[(MMSI_id, i + 1)])

        # 计算两点的权重
        gd = GD([g.vs[i - 1]['LAT'], g.vs[i - 1]['LON']],
                [[g.vs[i]['LAT'], g.vs[i]['LON']]]).kilometers

        edge_attrs = {'SubGraphnumber': SubGraph_number, 'GNN-label': g.vs[i - 1]['GNN-label'], 'distance': gd
                      }
        g.add_edge(source_node, target_node, **edge_attrs)
        edge_ += 1
    return g



#############################可视化Delaunay####################
def visual_Delaunay(trajectory_pointsets, delaunay):
    plt.triplot(trajectory_pointsets[:, 1:2].flatten(), trajectory_pointsets[:, :1].flatten(), delaunay.simplices,
                c='black',
                linewidth=0.1)
    # selected_indices = [47, 91, 99]
    # selected_rows = trajectory_pointsets[selected_indices]
    # print(selected_rows)
    # plt.plot(selected_rows[:, 1:2].flatten(), selected_rows[:, :1].flatten(), '.', c='red', markersize=5)
    plt.plot(trajectory_pointsets[:, 1:2].flatten(), trajectory_pointsets[:, :1].flatten(), '.', c='green',
             markersize=3)
    plt.axis('equal')
    plt.show()


#############################可视化graph####################
# def visual_graph(g):
#     # 创建节点的经纬度坐标
#     node_x = g.vs['LON']
#     node_y = g.vs['LAT']
#     # 创建节点的文本标签（可选）
#     node_labels = [f"Node {i}" for i in range(len(g.vs))]
#     # 创建节点的散点图
#     node_trace = go.Scatter(
#         x=node_x,
#         y=node_y,
#         text=node_labels,
#         # mode='markers+text',  # 可以添加文本标签
#         marker=dict(size=2, color='blue')
#     )
#     # 创建边的线图
#     edge_x = []
#     edge_y = []
#     for edge in g.get_edgelist():
#         x0, y0 = g.vs[edge[0]]['LON'], g.vs[edge[0]]['LAT']
#         x1, y1 = g.vs[edge[1]]['LON'], g.vs[edge[1]]['LAT']
#         edge_x.extend([x0, x1, None])
#         edge_y.extend([y0, y1, None])
#     edge_trace = go.Scatter(
#         x=edge_x,
#         y=edge_y,
#         line=dict(width=1, color='gray'),
#         hoverinfo='none',
#         mode='lines'
#     )
#     # 创建图布局
#     layout = go.Layout(
#         showlegend=False,
#         hovermode='closest',
#         xaxis=dict(showgrid=False, zeroline=False),
#         yaxis=dict(showgrid=False, zeroline=False)
#     )
#     # 创建图形并绘制
#     fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
#     fig.show()


################################ 图稀疏矩阵(DS_A.txt) #############################
def to_SHIP_A(g, outpath):
    edge_data = []
    dataset_name=outpath.split("\\")[-2]

    for edge in g.es:
        source = edge.source
        target = edge.target
        # 使用索引获取顶点的名称
        source_name = g.vs[source]['name']
        target_name = g.vs[target]['name']
        attributes = edge.attributes()  # 获取所有属性
        edge_data.append({"source": source_name, "target": target_name, **attributes})
    edges = pd.DataFrame(edge_data)
    # edges[["source", "target"]] += 1  # 列值加一 模型编码从1起始
    # print(edges[["source", "target"]])
    edges[["source", "target"]].to_csv(outpath + f"/{dataset_name}_A.txt",
                                       sep=',', index=False, header=False, mode='a', lineterminator='\r\n')


################################ 节点所属图编号(DS_graph_indicator.txt) ##########################
def to_SHIP_graph_indicator(g, outpath):
    dataset_name=outpath.split("\\")[-2]
    node_data = []
    for node in g.vs:
        attributes = node.attributes()  # 获取所有属性
        node_data.append({**attributes})
    nodes = pd.DataFrame(node_data)
    nodes["SubGraphnumber"].to_csv(outpath +f"/{dataset_name}_graph_indicator.txt", sep=',',
                                   index=False, mode='a', header=False)


################################ 图标签(DS_graph_labels.txt) ################################
def to_graph_labels(Graph_labels, outpath):
    dataset_name=outpath.split("\\")[-2]
    # Graph_labels = [int(x) for x in Graph_labels]
    # Graph_labels = np.array(Graph_labels)
    # np.savetxt(filepath+"/SHIP_MST_graph_labels.txt", Graph_labels,mode="a",
    #            fmt='%d')
    with open(outpath + f"/{dataset_name}_graph_labels.txt", "a") as f:
        for i in Graph_labels:
            f.write(str(i))
            f.write("\n")


################################ 节点特征(DS_node_attributes.txt) ################################
def to_node_attributes(group, outpath, graph):
    # print(group.columns)
    dataset_name=outpath.split("\\")[-2]

    group = group[
        ['LAT', 'LON', 'SOG', 'COG', 'Length1', 'Beam', 'len_wids' ,
         'count','time',
         # 'sum_dt', 'sum_dl', 'sum_da', 'mean_S', 'Delta','S/D',
         'dt', 'dl', 'da']]  # 个体特征 & 片段特征
    #
    # '网络节点的拓扑特征'
    # '特征计算'
    # graph.vs["degree"] = graph.degree()  # 节点度
    # graph.vs["EC"] = graph.eigenvector_centrality(directed=False)  # 特征向量中心性
    # graph.vs["BC"] = graph.betweenness(directed=False)  # 中介中心性
    # graph.vs["CC"] = graph.closeness(normalized=False)  # 临近中心性
    # # 局部聚集系数
    # local_clustering_coefficients = []
    # for node_index in range(graph.vcount()):
    #     local_clustering_coefficient = graph.transitivity_local_undirected(vertices=node_index, mode="zero")
    #     local_clustering_coefficients.append(local_clustering_coefficient)
    # graph.vs["local_CC"] = local_clustering_coefficients
    # '特征提取'
    # node_attributes = {}
    # for attribute_name in graph.vs.attributes():
    #     node_attributes[attribute_name] = graph.vs[attribute_name]
    # topology_features = pd.DataFrame(node_attributes)
    # topology_features = topology_features[['degree', 'EC', 'BC', 'CC', 'local_CC']]
    # '个体特征 & 片段特征 & 拓扑特征'
    # out_df = pd.concat([group, topology_features], axis=1)
    out_df=group
    '空值检查'
    columns_to_check = out_df.columns
    null_counts = {}
    for col in columns_to_check:
        null_counts[col] = out_df[col].isna().sum()
    # print("各列空值数目:", null_counts)
    # print(out_df)

    # filtered_data["Length"] = filtered_data["Length"].replace("-", -1)
    # filtered_data["Beam"] = filtered_data["Beam"].replace("-", -1)
    # Node_feature_output_columns = ["LAT", "LON", "SOG", "COG", "Length", "Beam"]  # , "Length", "Width", "Draft"
    # filtered_data = filtered_data[Node_feature_output_columns].apply(pd.to_numeric, errors="coerce")[:n - 1]
    # # filtered_data = filtered_data[Node_feature_output_columns]

    # print(out_df)
    # out_df = out_df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))  # 特征归一化

    out_df.to_csv(outpath + f"/{dataset_name}_node_attributes.txt",
                  sep=',', index=False, header=False, mode='a', float_format='%.3f')  # 节点特征


################################ 节点特征(DS_node_attributes.txt) ################################

def to_edge_attributes(outpath, g):
    dataset_name=outpath.split("\\")[-2]
    edge_distance = [edge.attributes()['distance'] for edge in g.es]
    edge_weight = [1 / edge.attributes()['distance'] for edge in g.es]
    # edge_weight=[]
    # for edge in g.es:
    #     try:
    #         weight = 1 / edge.attributes()['distance']
    #         edge_weight.append(weight)
    #     except ZeroDivisionError:
    #         print(edge.attributes()['distance'])
    #         # print(f"Error: Division by zero for edge attributes: {edge.attributes()}")
    #         edge_weight.append(None)  # 或者你可以将 edge_weight 中的值设定为适当的默认值，这里使用 None 作为示例
    edges = pd.DataFrame({'edge_weight': edge_weight, 'edge_distance': edge_distance})
    edges = edges.round(4)
    edges[["edge_weight", "edge_distance"]].to_csv(outpath + f"/{dataset_name}_edge_attributes.txt",
                                       sep=',', index=False, header=False, mode='a', lineterminator='\r\n')


def visual_graph(g):
    nxg = nx.Graph(g.get_edgelist())

    # 使用 networkx 绘制图形
    pos = nx.spring_layout(nxg)  # 使用弹簧布局算法
    nx.draw(nxg, pos, with_labels=True, font_weight='bold')

    # 显示图形
    plt.show()




if __name__ == "__main__":

    filepath = r"E:\项目\model_practice\python\GNN\New_York_data\4_feature"
    outpath = r"E:\项目\model_practice\python\GNN\New_York_data\5_network\New_York_PL\raw"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    for filename in os.listdir(outpath):
        os.remove(os.path.join(outpath, filename))


    file_names = [f for f in os.listdir(filepath) if f.endswith('.csv')]

    node_index_map = {}
    current_index = 1
    SubGraph_number = 1

    for file_name in file_names:
        file_path = os.path.join(filepath, file_name)

        Graph_labels = []
        subgraph = 1

        df = pd.read_csv(file_path)

        for MMSI_id, group in tqdm(df.groupby('SegmentID')):
            '预处理'

            group.reset_index(inplace=True)
            '构网 network_deluanay network_MST creat_track'
            g = creat_track(MMSI_id, group, node_index_map, current_index, SubGraph_number)
            # visual_graph(g)
            '当前图标签添加，同类型船舶同时输出'
            Graph_labels.append(group['GNN-label'].iloc[0])

            to_SHIP_A(g, outpath)
            to_SHIP_graph_indicator(g, outpath)
            to_node_attributes(group, outpath, g)
            to_edge_attributes(outpath, g)
            current_index += len(group)
            SubGraph_number += 1

            if subgraph == 1000000:
                break
            subgraph += 1

        to_graph_labels(Graph_labels, outpath)