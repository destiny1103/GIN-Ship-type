import os.path as osp
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, Normalizer

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")


class NormalizedDegree:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def get_dataset(name, sparse=True, cleaned=False, normalization='min-max'):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    dataset = TUDataset(path, name, use_node_attr=True, use_edge_attr=True, cleaned=cleaned)
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    if not sparse:
        num_nodes = max_num_nodes = 0
        for data in dataset:
            num_nodes += data.num_nodes
            max_num_nodes = max(data.num_nodes, max_num_nodes)

        # Filter out a few really large graphs in order to apply DiffPool.
        if name == 'REDDIT-BINARY':
            num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
        else:
            num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)

        indices = []
        for i, data in enumerate(dataset):
            if data.num_nodes <= num_nodes:
                indices.append(i)
        dataset = dataset.copy(torch.tensor(indices))

        if dataset.transform is None:
            dataset.transform = T.ToDense(num_nodes)
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(num_nodes)])

    ''' ['LAT', 'LON', 'SOG', 'COG', 'Length1', 'Beam', 'len_wids', 'count','time', 'dt', 'dl', 'da']] '''
    # Ablation experiment, Feature selection
    # Feature combinations(x_columns): LF(0, 1) / MF(2) / L&B(4, 5) / AR(6) / NF(7) / MCF(9, 10, 11)
    x_columns = [0, 1, 2, 4, 5, 7, 9, 10, 11]
    new_x = dataset.x[:, x_columns]
    dataset.data.x = new_x

    # attr_columns = [1]  # ["edge_weight", "edge_distance"]
    # new_edge_attr = dataset.edge_attr[:, attr_columns]
    # dataset.data.edge_attr = new_edge_attr

    scaler = None
    if normalization == 'min-max':
        scaler = MinMaxScaler()
    elif normalization == 'z-score':
        scaler = StandardScaler()
    elif normalization == 'MaxAbs':
        scaler = MaxAbsScaler()
    elif normalization == 'RobustScaler':
        scaler = RobustScaler()
    elif normalization == 'Normalizer':
        scaler = Normalizer()
    dataset.data.x = torch.tensor(scaler.fit_transform(dataset.data.x), dtype=torch.float32)
    dataset.data.edge_attr = torch.tensor(scaler.fit_transform(dataset.data.edge_attr), dtype=torch.float32)
    # Calculation of sample proportion: lf_weight
    _, lf_weight = Hybrid_lf(dataset)

    return dataset, lf_weight


def Hybrid_lf(dataset):
    from collections import Counter
    class_count = Counter()
    for data in dataset:
        label = int(data.y)
        class_count[label] += 1

    counter_dict = dict(class_count)
    sorted_dict = dict(sorted(counter_dict.items()))
    sorted_values = list(sorted_dict.values())
    weighted = [sum(sorted_values) / num for num in sorted_values]
    weight = torch.tensor(weighted, dtype=torch.float)
    print(weight)

    return class_count, weight


def print_dataset(dataset):
    num_nodes = num_edges = 0
    for data in dataset:
        num_nodes += data.num_nodes
        num_edges += data.num_edges

    print('Name:', dataset)
    print('Graphs:', len(dataset))
    print('Nodes:', num_nodes / len(dataset))
    print('Edges:', (num_edges // 2) / len(dataset))
    print('Features:', dataset.num_features)
    print('Classes:', dataset.num_classes)
    print()


if __name__ == '__main__':
    # (Gulf_of_Mexico_ / Delaware_Bay_) & (DT / MST / PL)
    dataset, lf_weight = get_dataset('Gulf_of_Mexico_PL', sparse=True)
    print_dataset(dataset)
    print(dataset.data)

    class_count, result_tensor = Hybrid_lf(dataset)
    for label, count in class_count.items():
        print(f'Class {label}: {count} graphs')
    print(result_tensor)