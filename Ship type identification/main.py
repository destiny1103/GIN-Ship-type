# parameter setting
# Save the process file Go to evaluate.py to see the code

import argparse
from itertools import product

from datasets import get_dataset
from model.diff_pool import DiffPool
from model.gcn import GCN, GCNWithJK
from model.gat import GAT, GATWithJK
from model.gin import GIN, GIN0, GIN0WithJK, GINWithJK
from model.graph_sage import GraphSAGE, GraphSAGEWithJK
from evaluate import cross_validation_with_val_set

# model hyperparameterization
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
args = parser.parse_args()
# model module
layers = [2]
hiddens = [64]
# Data set selection: (Gulf_of_Mexico_ / Delaware_Bay_) & (DT / MST / PL)
datasets = ['Gulf_of_Mexico_PL']
# Model Selection: GCN, GCNWithJK, GAT, GATWithJK, GIN, GINWithJK, GraphSAGE, GraphSAGEWithJK,
nets = [GINWithJK]


def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print(f'{fold:02d}/{epoch:03d}: Val Loss: {val_loss:.4f}, '
          f'Test Accuracy: {test_acc:.3f}')


results = []
for dataset_name, Net in product(datasets, nets):
    best_result = (float('inf'), 0, 0)  # (loss, acc, std)
    print(f'--\n{dataset_name} - {Net.__name__}')
    for num_layers, hidden in product(layers, hiddens):
        dataset, lf_weight = get_dataset(dataset_name, sparse=Net != DiffPool, normalization='z-score')  # Normalizer
        model = Net(dataset, num_layers, hidden)
        loss, acc, std = cross_validation_with_val_set(
            dataset,
            model,
            folds=5,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0.001,  # weight_decay
            logger=None,
            dataset_name=dataset_name,
            lf_weight=lf_weight
        )
        if loss < best_result[0]:
            best_result = (loss, acc, std)

    desc = f'{best_result[1]:.3f} ± {best_result[2]:.3f}'
    print(f'Best result - {desc}')
    results += [f'{dataset_name} - {model}: {desc}']
results = '\n'.join(results)
print(f'--\n{results}')
