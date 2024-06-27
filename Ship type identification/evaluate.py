import time
import os
import os.path as osp
import shutil
import pandas as pd
import numpy as np

from tqdm import tqdm
import torch
import torch.mps
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from torch import tensor
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary


if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


def cross_validation_with_val_set(dataset, model, folds, epochs, batch_size,
                                  lr, lr_decay_factor, lr_decay_step_size,
                                  weight_decay, dataset_name, lf_weight, logger=None):
    outpath = osp.join(osp.dirname(osp.realpath(__file__)), '..', f'data/{dataset_name}/result')
    try:
        shutil.rmtree(outpath)
    except Exception as e:
        pass

    train_accs, val_losses, test_accs, durations, best_model_predictions = [], [], [], [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, folds))):

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()  # Replacement model

        # View the number of model parameters
        for data in train_loader:
            data = data.to('cuda:0')
            print(summary(model, data))
            break

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # Optimizer: Learning rate, weight decay

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch.backends,
                     'mps') and torch.backends.mps.is_available():
            try:
                torch.mps.synchronize()
            except ImportError:
                pass

        t_start = time.perf_counter()

        best_val_loss = float('inf')
        best_model_state = None

        k = str(fold + 1) + '-fold'
        with tqdm(total=epochs, desc=k, position=0, leave=True) as pbar:
            for epoch in range(1, epochs + 1):
                train_loss = train(model, optimizer, lf_weight, train_loader)

                train_acc = eval_acc(model, train_loader)
                train_accs.append(train_acc)
                val_loss = eval_loss(model, val_loader)
                val_losses.append(val_loss)
                test_acc = eval_acc(model, test_loader)
                test_accs.append(test_acc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()

                pbar.update(1)
                pbar.set_postfix(
                    {'current_train_acc': '{:.4f}'.format(train_acc), 'best_val_loss': '{:.4f}'.format(best_val_loss)})

                eval_info = {
                    'fold': fold,
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_losses[-1],
                    'test_acc': test_accs[-1],
                }

                if logger is not None:
                    logger(eval_info)

                # Learning rate decay
                if epoch % lr_decay_step_size == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_decay_factor * param_group['lr']

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch.backends,
                     'mps') and torch.backends.mps.is_available():
            torch.mps.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

        '''more performance indicators'''
        model.load_state_dict(best_model_state)
        p_t_l, cm_pct, precision, recall, f1 = classification_model_performance(model, test_loader)
        best_model_predictions.append([precision, recall, f1])
        p_t_l['id'] = pd.Series(test_idx)

        outpathp_t_l = osp.join(osp.dirname(osp.realpath(__file__)), '..', f'data/{dataset_name}/result/P_T_L')
        if not os.path.exists(outpathp_t_l):
            os.makedirs(outpathp_t_l)
        p_t_l.to_csv(outpathp_t_l + f"/{dataset_name}_{k}_P_T_L.txt", index=False, header=False, sep=',')

        '''y_score'''
        true_labels, predicted_probs = eval_roc(model, test_loader)
        predicted_probs = pd.DataFrame(predicted_probs)
        outpath_y_score = osp.join(osp.dirname(osp.realpath(__file__)), '..', f'data/{dataset_name}/result/y_score')
        if not os.path.exists(outpath_y_score):
            os.makedirs(outpath_y_score)
        predicted_probs.to_csv(outpath_y_score + f"/{dataset_name}_{k}_y_score.txt", index=False, header=False, sep=',')

    # print(val_losses)
    # print(test_accs)
    # print(durations)
    out_train_accs = pd.DataFrame({'train_accs': train_accs})
    out_val_losses = pd.DataFrame({'val_losses': val_losses})
    out_test_accs = pd.DataFrame({'test_accs': test_accs})

    out_train_accs.to_csv(outpath + f"/{dataset_name}_train_accs.txt", index=False, header=False)
    out_val_losses.to_csv(outpath + f"/{dataset_name}_val_losses.txt", index=False, header=False)
    out_test_accs.to_csv(outpath + f"/{dataset_name}_test_accs.txt", index=False, header=False)

    loss, acc, duration = tensor(val_losses), tensor(test_accs), tensor(durations)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    loss, argmin = loss.min(dim=1)
    acc = acc[torch.arange(folds, dtype=torch.long), argmin]

    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    duration_mean = duration.mean().item()
    print(f'Val Loss: {loss_mean:.4f}, Test Accuracy: {acc_mean:.3f} '
          f'± {acc_std:.3f}, Duration: {duration_mean:.3f}')

    return loss_mean, acc_mean, acc_std


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, lf_weight, loader):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        lf_weight = lf_weight.to(device)
        """cross_entropy = log_softmax + nll_loss  """
        loss = F.nll_loss(out, data.y.view(-1), weight=lf_weight)  # , weight=lf_weight
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset), model


def eval_acc(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)


def classification_model_performance(model, loader):
    model.eval()
    predictions = []
    true_labels = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            predicted_labels = torch.argmax(out, dim=1)
            predictions.extend(predicted_labels.cpu().numpy())
            true_labels.extend(data.y.cpu().numpy())

    P_T = pd.DataFrame({'predictions': predictions, 'true_labels': true_labels})
    cm = confusion_matrix(true_labels, predictions)
    cm_pct = pd.DataFrame(np.round(cm / cm.sum(axis=1)[:, None] * 100, decimals=2),
                          index=['Cargo', 'Tanker', 'Passenger', 'Tug', 'Fishing', 'Craft'],
                          columns=['Cargo', 'Tanker', 'Passenger', 'Tug', 'Fishing', 'Craft'])

    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')

    print(cm_pct)
    print(precision, recall, f1)
    print(' ')

    return P_T, cm_pct, precision, recall, f1


def eval_roc(model, loader):
    model.eval()

    all_true = []
    all_probs = []

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            probs = torch.exp(output)  # Getting a probability distribution
            all_probs.extend(probs.cpu().numpy())
            all_true.extend(data.y.cpu().numpy())

    return all_true, all_probs


@torch.no_grad()
def inference_run(model, loader, bf16):
    model.eval()
    for data in loader:
        data = data.to(device)
        if bf16:
            data.x = data.x.to(torch.bfloat16)
        model(data)
