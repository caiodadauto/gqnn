import os
import time

import torch
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from torch_geometric.data import Data
from sklearn.metrics import balanced_accuracy_score


NAME_INFO_FILE = "info"
NAME_ENV_FILE = "env_state"
NAME_STATS_FILE = "accs"

NAME_STATS_DIR = "stats"
NAME_BKP_DIR = "checkpoint"

def from_networkx(G, G_target):
    def add_edge_nodes(edges, n_nodes):
        new_edges = []
        edgenode_index = {}
        for e, i in zip(edges, list(range(n_nodes, n_nodes + len(edges)))):
            edgenode_index[e] = i
        for (s, r), i in edgenode_index.items():
            j = edgenode_index[(r, s)]
            new_edges = new_edges + [(s, i), (i, j), (j, r)]
        return new_edges

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G
    new_edges = add_edge_nodes(list(G.edges), n_nodes)
    edge_index = torch.tensor(new_edges).t().contiguous()

    for i, value in dict(G.nodes(data="features")).items():
        X = [value] if i == 0 else X + [value]
    for _, _, value in G.edges(data="features"):
        value = np.concatenate([value, np.zeros(32 - len(value))])
        X = X + [value]

    y = [y for _, _, y in G_target.edges(data="features")]
    y = torch.Tensor(y)

    data = Data(x=torch.Tensor(X),
                edge_index=edge_index,
                num_routers=n_nodes,
                num_interfaces=n_edges,
                num_all_edges=len(new_edges),
                targets=torch.Tensor([G.graph["features"]]),
                y=y)
    return data

def from_data(data):
    graphs = []
    x = data.x.numpy()
    y = data.y.numpy()
    num_routers = data.num_routers.numpy()
    num_interfaces = data.num_interfaces.numpy()
    num_all_edges = data.num_all_edges.numpy()
    edge_index = data.edge_index.t().numpy()

    cum_num_edges = np.cumsum(num_all_edges, axis=0)
    cum_num_nodes = np.cumsum(num_routers + num_interfaces, axis=0)
    cum_num_interfaces = np.cumsum(num_interfaces, axis=0)
    edge_bounds = np.stack([np.concatenate([[0], cum_num_edges])[:-1], cum_num_edges]).T
    node_bounds = np.stack([np.concatenate([[0], cum_num_nodes])[:-1], cum_num_nodes]).T
    interface_bounds = np.stack([np.concatenate([[0], cum_num_interfaces])[:-1], cum_num_interfaces]).T
    for n, ((sn, en), (se, ee), (si, ei)) in enumerate(zip(node_bounds, edge_bounds, interface_bounds)):
        translate = dict(zip(range(sn, en), range(en - sn)))
        nodes = [(i, dict(features=f)) for i, f in enumerate(x[sn:en])]
        edges = [(translate[s], translate[r]) for s, r in edge_index[se:ee]]
        for out, (_, d) in zip(y[si:ei], nodes[-num_interfaces[n]:]):
            d["out"] = int(out)
        target_ip = data.targets[n].numpy()
        for i, d in nodes:
            if np.all(target_ip == d["features"]):
                target_idx = i
                break

        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        G.graph["num_interfaces"] = int(data.num_interfaces[n])
        G.graph["target"] = target_idx
        graphs.append(G)
    return graphs

def save_model(model, optimizer, scheduler, loss, acc, n_batch, epoch, duration, path, best):
    if not os.path.isdir(path):
        os.makedirs(path)

    env_state = {'duration': duration,
                 'epoch': epoch,
                 'n_batch':n_batch,
                 'loss': loss,
                 'acc': acc,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict()}
    if best:
        torch.save(os.path.join(env_state, path, NAME_ENV_FILE + "_best.pt"))
    torch.save(os.path.join(env_state, path, NAME_ENV_FILE + "_last.pt"))

def save_info(n_epoch, n_batch, duration, loss, acc, path):
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(os.path.join(path, NAME_INFO_FILE + ".csv"), "a") as f:
        f.write("{:d},{:d},{:.2f},{:.5f},{:.5f}\n".format(n_epoch, n_batch, duration, loss, acc))

def is_previous_trainig(path):
    last_file =  os.path.isfile(path + NAME_ENV_FILE + "_last.pt")
    best_file =  os.path.isfile(path + NAME_ENV_FILE + "_best.pt")
    if last_file and best_file:
        return True
    return False

def load_model(model, path, optimizer=None, scheduler=None, test=False):
    best_cp = torch.load(os.path.join(path, NAME_ENV_FILE + "_best.pt"))
    best_acc = best_cp['acc']
    best_loss = best_cp['loss']
    best_batch = best_cp['n_batch']
    best_duration = best_cp['duration']

    last_cp = torch.load(os.path.join(path, NAME_ENV_FILE + "_last.pt"))
    last_epoch = last_cp['epoch']
    last_batch = last_cp['n_batch']
    last_duration = last_cp['duration']

    if not test:
        if not optimizer:
            raise ValueError("Need an initialized optimizer to load model")
        model.load_state_dict(last_cp['model_state_dict'])
        optimizer.load_state_dict(last_cp['optimizer_state_dict'])
        if scheduler:
            scheduler.load_state_dict(last_cp['scheduler_state_dict'])
        return last_batch, last_epoch, duration, best_acc, best_loss
    else:
        model.load_state_dict(best_cp['model_state_dict'])
        return best_batch, best_duration, best_acc

def train_step(model, data, optimizer, loss_fn, threshold, class_weight):
    optimizer.zero_grad()
    output = model(data)
    label = data.y
    loss_weight = ( ( ( label - 1 ) * -1 ) * class_weight[0] ) + ( label * class_weight[1] )
    loss = loss_fn(output, label, weight=loss_weight)
    loss.backward()
    optimizer.step()
    label_np = data.y.detach().cpu().numpy()
    output_np = output.detach().cpu().numpy()
    return loss.item(), np.array(output_np > threshold, dtype=int), label

def train(device, model, loader, optimizer, scheduler, loss_fn, epochs, path, threshold=.35, dt=20, class_weight=[1., 1.]):
    n_batch = 0
    n_epoch = 0
    best_acc = 0
    last_batch = 0
    time_offset = 0
    best_loss = np.Inf
    path = os.path.join(path, NAME_BKP_DIR)

    if is_previous_trainig(path):
        last_batch, n_epoch, time_offset, best_acc, best_loss = load_model(model, path, optimizer)
        print("Last model loaded,\n\tLast batch: {}".format(last_batch))

    model.train()
    start_time = time.time()
    last_log_time = start_time
    for epoch in range(n_epoch, epochs):
        for data in tqdm(loader):
            if n_batch > last_batch:
                data = data.to(device)
                loss, output, label = train_step(model, data, optimizer, loss_fn, threshold, class_weight)
                current_time = time.time()
                if current_time - last_log_time > dt:
                    last_log_time = current_time
                    duration = current_time - start_time + time_offset
                    acc = balanced_accuracy_score(label, output)
                    if best_loss > loss and best_acc < acc:
                        best_loss = loss
                        best_acc = acc
                        best = True
                    else:
                        best = False
                    save_model(model, optimizer, loss, acc, n_batch, epoch, duration, path, best)
                    save_info(epoch, n_batch, duration, loss, acc, path)
                scheduler.step()
            n_batch += 1
        n_batch = 0
        last_batch = 0

def split_per_graphs(x, n_inter):
    cum_inter = np.cumsum(n_inter)
    x_per_graphs = [x[(n - n_inter[i]):n] for i, n in enumerate(cum_inter)]
    return x_per_graphs

def true_positive_rate(expect, predict):
    mask = expect == 1
    return np.mean(expect[mask] == predict[mask])

def true_negative_rate(expect, predict):
    mask = expect == 0
    return np.mean(expect[mask] == predict[mask])

def get_stats(model, data, threshold):
    output = model(data)
    label_np = data.y.detach().cpu().numpy()
    output_np = output.detach().cpu().numpy()
    output_np = np.array(output_np > threshold, dtype=int)
    num_interfaces_np = data.num_interfaces.detach().cpu().numpy()

    stats = []
    label_per_graphs = split_per_graphs(label_np, num_interfaces_np)
    output_per_graphs = split_per_graphs(output_np, num_interfaces_np)
    for expect, predict in zip(label_per_graphs, output_per_graphs):
        acc = balanced_accuracy_score(expect, predict)
        tpr = true_positive_rate(expect, predict)
        tnr = true_negative_rate(expect, predict)
        stats.append([acc, tpr, tnr])
    return pd.DataFrame(stats, columns=["ACC", "TPR", "TNR"])

def save_stats(df_stats, path):
    if not os.path.isdir(path):
        os.makedirs(path)
    df_stats.to_csv(os.path.join(path, NAME_STATS_FILE + ".csv"))

def test(device, model, loader, path, threshold=.35):
    bkp_path = os.path.join(path, NAME_BKP_DIR)
    stats_path = os.path.join(path, NAME_STATS_DIR)

    best_batch, best_duration, best_acc = load_model(model, bkp_path, test=True)
    print("Model loaded for the best training perform at batch {} after {} of training, where it achieved {} of balanced accuracy".format(
            best_batch, best_duration, best_acc))
    model.eval()

    data = next(iter(loader))
    data = data.to(device)
    df_stats = get_stats(model, data, threshold)
    save_stats(df_stats, stats_path)
