import os
import time

import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from torch_geometric.data import Data
from sklearn.metrics import balanced_accuracy_score


NAME_ENV = "env_state"
NAME_INFO = "info"

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

def save_model(model, optimizer, loss, acc, n_batch, epoch, duration, path, best):
    if not os.path.isdir(path):
        os.makedirs(path)

    env_state = {'duration': duration,
                 'epoch': epoch,
                 'n_batch':n_batch,
                 'loss': loss,
                 'acc': acc,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict()}
    if best:
        torch.save(env_state, path + NAME_ENV + "_best.pt")
    torch.save(env_state, path + NAME_ENV + "_last.pt")

def save_info(n_epoch, n_batch, duration, loss, acc, path):
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(os.path.join(path, NAME_INFO + ".csv"), "a") as f:
        f.write("{:d},{:d},{:.2f},{:.5f},{:.5f}\n".format(n_epoch, n_batch, duration, loss, acc))

def is_previous_trainig(path):
    last_file =  os.path.isfile(path + NAME_ENV + "_last.pt")
    best_file =  os.path.isfile(path + NAME_ENV + "_best.pt")
    if last_file and best_file:
        return True
    return False

def load_model(model, optimizer, path):
    last_cp = torch.load(path + NAME_ENV + "_last.pt")
    best_cp = torch.load(path + NAME_ENV + "_best.pt")

    model.load_state_dict(last_cp['model_state_dict'])
    optimizer.load_state_dict(last_cp['optimizer_state_dict'])
    last_epoch = last_cp['epoch']
    last_batch = last_cp['n_batch']
    duration = last_cp['duration']
    best_acc = best_cp['acc']
    best_loss = best_cp['loss']
    return last_batch, last_epoch, duration, best_acc, best_loss

def train_step(model, data, optimizer, loss_fn):
    optimizer.zero_grad()
    output = model(data)
    label = data.y
    loss = loss_fn(output, label)
    loss.backward()
    optimizer.step()
    label_np = data.y.detach().cpu().numpy()
    output_np = output.detach().cpu().numpy()
    return loss.item(), np.array(output_np > .35, dtype=int), label

def train(device, model, loader, optimizer, scheduler, loss_fn, epochs, path, dt=20):
    n_batch = 0
    n_epoch = 0
    best_acc = 0
    last_batch = 0
    time_offset = 0
    best_loss = np.Inf
    path = os.path.join(path, "checkpoint/")

    if is_previous_trainig(path):
        last_batch, n_epoch, time_offset, best_acc, best_loss = load_model(model, optimizer, path)

    model.train()
    start_time = time.time()
    last_log_time = start_time
    for epoch in range(n_epoch, epochs):
        for data in tqdm(loader):
            if n_batch > last_batch:
                data = data.to(device)
                loss, output, label = train_step(model, data, optimizer, loss_fn)
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
