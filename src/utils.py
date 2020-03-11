import time

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from sklearn.metrics import balanced_accuracy_score


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

def save_model(model, optimizer, loss, n_batch, epoch, duration, path, best):
    path = os.path.join(path, "checkpoint/")
    if not os.path.isdir(path):
        os.makedirs(path)

    env_state = {'duration': duration,
                 'epoch': epoch,
                 'n_batch':n_batch,
                 'loss': loss,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict()}
    if best:
        torch.save(model.state_dict(), path + "env_state_best.pt")
    torch.save(optimizer.state_dict(), path + "env_state.pt")

def save_info(duration, n_batch, loss, acc, path):
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(os.path.join(path, "info.csv"), "a") as f:
        f.write("{:.2f},{:d},{:.5f},{:.5f}\n".format(duration, n_batch, loss, acc))

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

def train(device, model, loader, optimizer, scheduler, loss_fn, path, dt=20, time_offset=0):
    n_batch = 1
    best_acc = 0
    best_loss = np.Inf

    model.train()
    start_time = time.time()
    last_log_time = start_time
    for data in tqdm(loader):
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
            save_model(model, optimizer, loss, n_batch, 1, , path, best)
            save_info(duration, n_batch, loss, acc, path)
        scheduler.step()
        n_batch += 1
