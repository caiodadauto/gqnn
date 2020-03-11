import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

from .utils import from_data

def draw_graphs(data):
    graphs = from_data(data)
    n_graphs = len(graphs)
    fig, axs = plt.subplots(1, n_graphs, dpi=120, figsize=(12, 8))
    for G, ax in zip(graphs, axs):
        pos = nx.spring_layout(G)
        num_interfaces = G.graph["num_interfaces"]
        nodes = list(G.node())
        router_idx = nodes[:-num_interfaces]
        interface_idx = nodes[-num_interfaces:]
        ax.set_axis_off()

        colors_router = []
        target_idx = G.graph["target"]
        for i in router_idx:
            if i == target_idx:
                colors_router.append("yellow")
            else:
                colors_router.append("gray")
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=router_idx,
            node_color=colors_router,
            linewidths=1.8,
            node_size=280,
            edgecolors="k",
            ax=ax)
        labels = dict(zip(router_idx, [str(i) for i in router_idx]))
        nx.draw_networkx_labels(G, pos, font_color="k", labels=labels, ax=ax)

        colors_interface = []
        for _, out in list(G.node(data="out"))[-num_interfaces:]:
            if out == 1:
                colors_interface.append("g")
            else:
                colors_interface.append("r")
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=interface_idx,
            node_color=colors_interface,
            linewidths=1.8,
            node_size=180,
            edgecolors="k",
            alpha=.6,
            ax=ax)
        labels = dict(zip(interface_idx, [str(i) for i in interface_idx]))
        nx.draw_networkx_labels(G, pos, font_color="k", labels=labels, ax=ax)

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax)
    plt.save

def plot_data(data):
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
    draw_graphs(graphs)
