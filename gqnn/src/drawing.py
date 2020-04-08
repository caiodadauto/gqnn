import os

import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader

from .utils import from_data


#def draw_accs(df, path):
#    fig = plt.figure(dpi=300)
#    ax = fig.subplots(1, 1, sharey=False)
#    if k == r"ACC":
#        lmean = r"$\overline{ACC}$"
#        file_name = "acc"
#    elif k == r"TPR":
#        lmean = r"$\overline{TPR}$"
#        file_name = "true_acc"
#    else:
#        lmean = r"$\overline{TNR}$"
#        file_name = "false_acc"

#    sns.distplot(v_tr, ax=ax, hist_kws=dict(zorder=1, cumulative=True, density=True, range=(0,1), label=r"Non-Generalization, {} = {:.3f}".format(lmean, v_tr.mean())), kde_kws=dict(cumulative=True))
#    sns.distplot(v_ge, ax=ax, hist_kws=dict(zorder=0, cumulative=True, density=True, range=(0,1), label=r"Generalization, {} = {:.3f}".format(lmean, v_ge.mean())), kde_kws=dict(cumulative=True))
#    ax.set_xlabel(k)
#    ax.set_ylabel("Cumulative Frequency")
#    ax.legend()
#    ax.set_yticks(np.arange(0, 1.25, .25))
#    #plt.axhline(.5, ls="--", alpha=.7, c="k")
#    #ax.xaxis.grid(True)
#    ax.yaxis.grid(True)
#    fig.tight_layout()
#    plt.savefig(file_name + ".pdf", transparent=True)
#    plt.show()
#    fig.clear()
#    plt.close()

def draw_batch(dataset, path, logger=None):
    if not os.path.isdir(path):
        os.makedirs(path)

    loader = DataLoader(dataset, batch_size=3)
    batch = next(iter(loader))
    graphs = from_data(batch)
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
    if logger:
        logger.info("Save graph input ilustration in {}".format(os.path.join(path, "graphs.png")))
    plt.savefig(os.path.join(path, "graphs.png"))
