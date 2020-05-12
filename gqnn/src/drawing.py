import os

import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader

from .nxutils import from_data


def get_dist_params(acc_name, type_name, v, cumulative=True):
    names = dict(non_generalization="Non-Generalization",
                 generalization="Generalization",
                 ACC=r"$\overline{ACC}$",
                 TPR=r"$\overline{TPR}$",
                 TNR=r"$\overline{TNR}$")
    hist_kws=dict(zorder=1,
                  cumulative=True,
                  density=True,
                  range=(0,1),
                  label=r"{}, {} = {:.3f}".format(names[type_name], names[acc_name], v.mean()))
    kde_kws=dict(cumulative=cumulative)
    return dict(hist_kws=hist_kws, kde_kws=kde_kws)

def draw_accuracies(path):
    df = pd.read_csv(path)
    types_db = df["Type DB"].unique()
    generators = df["Gen"].unique()
    types_top = df["Type Top"].unique()
    accuracies = ["ACC", "TPR", "TNR"]
    dir_name = os.path.dirname(path)

    sns.set_style("ticks")
    for acc_name in accuracies:
        for g in generators:
            dfgen = df[df["Gen"] == g]
            for ttop in types_top:
                dftop = dfgen[dfgen["Type Top"] == ttop]
                fig = plt.figure(dpi=300)
                ax = fig.subplots(1, 1, sharey=False)
                for tdb in types_db:
                    save_dir = os.path.join(dir_name, g, ttop)
                    if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)
                    v = dftop[dftop["Type DB"] == tdb][acc_name].values
                    dist_params = get_dist_params(acc_name, tdb, v)
                    sns.distplot(v, ax=ax, **dist_params)
                ax.set_xlabel(acc_name)
                ax.set_ylabel("Cumulative Frequency")
                ax.legend()
                ax.set_yticks(np.arange(0, 1.25, .25))
                ax.yaxis.grid(True)
                fig.tight_layout()
                plt.savefig(os.path.join(save_dir, acc_name + ".pdf"), transparent=True)
                fig.clear()
                plt.close()

def draw_batch(dataset, path, name, logger=None):
    if not os.path.isdir(path):
        os.makedirs(path)

    loader = DataLoader(dataset, batch_size=3)
    batch = next(iter(loader), None)
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
        logger.info("Save graph input ilustration in {}".format(os.path.join(path, name + ".png")))
    plt.savefig(os.path.join(path, name + ".png"))
    plt.close()
