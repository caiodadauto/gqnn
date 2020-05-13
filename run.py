import os
import json
import argparse
import logging
import logging.handlers as handlers
from pathlib import Path

import torch
import numpy as np
from gqnn import draw_batch, Brite, train, test, QGNN, draw_accuracies
from torch_geometric.data import DataLoader


def losses(s):
    losses_opt = dict(bce=torch.nn.functional.binary_cross_entropy)#, mse=torch.nn.MSELoss)
    try:
        loss = losses_opt[str.casefold(s)]
        return loss
    except:
        raise argparse.ArgumentTypeError("Interval must be bce")

def interval(s):
    try:
        l = list(map(int, s.split(',')))
        return l
    except:
        raise argparse.ArgumentTypeError("Interval must be a sequence of integers splited by commas")

def weights(s):
    try:
        l = list(map(float, s.split(',')))
    except:
        raise argparse.ArgumentTypeError("Class weights must be a sequence of two floats splited by commas")

    if len(l) != 2:
        raise argparse.ArgumentTypeError("Class weights must be a sequence of two floats splited by commas")
    return l

def get_db_size(name):
    path = Path(name + "_raw")
    if not os.path.isdir(path):
        path = Path(name + "_processed")
        size = len(list(path.glob("data*")))
    else:
        size = len(list(path.glob("*.gpickle")))
    return size

def load_dataloader(perform, batch_size, path, logger):
    if perform == "train":
        data_names = [(path, "train", batch_size),
                      (path, "valid_non_generalization", batch_size)]
    else:
        data_names = []
        for generator in ["brite", "zoo"]:
            for type_db in ["non_generalization"]:
                for name_top in ["Star", "H&S", "Ladder"]:
                    root = os.path.join(path, "test_" + generator + "_" + type_db)
                    data_names.append( (root, name_top, 4) )

    loader = {}
    for p, dn, btc in data_names:
        dataset = Brite(p, type_db=dn, logger=logger)
        draw_batch(dataset, p, logger=logger, name=dn + "_sample")
        if perform == "train":
            loader[dn + "_loader"] = DataLoader(dataset, batch_size=btc)
        else:
            loader[p.split("/")[-1] + "_" + dn] = DataLoader(dataset, batch_size=btc)
    return loader

def save_args(args):
    root_path = args["root_path"]
    with open(os.path.join(root_path, "cmd_line_args.txt"), "w") as f:
        json.dump(args, f, default=lambda x: x.__name__, ensure_ascii=False, indent=4)

def run(perform, root_path, data_path, delta_time, seed, new_milestones,
        epochs, batch_size, hidden_size, msgs, dropout_ratio, packet_loss, init_lr, loss_fn, threshold, decay, class_weight, milestones):
    torch.manual_seed(seed)
    np.random.seed(seed)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_logger = logging.getLogger('Logger')
    file_logger.setLevel(logging.DEBUG)
    file_handler = handlers.RotatingFileHandler(
        os.path.join(root_path, 'run.log'), maxBytes=200 * 1024 * 1024, backupCount=1
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_logger.addHandler(file_handler)

    loader = load_dataloader(perform, batch_size, data_path, file_logger)
    device = torch.device('cpu')#'cuda' if torch.cuda.is_available() else 'cpu')
    model = QGNN(out_channels=hidden_size, num_msg=msgs, dropout_ratio=dropout_ratio, packet_loss=packet_loss).to(device)
    if perform == "train":
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=decay)
        train(device=device, model=model, optimizer=optimizer, scheduler=scheduler,
              loss_fn=loss_fn, epochs=epochs, root_path=root_path, threshold=threshold, dt=delta_time,
              class_weight=class_weight, logger=file_logger, new_milestones=new_milestones, **loader)
    else:
        test(device=device, model=model, root_path=root_path, threshold=threshold, logger=file_logger, test_loaders=loader)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Stting environment
    p.add_argument("perform", type=str, choices=["test", "train", "draw"])
    p.add_argument("--root-path", type=str, default="assets/",
                   help="Directory where model and optimizer states, figures, and training information will be saved")
    p.add_argument("--data-path", type=str, default="assets/", help="Directory where dataset will be saved")
    p.add_argument("--delta-time", type=float, default=20, help="Log time interval [IGNORED IN TEST]")
    p.add_argument("--seed", type=int, default=2, help="Seed for Pytorch random state")
    p.add_argument("--new-milestones", action="store_true", help="Indicates the use of a new milestones [IGNORED IN TEST]")

    # Stting model
    p.add_argument("--epochs", type=int, default=1, help="Number of epochs [IGNORED IN TEST]")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size to be used in the training")
    p.add_argument("--hidden-size", type=int, default=160, help="Latent dimension")
    p.add_argument("--msgs", type=int, default=20, help="Number of messages used for massage passing")
    p.add_argument("--dropout-ratio", type=float, default=.15, help="Probability to make zeros in the dropout's input tensor")
    p.add_argument("--packet-loss", type=float, default=.15, help="Probability of losing packets")
    p.add_argument("--init-lr", type=float, default=.5, help="Initial learning rate [IGNORED IN TEST]")
    p.add_argument("--loss-fn", type=losses, default=torch.nn.functional.binary_cross_entropy, help="Loss function [IGNORED IN TEST]")
    p.add_argument("--threshold", type=float, default=.35, help="Threshold to decided if a link is for routing or not")
    p.add_argument("--decay", type=float, default=.1, help="Dacay ratio for learning rate scheduler [IGNORED IN TEST]")
    p.add_argument("--class-weight", type=weights, default=[1, 1], help="Weights for each class applied to loss function [IGNORED IN TEST]")
    p.add_argument("--milestones", type=interval, default=[500, 2000, 3000, 6000],
                   help="Interval of steps where scheduler will decay the learning rate [IGNORED IN TEST]")
    args = p.parse_args()
    save_args(vars(args))

    if args.perform != "draw":
        run(**vars(args))
    else:
        draw_accuracies("assets/stats/accuracies.csv")
