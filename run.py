import os
import argparse

import torch
import numpy as np
from gqnn import draw_batch, Brite, train, QGNN
from torch_geometric.data import DataLoader


def losses(s):
    losses_opt = dict(bce=torch.nn.BCELoss, mse=torch.nn.MSELoss)
    try:
        loss = losses_opt[str.casefold(s)]()
        return loss
    except:
        raise argparse.ArgumentTypeError("Interval must be mse or bce")

def interval(s):
    try:
        l = map(int, s.split(','))
        return l
    except:
        raise argparse.ArgumentTypeError("Interval must be a sequence of integers splited by commas")

def run(root_path, data_path, type_db, delta_time, seed, debug,
        epochs, batch_size, hidden_size, msgs, dropout_ratio, packet_loss, init_lr, loss_fn, threshold, decay, milestones):
    torch.manual_seed(seed)
    np.random.seed(seed)

    type_db = None if type_db == "" else type_db
    dataset = Brite(data_path, type_db=type_db, debug=debug)

    if debug:
        draw_batch(dataset, data_path)
    loader = DataLoader(dataset, batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QGNN(out_channels=hidden_size, num_msg=msgs, dropout_ratio=dropout_ratio, packet_loss=packet_loss).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=decay)
    train(device, model, loader, optimizer, scheduler, loss_fn, epochs, root_path, threshold, dt=delta_time)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Stting environment
    p.add_argument("--root-path", type=str, default="assets/",
                   help="Directory where model and optimizer states, figures, and training information will be saved")
    p.add_argument("--data-path", type=str, default="assets/", help="Directory where dataset will be saved")
    p.add_argument("--type_db", type=str, default="", choices=["", "train", "test_non_generalization", "test_generalization"],
                   help="Type of dataset")
    p.add_argument("--delta-time", type=float, default=20, help="Log time interval")
    p.add_argument("--seed", type=int, default=2, help="Seed for Pytorch random state")
    p.add_argument("--debug", action="store_true", help="Debugging mode")

    # Stting model
    p.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size to be used in the training")
    p.add_argument("--hidden-size", type=int, default=160, help="Latent dimension")
    p.add_argument("--msgs", type=int, default=20, help="Number of messages used for massage passing")
    p.add_argument("--dropout-ratio", type=float, default=.15, help="Probability to make zeros in the dropout's input tensor")
    p.add_argument("--packet-loss", type=float, default=.15, help="Probability of losing packets")
    p.add_argument("--init-lr", type=float, default=.5, help="Initial learning rate")
    p.add_argument("--loss-fn", type=losses, default=torch.nn.BCELoss(), help="Loss function")
    p.add_argument("--threshold", type=float, default=.35, help="Threshold to decided if a link is for routing or not")
    p.add_argument("--decay", type=float, default=.1, help="Dacay ratio for learning rate scheduler")
    p.add_argument("--milestones", type=interval, default=[500, 2000, 3000, 6000],
                   help="Interval of steps where scheduler will decay the learning rate")
    args = p.parse_args()

    run(**vars(args))
