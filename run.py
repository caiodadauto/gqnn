import os
import argparse

import torch
from gqnn import draw_batch, Brite, train, QGNN
from torch_geometric.data import DataLoader


def run(epochs, batch_size, hidden_size, msgs, dropout_ratio, path, debug, delta_time):
    dataset = Brite(root=path, size=128, n_interval=(8,20))
    if debug:
        draw_batch(dataset, path)
    loader = DataLoader(dataset, batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QGNN(out_channels=hidden_size, num_msg=msgs, dropout_ratio=dropout_ratio).to(device)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 2000, 3000, 6000], gamma=0.1)
    train(device, model, loader, optimizer, scheduler, loss_fn, epochs, path, dt=delta_time)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root-path", type=str, default="assets/",
                   help="Directory where model and optimizer states, figures, training information, and dataset will be saved")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size to be used in the training")
    p.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    p.add_argument("--hidden-size", type=int, default=160, help="Latent dimension")
    p.add_argument("--msgs", type=int, default=20, help="Number of messages used for massage passing")
    p.add_argument("--dropout-ratio", type=float, default=.15, help="Probability to make zeros in the dropout's input tensor")
    p.add_argument("--delta-time", type=float, default=20, help="Log time interval")
    p.add_argument("--seed", type=int, default=2, help="Seed for Pytorch random state")
    p.add_argument("--debug", action="store_true", help="Debugging mode")
    args = p.parse_args()

    run(args.epochs, args.batch_size, args.hidden_size, args.msgs, args.dropout_ratio, args.root_path, args.debug, args.delta_time)
