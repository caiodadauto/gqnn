import os
import argparse

import torch
import numpy as np
from gqnn import draw_batch, Brite, train, QGNN
from torch_geometric.data import DataLoader


def run(root_path, data_path, type_db, batch_size, epochs, hidden_size, msgs, dropout_ratio, delta_time, seed, debug):
    torch.manual_seed(seed)
    np.random.seed(seed)

    type_db = None if type_db == "" else type_db
    dataset = Brite(data_path, type_db=type_db)

    if debug:
        draw_batch(dataset, data_path)
    loader = DataLoader(dataset, batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QGNN(out_channels=hidden_size, num_msg=msgs, dropout_ratio=dropout_ratio).to(device)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 2000, 3000, 6000], gamma=0.1)
    train(device, model, loader, optimizer, scheduler, loss_fn, epochs, root_path, dt=delta_time)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root-path", type=str, default="assets/",
                   help="Directory where model and optimizer states, figures, and training information will be saved")
    p.add_argument("--data-path", type=str, default="assets/", help="Directory where dataset will be saved")
    # p.add_argument("--secrets-path", type=str, default="client_secrets.json", help="Client secrets for drive manipulation")
    # p.add_argument("--version", type=str, default="v1.0", choices=["v1.0", "toy"], help="Verion of dataset that will be used")
    p.add_argument("--type_db", type=str, default="train", choices=["", "train", "test_non_generalization", "test_generalization"],
                   help="Type of dataset")
    # p.add_argument("--id-folder", type=str, default="1DEHJZQC6AFoolUeQqC6NnwPVg0RFZ8iK", help="FolderID with dataset")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size to be used in the training")
    p.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    p.add_argument("--hidden-size", type=int, default=160, help="Latent dimension")
    p.add_argument("--msgs", type=int, default=20, help="Number of messages used for massage passing")
    p.add_argument("--dropout-ratio", type=float, default=.15, help="Probability to make zeros in the dropout's input tensor")
    p.add_argument("--delta-time", type=float, default=20, help="Log time interval")
    p.add_argument("--seed", type=int, default=2, help="Seed for Pytorch random state")
    p.add_argument("--debug", action="store_true", help="Debugging mode")
    args = p.parse_args()

    run(**vars(args))
