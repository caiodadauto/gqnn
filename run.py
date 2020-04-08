import os
import argparse
import logging
import logging.handlers as handlers

import torch
import numpy as np
from gqnn import draw_batch, Brite, train, test, QGNN
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

def run(perform, root_path, data_path, type_db, delta_time, seed,
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

    dataset = Brite(data_path, type_db=type_db, logger=file_logger)
    draw_batch(dataset, data_path, logger=file_logger)

    loader = DataLoader(dataset, batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QGNN(out_channels=hidden_size, num_msg=msgs, dropout_ratio=dropout_ratio, packet_loss=packet_loss).to(device)
    if perform == "train":
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=decay)
        train(device, model, loader, optimizer, scheduler, loss_fn, epochs, root_path, threshold, delta_time, class_weight, file_logger)
    else:
        test(device, model, loader, root_path, threshold, file_logger)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Stting environment
    p.add_argument("perform", type=str, choices=["test", "train"])
    p.add_argument("--root-path", type=str, default="assets/",
                   help="Directory where model and optimizer states, figures, and training information will be saved")
    p.add_argument("--data-path", type=str, default="assets/", help="Directory where dataset will be saved")
    p.add_argument("--type-db", type=str, default="train", choices=["train", "test_non_generalization", "test_generalization"],
                   help="Type of dataset")
    p.add_argument("--delta-time", type=float, default=20, help="Log time interval")
    p.add_argument("--seed", type=int, default=2, help="Seed for Pytorch random state")

    # Stting model
    p.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size to be used in the training")
    p.add_argument("--hidden-size", type=int, default=160, help="Latent dimension")
    p.add_argument("--msgs", type=int, default=20, help="Number of messages used for massage passing")
    p.add_argument("--dropout-ratio", type=float, default=.15, help="Probability to make zeros in the dropout's input tensor")
    p.add_argument("--packet-loss", type=float, default=.15, help="Probability of losing packets")
    p.add_argument("--init-lr", type=float, default=.5, help="Initial learning rate")
    p.add_argument("--loss-fn", type=losses, default=torch.nn.functional.binary_cross_entropy, help="Loss function")
    p.add_argument("--threshold", type=float, default=.35, help="Threshold to decided if a link is for routing or not")
    p.add_argument("--decay", type=float, default=.1, help="Dacay ratio for learning rate scheduler")
    p.add_argument("--class-weight", type=weights, default=[1, 1], help="Weights for each class applied to loss function")
    p.add_argument("--milestones", type=interval, default=[500, 2000, 3000, 6000],
                   help="Interval of steps where scheduler will decay the learning rate")
    args = p.parse_args()

    run(**vars(args))
