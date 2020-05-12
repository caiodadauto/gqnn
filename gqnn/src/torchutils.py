import re
import os
import time

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score

from .drawing import draw_accuracies


NAME_INFO_FILE = "info"
NAME_ENV_FILE = "env_state"
NAME_ACC_FILE = "accuracies"

NAME_STATS_DIR = "stats"
NAME_BKP_DIR = "checkpoint"


def save_model(model, optimizer, scheduler, loss, acc, n_batch, epoch, duration, path, best):
    if not os.path.isdir(path):
        os.makedirs(path)

    env_state = {'duration': duration,
                 'epoch': epoch,
                 'n_batch':n_batch,
                 'loss': loss,
                 'acc': acc,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict()}
    if best:
        torch.save(env_state, os.path.join(path, NAME_ENV_FILE + "_best.pt"))
    torch.save(env_state, os.path.join(path, NAME_ENV_FILE + "_last.pt"))

def save_info(n_epoch, n_batch, duration, loss, acc_train, acc_valid, path):
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(os.path.join(path, NAME_INFO_FILE + ".csv"), "a") as f:
        f.write("{:d},{:d},{:.2f},{:.5f},{:.5f},{:.5f}\n".format(n_epoch, n_batch, duration, loss, acc_train, acc_valid))

def is_previous_trainig(path):
    last_file =  os.path.isfile(os.path.join(path, NAME_ENV_FILE + "_last.pt"))
    best_file =  os.path.isfile(os.path.join(path, NAME_ENV_FILE + "_best.pt"))
    if last_file and best_file:
        return True
    return False

def state2str(cp, path):
    s = ""
    r = re.compile(r".*state.*")
    root_path = path.split("/")[0]
    for key, value in cp.items():
        if r.match(key):
            s += "        {}:  {}\n".format(key, "State saved in " + os.path.join(root_path, key))
            with open(os.path.join(root_path, key), "w") as f:
                f.write("{}".format(value))
        else:
            s += "        {}:  {}\n".format(key, value)
    return s

def load_model(model, path, optimizer=None, scheduler=None, test=False, logger=None):
    best_cp = torch.load(os.path.join(path, NAME_ENV_FILE + "_best.pt"))
    best_acc = best_cp['acc']
    best_loss = best_cp['loss']

    last_cp = torch.load(os.path.join(path, NAME_ENV_FILE + "_last.pt"))
    last_epoch = last_cp['epoch']
    last_batch = last_cp['n_batch']
    last_duration = last_cp['duration']

    if test:
        model.load_state_dict(best_cp['model_state_dict'])
        if logger:
            logger.info("Best model loaded:\n" + state2str(best_cp, path))
    else:
        if not optimizer:
            raise ValueError("Need an initialized optimizer to load model")
        model.load_state_dict(last_cp['model_state_dict'])
        optimizer.load_state_dict(last_cp['optimizer_state_dict'])
        if scheduler:
            try:
                scheduler.load_state_dict(last_cp['scheduler_state_dict'])
            except KeyError:
                if logger:
                    logger.info("There is not 'scheduler_state_dict' key in the saved state")
        if logger:
            logger.info("Last model loaded:\n" + state2str(last_cp, path))
        return last_batch, last_epoch, last_duration, best_acc, best_loss

@torch.enable_grad()
def train_step(model, data, optimizer, loss_fn, threshold, class_weight):
    optimizer.zero_grad()
    output = model(data)
    label = data.y
    loss_weight = ( ( ( label - 1 ) * -1 ) * class_weight[0] ) + ( label * class_weight[1] )
    loss = loss_fn(output, label, weight=loss_weight)
    loss.backward()
    optimizer.step()
    label_np = data.y.detach().cpu().numpy()
    output_np = output.detach().cpu().numpy()
    return loss.item(), np.array(output_np > threshold, dtype=int), label_np

@torch.no_grad()
def model_eval(model, data, threshold):
    output = model(data)
    label_np = data.y.detach().cpu().numpy()
    output_np = output.detach().cpu().numpy()
    return np.array(output_np > threshold, dtype=int), label_np

def train(device, model, train_loader, valid_non_generalization_loader, optimizer, scheduler, loss_fn, epochs,
          root_path, threshold=.35, dt=20, class_weight=[1., 1.], logger=None, new_milestones=False):
    n_batch = 0
    n_epoch = 0
    best_acc = 0
    last_batch = 0
    time_offset = 0
    best_loss = np.Inf
    path = os.path.join(root_path, NAME_BKP_DIR)

    if is_previous_trainig(path):
        if new_milestones:
            if logger:
                logger.info("The saved scheduler state will be ignored. New milestones are setted.")
            last_batch, n_epoch, time_offset, best_acc, best_loss = load_model(model, path, optimizer, logger=logger)
        else:
            last_batch, n_epoch, time_offset, best_acc, best_loss = load_model(model, path, optimizer, scheduler, logger=logger)
        if logger:
            logger.info("Up to this trainig point, the best achieved acc and loss were {:.4f} and {:.4f}, respectively".format(
                best_acc, best_loss))

    model.train()
    start_time = time.time()
    last_log_time = start_time
    valid_data = next(iter(valid_non_generalization_loader), None)
    for epoch in range(n_epoch, epochs):
        for data in tqdm(train_loader):
            if n_batch > last_batch:
                data = data.to(device)
                loss, output, label = train_step(model, data, optimizer, loss_fn, threshold, class_weight)
                current_time = time.time()
                if current_time - last_log_time > dt:
                    model.eval()
                    last_log_time = current_time
                    duration = current_time - start_time + time_offset
                    val_output, val_label = model_eval(model, valid_data, threshold)
                    acc_train = balanced_accuracy_score(label, output)
                    acc_valid = balanced_accuracy_score(val_label, val_output)
                    if best_acc < acc_valid:
                        best_acc = acc_valid
                        best = True
                    else:
                        best = False
                    save_model(model, optimizer, scheduler, loss, acc_valid, n_batch, epoch, duration, path, best)
                    save_info(epoch, n_batch, duration, loss, acc_train, acc_valid, path)
                    model.train()
                scheduler.step()
            n_batch += 1
        n_batch = 0
        last_batch = 0

def split_per_graphs(x, n_inter):
    cum_inter = np.cumsum(n_inter)
    x_per_graphs = [x[(n - n_inter[i]):n] for i, n in enumerate(cum_inter)]
    return x_per_graphs

def true_positive_rate(expect, predict):
    mask = expect == 1
    return np.mean(expect[mask] == predict[mask])

def true_negative_rate(expect, predict):
    mask = expect == 0
    return np.mean(expect[mask] == predict[mask])

def get_stats(model, data, threshold, name):
    output = model(data)
    label_np = data.y.detach().cpu().numpy()
    output_np = output.detach().cpu().numpy()
    output_np = np.array(output_np > threshold, dtype=int)
    num_interfaces_np = data.num_interfaces.detach().cpu().numpy()

    stats = []
    label_per_graphs = split_per_graphs(label_np, num_interfaces_np)
    output_per_graphs = split_per_graphs(output_np, num_interfaces_np)
    for expect, predict in zip(label_per_graphs, output_per_graphs):
        acc = balanced_accuracy_score(expect, predict)
        tpr = true_positive_rate(expect, predict)
        tnr = true_negative_rate(expect, predict)
        stats.append([name, acc, tpr, tnr])
    return pd.DataFrame(stats, columns=["Type", "ACC", "TPR", "TNR"])

def save_stats(df_stats, path):
    name_file = os.path.join(path, NAME_ACC_FILE + ".csv")
    if not os.path.isdir(path):
        os.makedirs(path)
    else:
        if os.path.isfile(name_file):
            df = pd.read_csv(name_file, header=0, index_col=False)
            df_stats = pd.concat([df, df_stats], ignore_index=True)
    df_stats.to_csv(name_file, index=False, header=True)

def test(device, model, test_generalization_loader, test_non_generalization_loader, root_path, threshold=.35, logger=None):
    bkp_path = os.path.join(root_path, NAME_BKP_DIR)
    stats_path = os.path.join(root_path, NAME_STATS_DIR)

    load_model(model, bkp_path, test=True, logger=logger)

    model.eval()
    for name, loader in [("non_generalization", test_non_generalization_loader), ("generalization", test_generalization_loader)]:
        data = next(iter(loader))
        data = data.to(device)
        df_stats = get_stats(model, data, threshold, name)
        save_stats(df_stats, stats_path)
    draw_accuracies(os.path.join(stats_path, NAME_ACC_FILE + "csv"))
