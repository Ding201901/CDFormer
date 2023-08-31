# -*- coding:utf-8 -*-
# Author:Ding
import os
import math
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

import torch
import torch.nn as nn
from torch.optim import AdamW

from utils import get_dataloader, set_seed, make_data, output_metric
from model import ChangeDetection, get_cosine_schedule_with_warmup, model_fn, valid


def main(dataset, lr, n_head, n_layers, batch_size, patch_size, epochs, n_classes, warmup_steps, save_epochs,
         test_ratio, out_dir, search_num, seed):
    """main function"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    train_loader, valid_loader = get_dataloader(dataset, batch_size, patch_size, test_ratio, seed)
    print(f"[Info]: Finish loading data!", flush = True)

    mymodel = ChangeDetection(n_head = n_head, num_layers = n_layers, n_classes = n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(mymodel.parameters(), lr = lr)
    total = math.ceil(len(train_loader.dataset) / batch_size)
    schedule = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total)
    print(f"[Info]: Finish creating mymodel!", flush = True)

    best_accuracy = -1.0
    best_state_dict = None

    for epoch in range(epochs):
        # get data
        acc_train = 0
        pbar = tqdm(total = total, ncols = 0, desc = f"{search_num}-{epoch + 1} Train", unit = "step")
        for idx, batch in enumerate(train_loader):
            loss, accuracy = model_fn(batch, mymodel, criterion, device)
            batch_loss = loss.item()
            batch_accuracy = accuracy.item()
            acc_train += batch_accuracy

            # updata mymodel
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # log
            pbar.update()
            pbar.set_postfix(
                loss = f"{batch_loss:.4f}",
                accuracy = f"{batch_accuracy:.4f}",
                step = idx + 1,
            )
        schedule.step()

        # do validation
        pbar.close()

        valid_accuracy = valid(valid_loader, mymodel, criterion, device)

        # keep the best mymodel
        if valid_accuracy * 0.80 + (acc_train / total) * 0.20 > best_accuracy:
            best_accuracy = valid_accuracy * 0.80 + (acc_train / total) * 0.20
            best_state_dict = mymodel.state_dict()

        # Save the best mymodel so far.
        if (epoch + 1) % save_epochs == 0 and best_state_dict is not None:
            # path = str(eval) + "_" + save_path
            if not os.path.exists('./checkpoints'):
                os.mkdir('./checkpoints')
            save_path = f"./checkpoints/{dataset}-{search_num}_model_parameter.pkl"
            torch.save(best_state_dict, save_path)
            pbar.write(f"Step {epoch + 1}, best mymodel saved. (accuracy={best_accuracy:.4f})")
    # test
    h, w = {'China': [420, 140], 'USA': [307, 241]}[dataset]

    if os.path.exists(f"./checkpoints/{dataset}-{search_num}_model_parameter.pkl"):
        mymodel.load_state_dict(torch.load(f"./checkpoints/{dataset}-{search_num}_model_parameter.pkl"))
        mymodel.eval()
        pixel_seq_T1, pixel_seq_T2, y = make_data(dataset, patch_size = patch_size)
        pixel_seq_T1 = torch.FloatTensor(pixel_seq_T1).to(device)  # FloatTensor change numpy array into tensor.float32
        pixel_seq_T2 = torch.FloatTensor(pixel_seq_T2).to(device)
        y = torch.FloatTensor(y).squeeze().to(device)

        CM = []
        for i in range(h):
            T1 = pixel_seq_T1[i * w:(i + 1) * w, ...]
            T2 = pixel_seq_T2[i * w:(i + 1) * w, ...]
            label = y[i * w:(i + 1) * w, ...]
            outs = mymodel(T1, T2)
            preds = outs.argmax(1)  # 对于二分类结果就是0 or 1
            CM.append(preds.cpu().numpy())

        acc, aa_mean, Kappa, aa = output_metric(y.cpu().detach().numpy(), np.reshape(np.array(CM), (-1,)))

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        plt.imshow(CM, cmap = 'gray')
        plt.axis('off')
        plt.savefig(f'{out_dir}/{dataset}-{search_num}.pdf', bbox_inches = 'tight', pad_inches = -0.1)
        plt.show()
        sio.savemat(f'{out_dir}/{dataset}-{search_num}.mat', {'CM': CM})

        return acc, Kappa

    else:
        return -1, -1
    # return best_accuracy, A


def parse_args(num):
    param_grid = {
        "n_head": [2],  # , 4, 8, 16
        "n_layers": [4],  # 2, 4, 5, 6, 7, 8
        "batch_size": [32],  # , 32, 64
        "patch_size": [7],  # 7
        "test_ratio": [0.97],
        "seed": [2024],
    }
    hyperparams = list(ParameterGrid(param_grid))[num]
    config = {
        "dataset": "USA",
        "lr": 1e-4,
        "epochs": 50,
        "n_classes": 4,
        "warmup_steps": 0,
        "save_epochs": 10,
        "out_dir": './exp_result'
    }
    config.update(hyperparams)
    return hyperparams, config


if __name__ == "__main__":
    max_search = 1
    results = []
    for search in range(0, max_search):
        params, config = parse_args(search)
        set_seed(config['seed'])
        acc, kappa = main(search_num = search + 1, **config)
        params['search_num'] = search + 1
        temp = list(params.values())
        temp.append(acc)
        temp.append(kappa)
        results.append(temp)
        np.savetxt('hyperparams_acc.txt', np.array(results), fmt = "%i, %i, %i, %i, %i, %.2f, %i, %.4f, %.4f")


