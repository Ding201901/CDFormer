# -*- coding:utf-8 -*-
# Author:Ding
import platform
import random
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader


def get_pixel_sequence(data, img_height, img_width, channel, patch_size = 5):
    """get pixel sequence from patch"""
    # patch_size:the size of target pixel's neighborhood
    pad_size = int(patch_size / 2)
    # pad zeros on the edge
    data = np.pad(data, [(pad_size, pad_size), (pad_size, pad_size), (0, 0)], mode = 'constant')
    data = np.transpose(data, (2, 0, 1))
    patches = np.empty([img_height * img_width, channel, patch_size, patch_size])  # img_height * img_width
    for i in range(img_height):
        for j in range(img_width):
            patches[i * img_width + j, ...] = data[:, i:i + patch_size, j:j + patch_size]

    # Flat the patches into pixel sequence
    pixel_sequence = np.reshape(patches, (-1, channel, patch_size * patch_size))
    pixel_sequence = pixel_sequence.transpose(0, 2, 1)

    return pixel_sequence


def make_data(dataset: str, patch_size = 5):
    """读取数据——>将数据整理为B×N×b格式——>标准化——>数据类型转换"""
    if platform.system().lower() == 'windows':
        print("[Info]: Running on Windows.")
        path = '.'
    elif platform.system().lower() == 'linux':
        print("[Info]: Running on Linux.")
        path = '..'
    if dataset == 'China':
        data = sio.loadmat(path + '/DataSet/China_Change_Dataset.mat')  # data set X
        data_T1 = 1.0 * data['T1']
        data_T2 = 1.0 * data['T2']
        y = 1.0 * data['Binary']  # the binary ground truth
    elif dataset == 'USA':
        data = sio.loadmat(path + '/DataSet/USA_Change_Dataset.mat')  # data set X
        data_T1 = 1.0 * data['T1']
        data_T2 = 1.0 * data['T2']
        y = 1.0 * data['Binary']  # the binary ground truth

    img_height, img_width, channel = data_T1.shape
    pixel_sequence_T1 = get_pixel_sequence(data_T1, img_height, img_width, channel, patch_size = patch_size)
    pixel_sequence_T2 = get_pixel_sequence(data_T2, img_height, img_width, channel, patch_size = patch_size)

    x_min = min(pixel_sequence_T1.min(), pixel_sequence_T2.min())
    x_max = max(pixel_sequence_T1.max(), pixel_sequence_T2.max())
    pixel_sequence_T1 = (pixel_sequence_T1 - x_min) / (x_max - x_min)
    pixel_sequence_T2 = (pixel_sequence_T2 - x_min) / (x_max - x_min)
    y = np.reshape(y, (-1,))

    # 数据类型与转为tensor后的类型一致
    # pixel_sequence_T1 = pixel_sequence_T1.astype(np.float32)
    # pixel_sequence_T2 = pixel_sequence_T2.astype(np.float32)
    # y = y.astype(np.float32)
    # X_T1, X_T2 = np.split(X, [img_height * img_width], axis = 0)

    return pixel_sequence_T1, pixel_sequence_T2, y


class MyDataset(Dataset):
    def __init__(self, pixel_seq_T1, pixel_seq_T2, y):
        self.pixel_seq_T1 = torch.FloatTensor(pixel_seq_T1)  # FloatTensor change numpy array into tensor.float32
        self.pixel_seq_T2 = torch.FloatTensor(pixel_seq_T2)
        self.y = torch.FloatTensor(y).long().squeeze()
        self.length = len(self.pixel_seq_T1)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.pixel_seq_T1[item], self.pixel_seq_T2[item], self.y[item]


def get_dataloader(dataset, batch_size, patch_size, test_ratio, seed):
    """generate dataloader"""
    pixel_seq_T1, pixel_seq_T2, y = make_data(dataset, patch_size = patch_size)
    # train_index = np.load(f'train_index_{int(int(100.0) - int(100 * test_ratio))}.npy')
    # valid_index = np.delete(np.array([i for i in range(len(pixel_seq_T1))]), train_index)
    # T1_train = pixel_seq_T1[train_index]
    # T1_valid = pixel_seq_T1[valid_index]
    # T2_train = pixel_seq_T2[train_index]
    # T2_valid = pixel_seq_T2[valid_index]
    # y_train = y[train_index]
    # y_valid = y[valid_index]
    T1_train, T1_valid, *_ = train_test_split(pixel_seq_T1, y, test_size = test_ratio, random_state = seed,
                                              stratify = y)
    T2_train, T2_valid, y_train, y_valid = train_test_split(pixel_seq_T2, y, test_size = test_ratio,
                                                            random_state = seed, stratify = y)
    # if not os.path.exists(f'train_index_{int(int(100.0) - int(100 * test_ratio))}.npy'):
    #     train_index = []
    #     for i in range(len(T1_train)):
    #         # train_index.append(np.argwhere(pixel_seq_T1 == T1_train[i])[0][0])
    #         for j in range(len(pixel_seq_T1)):
    #             if (T1_train[i, 12] == pixel_seq_T1[j, 12]).all():
    #                 train_index.append(j)
    #                 break
    #     np.save(f'train_index_{int(100 * (1 - test_ratio))}.npy', train_index)

    # reshape y
    y_train = np.reshape(y_train, (-1, 1))
    y_valid = np.reshape(y_valid, (-1, 1))
    # generate Dataset
    train_set = MyDataset(T1_train, T2_train, y_train)
    valid_set = MyDataset(T1_valid, T2_valid, y_valid)

    train_loader = DataLoader(train_set,
                              batch_size = batch_size,
                              shuffle = True,
                              pin_memory = True,
                              )

    valid_loader = DataLoader(valid_set,
                              batch_size = batch_size,
                              shuffle = True,
                              pin_memory = True,
                              )

    return train_loader, valid_loader


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype = np.float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


# -------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    print('Accuracy :', OA)
    print('Kappa :', Kappa)
    return OA, AA_mean, Kappa, AA


# set experiment seed
def set_seed(seed):
    # python
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
