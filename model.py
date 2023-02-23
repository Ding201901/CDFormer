# -*- coding:utf-8 -*-
# Author:Ding
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


class PositionEncoding(nn.Module):
    def __init__(self, dim_model, max_seq_len):
        """
        Args:
        dim_model: pe编码维度，一般与word embedding相同，方便相加。表示某个位置的向量的维度
        max_seq_len:语料库中最长句子的长度，即word embedding中的L
        """
        super().__init__()
        self.dim_model = dim_model
        pe = torch.zeros(max_seq_len, dim_model)  # 建立空表，每行代表一个词的位置，每列代表一个编码位
        # 建立一个arrange表示词的位置以便公式计算，size = （max_seq_len,1)
        po = torch.arange(max_seq_len).unsqueeze(1)  # unsqueeze 在指定数组中加入长度为1的维度
        # 计算公式中100000^(2i/dim_model)
        div_term = torch.exp(torch.arange(0., dim_model, 2) * -(math.log(10000.) / dim_model))
        pe[:, 0::2] = torch.sin(po * div_term)  # 计算偶数维度的pe值
        pe[:, 1::2] = torch.cos(po * div_term)  # 计算奇数维度的pe值
        self.register_buffer('pe', pe)  # pe值是不参加训练的

    def forward(self, x):
        """[L, N, dim]"""
        l, *_ = x.shape  # *_表示对_的重复，代表多个_
        x = x + self.pe[:l, :].unsqueeze(1) / math.sqrt(self.dim_model)
        return x


class SegmentEmbedding(nn.Module):
    def __init__(self, seq_len = 2, dim_model = 128):
        super().__init__()
        # self.device = torch.device("cuda：0" if torch.cuda.is_available() else "cpu")
        self.weight = nn.Embedding(seq_len, dim_model).weight.unsqueeze(0)  # .to(self.device)
        self.dim_model = dim_model

    def forward(self, x1, x2):
        _, batch_size, dim_model = x1.shape
        sep = torch.zeros(1, batch_size, dim_model).to(x1.device)
        x1 = torch.cat((x1, sep), 0)
        x1 = x1 + (self.weight[:, 0, :]).to(x1.device) / math.sqrt(self.dim_model)
        x2 = x2 + (self.weight[:, 1, :]).to(x1.device) / math.sqrt(self.dim_model)

        return x1, x2


# Transformer EncoderLayer
class Smoother(nn.Module):
    """Convolutional Transformer EncoderLayer"""

    def __init__(self, dim_model: int, n_head: int, dim_feedforward: int, dropout = 0.1):
        super(Smoother, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, n_head, dropout = dropout)
        self.conv1 = nn.Conv1d(dim_model, dim_feedforward, 5, padding = 2)
        self.conv2 = nn.Conv1d(dim_feedforward, dim_model, 1, padding = 0)

        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,
                src: 'Tensor',
                src_mask: 'Optional[Tensor]' = None,
                src_key_padding_mask: 'Optional[Tensor]' = None) -> 'Tensor':
        # multi-head self attention
        src2 = self.self_attn(src, src, src, attn_mask = src_mask, key_padding_mask = src_key_padding_mask)[0]

        # add & norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # conv1d
        src2 = src.transpose(0, 1).transpose(1, 2)
        src2 = self.conv2(F.relu(self.conv1(src2)))
        src2 = src2.transpose(1, 2).transpose(0, 1)

        # add & norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


# Transformer
class ChangeDetection(nn.Module):
    def __init__(self, band_size = 154, dim_model = 128, n_head = 16, num_layers = 2, n_classes = 2,
                 dropout = 0.1):
        """Args:
        dim_model:每个词的特征维度
        n_classes:变化检测的class数量
        """
        super().__init__()
        # Project the dimension of input's features into dim_model
        self.prenet = nn.Linear(band_size, dim_model)
        self.pe = PositionEncoding(dim_model = dim_model, max_seq_len = 128)
        self.se = SegmentEmbedding(dim_model = dim_model)
        self.encoder_layer = nn.TransformerEncoderLayer(dim_model, n_head, 256)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = num_layers)

        # project the dimension of features from dim_model to classes number
        # self.pred_layer = nn.Linear(dim_model, n_classes)
        self.pred_layer = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.ReLU(),
            nn.Linear(dim_model, n_classes),
        )
        # self.pred_layer = nn.Sequential(
        #     nn.Linear(dim_model, n_classes),
        #     nn.Sigmoid(),
        #     nn.Softmax(dim = 1)
        # )

    def forward(self, pixel_seq_T1, pixel_seq_T2):
        """
        args:
        pixel_aeq_T1:(batch size, length,band_size)
        return: (batch size,n_classes)
        """
        pixel_seq = torch.cat((pixel_seq_T1, pixel_seq_T2), 0)
        # out:(2 * batch size, length, dim_model)
        out = self.prenet(pixel_seq)  # 将pixel sequence 沿 batch size 方向拼接在一起，相当于 weight sharing
        out1, out2 = torch.split(out, pixel_seq_T1.shape[0], 0)
        # out: (length, batch size, dim_model)
        out1 = out1.permute(1, 0, 2)
        out2 = out2.permute(1, 0, 2)
        # Position Embedding
        out1 = self.pe(out1)
        out2 = self.pe(out2)
        # Segment embedding
        out1, out2 = self.se(out1, out2)
        # out:(2 * length, batch size, dim_model)
        out = torch.cat((out1, out2), 0)  # 将pixel sequence 沿pixel len维拼接在一起
        # the encoder layer expect features in the shape of (length, batch size, dim_model).
        out = self.encoder(out)
        # out:(batch size, length,dim_model)
        out = out.transpose(0, 1)
        # mean pooling: (batch size, dim_model)
        stats = out.mean(dim = 1)

        # out: (batch size, n_classes)
        out = self.pred_layer(stats)

        # out = self.sigmoid(out)

        return out


# lr schedule
def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
  initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
  initial lr set in the optimizer.

  Args:
    optimizer (:class:`~torch.optim.Optimizer`):
      The optimizer for which to schedule the learning rate.
    num_warmup_steps (:obj:`int`):
      The number of steps for the warmup phase.
    num_training_steps (:obj:`int`):
      The total number of training steps.
    num_cycles (:obj:`float`, `optional`, defaults to 0.5):
      The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
      following a half-cosine).
    last_epoch (:obj:`int`, `optional`, defaults to -1):
      The index of the last epoch when resuming training.

  Return:
    :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        # warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# model function
def model_fn(batch, model, criterion, device):
    """forward a batch through the model"""
    pixel_seq_T1, pixel_seq_T2, labels = batch
    pixel_seq_T1 = pixel_seq_T1.to(device)
    pixel_seq_T2 = pixel_seq_T2.to(device)
    labels = labels.to(device)

    outs = model(pixel_seq_T1, pixel_seq_T2)

    # loss = criterion(outs.squeeze(1), labels.float())  # BCE
    loss = criterion(outs, labels)  # cross

    # get the class id with higher probability
    preds = outs.argmax(1)  # 对于二分类结果就是0 or 1
    # preds = (torch.round(outs)).int()
    # preds = preds.squeeze(1)
    # compute accuracy
    accuracy = torch.mean((preds == labels).float())
    # accuracy = 1 - torch.mean(torch.abs(outs.squeeze(1) - labels.float()))

    return loss, accuracy


def valid(dataloader, model, criterion, device):
    """validate on validation set"""
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    pbar = tqdm(total = math.ceil(len(dataloader.dataset) / dataloader.batch_size),
                ncols = 0, desc = "Valid", unit = "step")

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, accuracy = model_fn(batch, model, criterion, device)
            running_loss += loss.item()
            running_accuracy += accuracy.item()

        pbar.update()
        pbar.set_postfix(
            loss = f"{running_loss / (i + 1):.4f}",
            accuracy = f"{running_accuracy / (i + 1):.4f}"
        )

    pbar.close()
    model.train()

    return running_accuracy / len(dataloader)
