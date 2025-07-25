import gc
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from torch import nn
from torchmetrics import PearsonCorrCoef
from argparse import ArgumentParser
import warnings
import shutil
import copy
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytorch_lightning as pl
from torch.nn import GELU

from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint,
                                         StochasticWeightAveraging)
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import SimpleProfiler

warnings.filterwarnings("ignore")

cpu_num = 10
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
torch.autograd.set_detect_anomaly(True)


# 超参数设置
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=15)  # 最大轮数
    parser.add_argument('--batch_size', type=int, default=32)  # batch中样本含几天
    parser.add_argument('--gpus', default=[3])
    parser.add_argument('--strategy', default='ddp')
    parser.add_argument('--find_unused_parameters', default=False)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--check_test_every_n_epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)  # 学习率
    parser.add_argument('--weight_decay', type=float, default=1e-2)  # weight_decay
    parser.add_argument('--seed', type=int, default=3253)  # 随机种子
    parser.add_argument('--optimizer', default='adam',
                        choices=['adam', 'adamw'])
    parser.add_argument('--log_every_n_steps', type=int, default=1)
    parser.add_argument('--loss', default='return_ic')  # 损失函数
    parser.add_argument('--early_stop', action='store_true')  # 早停设置
    parser.add_argument('--swa', action='store_true')  # swa设置
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--checkpoint', help='path to checkpoints (for test)')
    parser.add_argument('--dropout', action='store_true', help='Enable dropout')
    parser.add_argument('--dropout_rate', type=float, default=0.4, help='Dropout rate')
    args, unknown = parser.parse_known_args()
    return args


args = parse_args()

# 模型文件存储目录
root_path = rf'/home/user79/model_train_0604'
# 数据存储目录（share_1, share_2等）
data_path = rf'/project/model_share/share_1'
# 因子
fac_path = rf'{data_path}/factor_data'
fac_name = rf'fac20250212'
# 标签
label_path = rf'{data_path}/label_data'
label_name = rf'label1'
# 流动性数据
liquid_path = rf'{data_path}/label_data'
liquid_name = rf'can_trade_amt1'


class params:
    model_path = rf'{root_path}/model_train'
    profiler_path = rf'{root_path}/logs'
    # 模型前缀
    model_prefix = rf'nn'
    liquid_data = pd.read_feather(rf"{liquid_path}/{liquid_name}.fea").set_index("index")
    ret_data = pd.read_feather(rf"{label_path}/{label_name}.fea").set_index("index")
    # 是否使用dropout
    dropout = True
    # dropout率
    dropout_rate = 0.4
    # 因子标准化方法
    normed_method = 'zscore'


def get_basic_name():
    name = rf'{params.model_prefix}--{fac_name}--{label_name}'
    if params.dropout:
        name += rf'--dropout{params.dropout_rate}'
    return name


# 因子标准化
def normed_data(data, date, stage, normed_method=params.normed_method):
    data = data.reset_index()
    data = data.reindex(data[params.factor_list].dropna(thresh=int(0.1 * len(params.factor_list))).index)
    data = data.set_index('date')
    liquid_data = params.liquid_data.loc[date]
    data['liquid'] = liquid_data.reindex(data["Code"]).values
    ret_data = params.ret_data.loc[date]
    data['Label'] = ret_data.reindex(data["Code"]).values
    if stage == "train":
        data['Label'] = (data['Label'] - data['Label'].mean()) / data['Label'].std()
    data['Label'] = data['Label'].fillna(0)
    code_value = data['Code'].values
    data_X = data.drop(['Code', 'Label', 'liquid'], axis=1)
    data_y = data['Label']
    data_liquid = data['liquid']

    if normed_method == 'zscore':
        data_X = np.clip(data_X, data_X.quantile(0.005), data_X.quantile(0.995), axis=1)
        data_X = (data_X - data_X.mean()) / data_X.std()
        data_X = data_X.fillna(0)

    else:
        raise NotImplementedError

    return torch.tensor(data_X.values, dtype=torch.float32), \
        torch.tensor(data_y.values, dtype=torch.float32), \
        code_value, torch.tensor(data_liquid.values, dtype=torch.float32)


def collate_fn(datas):
    data_X = [temp for data in datas for temp in data[0]]
    data_y = [temp for data in datas for temp in data[1]]
    data_time = [temp for data in datas for temp in data[2]]
    code_value = [temp for data in datas for temp in data[3]]
    data_liquid = [temp for data in datas for temp in data[4]]
    return data_X, data_y, data_time, code_value, data_liquid


class DLDataset(torch.utils.data.Dataset):

    def __init__(self, date_list, stage='train'):
        self.date_list = date_list
        self.stage = stage

    def __getitem__(self, index):
        date = self.date_list[index]
        if date == 'out_sample':
            return ['out_sample'], ['out_sample'], ['out_sample'], ['out_sample'], ['out_sample']
        data = params.all_data.loc[date].copy()
        train_X = []
        train_y = []
        train_time = []

        train_code_value = []
        train_liquid = []
        data_X, data_y, code_value, data_liquid = normed_data(data, date, stage=self.stage)
        train_X.append(torch.nan_to_num(data_X, nan=0, posinf=0, neginf=0))
        train_y.append(torch.nan_to_num(data_y, nan=0, posinf=0, neginf=0))
        train_liquid.append(torch.nan_to_num(data_liquid, nan=0, posinf=0, neginf=0))
        train_time.append(date)
        train_code_value.append(code_value)

        return train_X, train_y, train_time, train_code_value, train_liquid

    def __len__(self):
        return len(self.date_list)


'''
train_X
[
    tensor(shape=[3, 4]),  # 第一天：3只股票，4个因子
    tensor(shape=[3, 4])   # 第二天：3只股票，4个因子
]

data_y
[
    tensor(shape=[3]),  # 第一天：3只股票的标签
    tensor(shape=[3])   # 第二天：3只股票的标签
]

data_time
['20230101', '20230102']  # 两个日期字符串

code_value
[
    ['AAPL', 'MSFT', 'GOOGL'],  # 第一天的股票代码
    ['AAPL', 'MSFT', 'GOOGL']   # 第二天的股票代码（假设相同）
]

data_liquid
[
    tensor(shape=[3]),  # 第一天：3只股票的流动性
    tensor(shape=[3])   # 第二天：3只股票的流动性
]
'''


# 训练数据加载器
class DLDataModule(pl.LightningDataModule):
    def __init__(self, args, train_date_list, valid_date_list, test_date_list):
        super().__init__()
        self.args = args
        self.tr = DLDataset(train_date_list, stage='train')
        self.val = DLDataset(valid_date_list, stage='valid')
        self.test = DLDataset(test_date_list, stage='test')

    def train_dataloader(self):
        return DataLoader(self.tr, batch_size=self.args.batch_size, collate_fn=collate_fn,
                          num_workers=4, shuffle=True,
                          persistent_workers=False,
                          drop_last=False,
                          pin_memory=False)

    def _val_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=1, collate_fn=collate_fn,
                          num_workers=0, persistent_workers=False,
                          pin_memory=False, drop_last=False)

    def val_dataloader(self):
        return self._val_dataloader(self.val)

    def test_dataloader(self):
        return self._val_dataloader(self.test)


# 损失函数设置
# 计算加权pearson系数并转为损失函数
def get_loss_fn(loss):
    def wpcc(preds, y):  #模型预测值和真实标签
        _, argsort = torch.sort(preds, descending=True, dim=0)
        argsort = argsort.squeeze()
        weight = torch.zeros(preds.shape, device=preds.device)
        weight_new = torch.tensor([0.5 ** ((i - 1) / (preds.shape[0] - 1)) for i in range(1, preds.shape[0] + 1)]
                                  , device=preds.device).unsqueeze(dim=1)

        weight[argsort, :] = weight_new
        wcov = (preds * y * weight).sum(dim=0) / weight.sum(dim=0) - (
                (preds * weight).sum(dim=0) / weight.sum(dim=0)) * ((y * weight).sum(dim=0) / weight.sum(dim=0))
        pred_std = torch.sqrt(((preds - preds.mean(dim=0)) ** 2 * weight).sum(dim=0) / weight.sum(dim=0))
        y_std = torch.sqrt(((y - y.mean(dim=0)) ** 2 * weight).sum(dim=0) / weight.sum(dim=0))

        return -(wcov / (pred_std * y_std)).mean()

    def return_ic_loss(preds, y, liquid, top_k=500):
        preds = preds.squeeze()
        y = y.squeeze()
        liquid = liquid.squeeze()

        # 预测排序
        _, indices = torch.sort(preds, descending=True)
        preds_sorted = preds[indices]
        y_sorted = y[indices]
        liquid_sorted = liquid[indices]

        # 超额收益（限制最多买入 top_k）
        hold = torch.minimum(liquid_sorted[:top_k], torch.tensor(1e8, device=preds.device))  # 限每只股票持仓
        total_money = hold.sum() + 1e-8
        total_earned = (y_sorted[:top_k] * hold).sum()
        return_component = total_earned / total_money

        # Pearson IC
        ic = torch.corrcoef(torch.stack([preds, y]))[0, 1]

        # 最终损失
        return -(50 * return_component + ic)

    def output(loss):
        return {
            'wpcc': wpcc,
            'return_ic':return_ic_loss

        }[loss]

    return output(loss)


# 模型设置
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        # 线性投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 分割多头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力得分
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # 输出投影
        output = self.out_proj(output)
        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.0, use_layernorm=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)  # 主路径：输入维度 → 输出维度
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.norm = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()

        # **关键修改：残差连接的投影层**
        if in_dim != out_dim:
            # 当输入和输出维度不同时，用线性层调整残差连接的维度
            self.projection = nn.Linear(in_dim, out_dim)
        else:
            # 维度相同时，直接使用恒等映射
            self.projection = nn.Identity()

    def forward(self, x):
        residual = self.projection(x)  # 调整残差维度（关键步骤）
        out = self.linear(x)  # 主路径：线性变换
        out = self.activation(out)  # 激活函数
        out = self.dropout(out)  # Dropout
        out = self.norm(out + residual)  # 残差相加 + 层归一化
        return out

class SEAttention(nn.Module):
    """ 通道注意力模块（Squeeze-and-Excitation） """
    def __init__(self, embed_dim, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化压缩维度
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // reduction),  # 压缩
            nn.LeakyReLU(),
            nn.Linear(embed_dim // reduction, embed_dim),  # 恢复
            nn.Sigmoid()  # 输出0-1的通道权重
        )

    def forward(self, x):
        # x shape: [num_stocks, embed_dim]
        batch_size, embed_dim = x.size()
        
        # 生成与输入维度匹配的通道权重
        x_pool = self.avg_pool(x.unsqueeze(-1)).squeeze(-1)  # [batch_size, embed_dim]
        weights = self.fc(x_pool)  # [batch_size, embed_dim]
        
        # 应用权重（关键修改：直接相乘，保持2D形状）
        return x * weights  # [batch_size, embed_dim]

    
class PredictModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 1. 定义跨尺度维度（小→中→大，可根据因子数量调整）
        self.scale_dims = [128, 256, 512]  # 三个尺度的维度
        in_dim = args.factor_num  # 输入因子维度

        # 2. 输入层：将原始因子映射到最小尺度（128维）
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, self.scale_dims[0]),
            nn.LeakyReLU(),
            nn.Dropout(p=args.dropout_rate) if args.dropout else nn.Identity(),
            nn.LayerNorm(self.scale_dims[0])  # 层归一化稳定训练
        )

        # 3. 跨尺度交互模块：残差融合（小→中→大）
        self.interaction1 = ResidualBlock(  # 128 → 256
            in_dim=self.scale_dims[0],
            out_dim=self.scale_dims[1],
            dropout_rate=args.dropout_rate,
            use_layernorm=True
        )
        self.interaction2 = ResidualBlock(  # 256 → 512
            in_dim=self.scale_dims[1],
            out_dim=self.scale_dims[2],
            dropout_rate=args.dropout_rate,
            use_layernorm=True
        )

        # 4. 大尺度特征增强：多头自注意力（捕捉大尺度内的细粒度交互）
        self.attention = MultiHeadSelfAttention(
            embed_dim=self.scale_dims[2],
            num_heads=16,  # 16头，需保证 embed_dim % num_heads == 0（512÷16=32，符合）
            dropout=args.dropout_rate
        )
        self.norm_attention = nn.LayerNorm(self.scale_dims[2])
        self.dropout_attention = nn.Dropout(args.dropout_rate) if args.dropout else nn.Identity()

        # 5. 多尺度特征拼接 + 融合层（压缩维度 + 残差）
        concat_dim = sum(self.scale_dims)  # 128+256+512=896
        self.fusion_proj = ResidualBlock(
            in_dim=concat_dim,
            out_dim=512,  # 压缩到512维，平衡复杂度和表达力
            dropout_rate=args.dropout_rate,
            use_layernorm=True
        )

        # 6. 通道注意力：增强关键通道的特征
        self.se_attention = SEAttention(embed_dim=512, reduction=8)

        # 7. 输出层：从512维映射到预测值（1维）
        self.output_layers = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(p=args.dropout_rate) if args.dropout else nn.Identity(),
            nn.Linear(128, 1)
        )

    def forward(self, tsdata):
        x = tsdata.float()  # 输入形状：[num_stocks, factor_num]（单日股票因子矩阵）

        # Step 1：输入层 → 小尺度（128维）
        x_small = self.input_proj(x)  # [num_stocks, 128]

        # Step 2：跨尺度交互1 → 中尺度（256维，残差融合小尺度信息）
        x_medium = self.interaction1(x_small)  # [num_stocks, 256]

        # Step 3：跨尺度交互2 → 大尺度（512维，残差融合中尺度信息）
        x_large = self.interaction2(x_medium)  # [num_stocks, 512]

        # Step 4：大尺度内注意力增强（添加batch维度适配MultiHeadSelfAttention）
        x_attn = x_large.unsqueeze(0)  # [1, num_stocks, 512]（batch_size=1）
        x_attn = self.attention(x_attn)
        x_attn = x_attn.squeeze(0)  # [num_stocks, 512]
        x_large = x_large + self.dropout_attention(x_attn)
        x_large = self.norm_attention(x_large)

        # Step 5：多尺度特征拼接（小、中、大尺度特征并联）
        x_concat = torch.cat([x_small, x_medium, x_large], dim=-1)  # [num_stocks, 896]

        # Step 6：融合层压缩 + 残差
        x_fusion = self.fusion_proj(x_concat)  # [num_stocks, 512]

        # Step 7：通道注意力加权
        x_se = self.se_attention(x_fusion)  # [num_stocks, 512]

        # Step 8：输出预测
        out = self.output_layers(x_se)  # [num_stocks, 1]
        return out


# 训练，验证和测试步骤
class DLLitModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = PredictModel(args)
        self.test_pearson = PearsonCorrCoef()
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, *args):
        return self.model(*args)

    def training_step(self, batch, batch_idx):
        tsdatas, rets, times, code_values, liquids = batch
        loss_fn = get_loss_fn(self.args.loss)
        loss_list = []

        for i in range(len(tsdatas)):
            tsdata, ret, liquid = tsdatas[i], rets[i], liquids[i]
            ret = ret.unsqueeze(1)
            preds = self.forward(tsdata)
            loss = loss_fn(preds, ret, liquid)
            loss_list.append(loss)

        total_loss = sum(loss_list) / len(loss_list)
        self.log('train_loss', total_loss)
        return total_loss

    def _evaluate_step(self, batch, batch_idx, stage):

        def get_excess_return(preds, ret, liquid, money):
            _, sort = preds.sort(dim=0, descending=True, stable=True)
            sort = sort.squeeze(1)
            total_hold = torch.tensor(0.0, device=preds.device)
            total_earned = torch.tensor(0.0, device=preds.device)
            for num, idx in enumerate(sort):
                if num >= 500:
                    break
                if (money - total_hold) < 1:
                    break
                hold_money = min(money - total_hold, liquid[idx])
                total_hold += hold_money
                total_earned += ret[idx] * hold_money
            total_ret = total_earned / money
            return total_ret

        excess_return_list = []
        ic_list = []
        tsdatas, rets, times, code_values, liquids = batch
        model_cpu = copy.deepcopy(self.model)
        model_cpu = model_cpu.to('cpu')
        model_cpu.eval()
        save_model = False
        for i in range(len(tsdatas)):
            tsdata, ret, time, code_value, liquid = tsdatas[i], rets[i], times[i], code_values[i], liquids[i]
            if tsdata == 'out_sample':
                if not save_model:
                    try:
                        torch.jit.script(model_cpu).save(f"{params.model_name}.pth")
                        save_model = True
                    except:
                        pass
            else:
                preds = self.forward(tsdata)
                if stage == "test":
                    data_cpu = tsdata.to('cpu')
                    preds_cpu = model_cpu.forward(data_cpu)
                    res = pd.DataFrame(preds_cpu.cpu().detach().numpy(), index=code_value, columns=['value'])
                    res.index.name = 'Code'
                    res.to_pickle(f'{params.test_save_path}/{time}.pkl')
                    if not save_model:
                        try:
                            torch.jit.script(model_cpu).save(f"{params.model_name}.pth")
                            save_model = True
                        except:
                            pass

                excess_return = get_excess_return(preds, ret, liquid, money=1e9)
                excess_return_list.append(excess_return)
                ic_list.append(self.test_pearson(preds.squeeze(1), ret))
        try:
            res_list = [sum(excess_return_list) / len(excess_return_list), sum(ic_list) / len(ic_list)]
        except:
            res_list = [np.nan, np.nan]
        if stage == 'val':
            self.validation_step_outputs.append(res_list)
        if stage == "test":
            self.test_step_outputs.append(res_list)
        return res_list

    def test_step(self, batch, batch_idx):
        return self._evaluate_step(batch, batch_idx, 'test')

    def validation_step(self, batch, batch_idx):
        return self._evaluate_step(batch, batch_idx, 'val')

    def on_validation_epoch_end(self):
        val_step_outputs = self.validation_step_outputs
        num_batch = len(val_step_outputs)
        self.log('val_wei', sum([(data[0] * 50 + data[1]) for data in val_step_outputs]) / num_batch, prog_bar=True,
                 sync_dist=True)
        self.validation_step_outputs.clear()
        gc.collect()

    def on_test_epoch_end(self):
        test_step_outputs = self.test_step_outputs
        num_batch = len(test_step_outputs)
        self.log('test_wei', sum([data[0] for data in test_step_outputs]) / num_batch, prog_bar=True, sync_dist=True)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        kwargs = {
            'lr': self.args.lr,
            'weight_decay': self.args.weight_decay,
        }

        optimizer = {
            'adam': torch.optim.Adam(self.model.parameters(), **kwargs),
            'adamw': torch.optim.AdamW(self.model.parameters(), **kwargs),
        }[self.args.optimizer]

        # 使用 StepLR 学习率调度器
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)  # 每10个epoch衰减0.7

        # 返回优化器和调度器
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # 'epoch' 或 'step'
                'frequency': 1,       # 每个epoch执行一次
            },
        }

    def configure_callbacks(self):
        callbacks = [
            LearningRateMonitor(),
            ModelCheckpoint(monitor='val_wei', mode='max', save_top_k=8, save_last=False,
                            filename='{epoch}-{val_wei:.4f}')
        ]
        if self.args.swa:
            callbacks.append(StochasticWeightAveraging(swa_epoch_start=0.8,
                                                       device='gpu'))
        if self.args.early_stop:
            callbacks.append(EarlyStopping(monitor='val_wei',
                                           mode='max', patience=30))
        return callbacks


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump_pickle(path, data):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


# 模型训练
def train_single(args, name, seed, train_date_list, valid_date_list, test_date_list):
    params.factor_num = params.all_data.shape[1] - 1
    args.factor_num = params.factor_num
    torch.set_num_threads(args.threads)
    seed_everything(seed)
    logger = TensorBoardLogger(save_dir=params.model_path, name=name)
    profiler = SimpleProfiler(dirpath=params.profiler_path, filename=name)
    args_for_trainer = dict()
    for key, value in vars(args).items():
        try:
            Trainer(**{key: value})
            args_for_trainer[key] = value
        except:
            pass
    trainer = Trainer(**args_for_trainer,
                      num_sanity_val_steps=1,
                      profiler=profiler,
                      logger=logger,
                      deterministic=True)

    litmodel = DLLitModule(args)
    dm = DLDataModule(args, train_date_list, valid_date_list, test_date_list)
    trainer.fit(litmodel, dm)
    best_ckpt = trainer.checkpoint_callback.best_model_path
    test_result = trainer.test(ckpt_path=best_ckpt, datamodule=dm)
    print(test_result)


# 训练函数
def train(args, name, market, season, state='train'):
    # 保存最终模型的路径
    save_path = rf"{root_path}/model_test/{get_basic_name()}"
    try:
        os.makedirs(save_path, exist_ok=True)
        # 备份模型脚本
        current_file_path = os.path.abspath(__file__)
        shutil.copy(current_file_path, save_path)
    except Exception as e:
        print(e)

    params.model_name = f"{save_path}/{name[:len(market) + 19]}"
    params.test_save_path = f"{save_path}/{name[:len(market)] + name[len(market) + 6:len(market) + 19]}"
    os.makedirs(params.test_save_path, exist_ok=True)





    # 读取因子
    sel_fac_name_save = [x for x in os.listdir(rf'/home/user79/model_train_0427') if season in x]
    assert len(sel_fac_name_save) == 1
    sel_fac_name_save = sel_fac_name_save[0]
    params.all_data = pd.read_feather(rf'/home/user79/model_train_0427/{sel_fac_name_save}')
    # 检查索引是否为 date
    if params.all_data.index.name == 'date' or 'date' in params.all_data.index.names:
    # 将索引转换为列
        params.all_data = params.all_data.reset_index()
    date_list = list(params.all_data["date"].unique())
    date_list = [x for x in date_list if x in params.ret_data.index and x in params.liquid_data.index]
    date_list.sort()
    params.all_data = params.all_data.set_index("date").sort_index()

    def get_train_date_split(season, period):
        test_start = season[:4] + str(int(season.split("q")[1]) * 3 - 2).zfill(2)
        start_date = datetime.strptime(test_start, "%Y%m")
        valid_date_split = []
        for i in [-3, 0, 6, 12, 18, 24]:
            valid_date_split.append((start_date - relativedelta(months=i)).strftime("%Y%m"))
        valid_date_split.reverse()
        train_start = valid_date_split[0]
        valid_start = valid_date_split[period - 1]
        valid_end = valid_date_split[period]
        test_end = valid_date_split[-1]
        return train_start, valid_start, valid_end, test_start, test_end

    train_start, valid_start, valid_end, test_start, test_end = get_train_date_split(season, period=int(
        params.test_save_path[-1]))
    if train_start < "202101":
        train_start = "202101"

    # 获取训练集，验证集，测试集日期
    # 隔开10天以防泄露未来信息
    valid_date_list = [x for x in date_list if valid_start <= x < valid_end][:-10]
    train_date_list = [x for x in date_list if train_start <= x < valid_start][:-10] + [x for x in date_list if
                                                                                        valid_end <= x < test_start][
                                                                                       10:-10]
    test_date_list = [x for x in date_list if test_start <= x < test_end]
    # 极端行情不参与训练
    not_train_date = [x for x in date_list if (x >= "202402") & (x <= "20240223")]
    train_date_list = [x for x in train_date_list if x not in not_train_date]

    if len(test_date_list) == 0:
        test_date_list = ['out_sample']
    elif market == 'ALL':
        params.all_data = params.all_data.loc[:, params.all_data.replace(0, np.nan).dropna(how="all", axis=1).columns]
    else:
        raise NotImplementedError

    feature_map = list(params.all_data.columns[1:])
    params.factor_list = feature_map[:]
    with open(rf'{save_path}/{market}{name[len(market):len(market) + 6]}-feature_map.fea', 'w') as file:
        for idx, factor_name in enumerate(feature_map):
            file.write(rf'{factor_name}={idx}')
            file.write('\n')

    params.factor_num = params.all_data.shape[1] - 1
    if state == 'train':
        train_single(args, name, args.seed, train_date_list, valid_date_list, test_date_list)
    else:
        raise NotImplementedError
