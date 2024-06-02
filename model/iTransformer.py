import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm   #它的默认值为 True，表示使用归一化和反归一化处理
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        # 输入特征初始化，负责将输入序列进行编码，提取其特征表示
        self.encoder = Encoder(
            [# 每个 EncoderLayer 包括了自注意力层和前馈神经网络层
                EncoderLayer(
                    # 这是自注意力层，用于对输入序列中的不同位置进行关注和加权
                    AttentionLayer(
                        # 这里使用了全注意力（Full Attention）机制，它允许模型在计算注意力时同时考虑输入序列中的所有位置。
                        # configs.factor 是指自注意力机制中的缩放因子，用于控制注意力分布的范围。
                        # attention_dropout=configs.dropout 指定了在计算注意力时使用的丢弃率。
                        # output_attention=configs.output_attention 则指定了是否将注意力分布作为输出。
                        # configs.d_model表示模型的特征向量维度
                        # configs.n_heads表示模型的注意力头的数量
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    # 这是前馈神经网络层的隐藏层维度，用于对特征向量进行非线性变换
                    configs.d_ff,
                    # 这是指定了在网络中使用的丢弃率，用于防止过拟合
                    dropout=configs.dropout,
                    # 这是激活函数，通常使用 ReLU 或 GELU
                    # ReLU(f(x) = max(0, x)),GELU(f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    activation=configs.activation
                ) for l in range(configs.e_layers)# 这是指定了Encoder中包含的EncoderLayer的数量
            ],
            # 表示对 Encoder 输出的特征向量进行层归一化
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc):
        # 首先，对输入的时间序列数据进行归一化处理。通过计算均值和标准差来标准化输入数据，以确保数据分布稳定
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # 对解码器的输出进行反归一化处理，以获得原始数据的预测结果
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forward(self, x_enc, x_mark_enc):
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]