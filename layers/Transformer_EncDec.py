import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x

# 该层由自注意力机制（self-attention）和两个前馈神经网络（Feed Forward Neural Networks）组成
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        # 创建一个 1D 卷积层，用于前馈神经网络的第一层
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        # 创建一个 1D 卷积层，用于前馈神经网络的第二层
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # 创建一个 Layer Normalization 层，用于规范化自注意力机制的输出
        self.norm1 = nn.LayerNorm(d_model)
        # 创建一个 Layer Normalization 层，用于规范化前馈神经网络的输出
        self.norm2 = nn.LayerNorm(d_model)
        # 创建一个 Dropout 层，用于在前向传播过程中进行随机失活
        self.dropout = nn.Dropout(dropout)
        # 根据激活函数类型选择对应的激活函数
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # 将输入 x 传递给自注意力机制模块，并获得输出 new_x 和注意力权重 attn
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        # 将输入 x 和经过 Dropout 处理的注意力输出相加，得到注意力机制的输出，并保留了残差连接
        x = x + self.dropout(new_x)
        # 将注意力机制的输出规范化，并保存到 y 和 x 中
        y = x = self.norm1(x)
        #将 y 转置后传入第一个卷积层，并经过激活函数和 Dropout 处理
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # 将经过第一个卷积层处理的 y 再次转置后传入第二个卷积层，并经过 Dropout 处理
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        # 将自注意力机制的输出 x 和经过两个卷积层处理后的 y 相加，并规范化后返回，并返回注意力权重 attn
        return self.norm2(x + y), attn

# 该编码器由注意力层（self-attention layer）和卷积层（convolutional layer）组成
class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers) # 注意力层的列表
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None # 卷积层的列表
        self.norm = norm_layer # 规范化层，用于规范化编码器的输出

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = [] #用于存储每个注意力层的注意力权重
        if self.conv_layers is not None: # 如果存在卷积层
            # 遍历注意力层和卷积层，并逐个处理它们
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                # 如果是第一个注意力层，则将 delta 保持不变；否则，将 delta 设置为 None
                delta = delta if i == 0 else None
                # 将输入 x 传递给注意力层，并获得输出 x 和注意力权重 attn
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                # 将输出 x 传递给卷积层进行处理
                x = conv_layer(x)
                attns.append(attn)
                # 对于最后一个注意力层，不再传递 attn_mask，并将 delta 设置为 None
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                # 将输入 x 传递给注意力层，并获得输出 x 和注意力权重 attn
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
