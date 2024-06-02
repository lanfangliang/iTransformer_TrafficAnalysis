from keras.src.utils import plot_model

from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):# 选择优化器
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):# 选择损失函数
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):  # setting is the args for this model training
        # get train dataloader
        train_data, train_loader = self._get_data(flag='train')
        # set path of checkpoint for saving and loading model
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        # 早停是一种在深度学习训练中常用的技术，用于监视模型的性能，并在模型停止提升时停止训练，以防止过拟合。
        # 创建一个早停（EarlyStopping）对象，patience 参数表示容忍的迭代次数，verbose 参数表示是否输出信息
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 选择优化器（Optimizer）和损失函数（Loss Function）
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # AMP 训练是一种技术，它利用较低精度的数据类型
        # （例如，float16）来进行某些计算，以加速训练过程并减少内存使用量。
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler() #提供的一个用于梯度缩放的工具，避免梯度溢出
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()

            epoch_time = time.time()

            # begin training in this epoch
            for i, (batch_x, batch_x_mark) in enumerate(train_loader):
                iter_count += 1  # 迭代计数器加1，用于记录当前进行了多少次迭代
                model_optim.zero_grad()  # 清零模型参数的梯度，以便于后续的反向传播计算梯度

                batch_x = batch_x.float().to(self.device)  # 输入特征数据
                batch_x_mark = batch_x_mark.float().to(self.device)  # 输入数据的时间标记


                # torch.zeros_like 创建了一个与 batch_x[:, -self.args.pred_len:, :] 相同形状的张量，所有元素的值都初始化为零
                dec_inp = torch.zeros_like(batch_x[:, -self.args.pred_len:, :]).float()

                # dec_inp = torch.ones_like(batch_x[:, -self.args.pred_len:, :]).float()
                # torch.cat 将 batch_x[:, :self.args.label_len, :] 和之前创建的零张量 dec_inp 沿着指定的维度进行拼接
                dec_inp = torch.cat([batch_x[:, :-self.args.pred_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:  # 在 TimesNet 情况下，use_amp 应该为 False,amp（自动混合精度）提高训练速度并降低内存消耗。
                    with torch.cuda.amp.autocast():  # 进入 AMP 的上下文环境
                        # 判断是否需要注意力
                        if self.args.output_attention:
                            outputs = self.model(dec_inp, batch_x_mark)[0]
                        # 对输出数据进行建模
                        else:
                            outputs = self.model(dec_inp, batch_x_mark)

                        # 预测任务类型，选项：[M, S, MS]；M: 多变量预测多变量，S: 单变量预测单变量，MS: 多变量预测单变量'
                        # 如果是多变量预测单变量'，则输出应该是解码器输出的最后一列，所以 f_dim = -1 只包含最后一列，否则包含所有列
                        f_dim = 9
                        outputs = outputs[:, -self.args.pred_len:, :f_dim]
                        batch_x = batch_x[:, -self.args.pred_len:, :f_dim].to(self.device)

                        # 计算损失
                        loss = criterion(outputs, batch_x)
                        train_loss.append(loss.item())
                else:  # similar to when use_amp is True
                    if self.args.output_attention:
                        outputs = self.model(dec_inp, batch_x_mark)[0]
                    else:
                        outputs = self.model(dec_inp, batch_x_mark)
                        # outputs = self.model(batch_x, batch_x_mark)
                    f_dim = 9
                    outputs = outputs[:, -self.args.pred_len:, :f_dim]
                    batch_x = batch_x[:, -self.args.pred_len:, :f_dim].to(self.device)

                    loss = criterion(outputs, batch_x)
                    train_loss.append(loss.item())


                # BP
                if self.args.use_amp:
                    scaler.scale(loss).backward()  # 计算损失的梯度并缩放
                    scaler.step(model_optim)  # 使用优化器更新模型参数
                    scaler.update()  # 更新缩放器的比例
                else:
                    loss.backward()  # 计算损失的梯度
                    model_optim.step()  # 使用优化器更新模型参数

            # This epoch comes to end, print information
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))

            # Decide whether to trigger Early Stopping. if early_stop is true, it means that
            # this epoch's training is now at a flat slope, so stop further training for this epoch.
            early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 学习率调整
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        best_model_path = path + '/' + 'checkpoint.pth'

        # loading the trained model's state dictionary from a saved checkpoint file
        # located at best_model_path.
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=1):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        plot_model(self.model, to_file='my_dir/model_plot.png', show_shapes=True, show_layer_names=True)
        self.model.eval()
        with torch.no_grad():  # 关闭梯度
            for i, (batch_x, batch_x_mark) in enumerate(test_loader, start=1):

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_x[:, -self.args.pred_len:, :]).float()
                # dec_inp = torch.ones_like(batch_x[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_x[:, :-self.args.pred_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(dec_inp, batch_x_mark)[0]
                        else:
                            outputs = self.model(dec_inp, batch_x_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(dec_inp, batch_x_mark)[0]
                    else:
                        outputs = self.model(dec_inp, batch_x_mark)
                        # outputs = self.model(batch_x, batch_x_mark)
                f_dim = 9
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_x = batch_x[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_x = batch_x.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_x = test_data.inverse_transform(batch_x.squeeze(0)).reshape(shape)

                outputs = outputs[:, :, :f_dim]
                batch_x = batch_x[:, :, :f_dim]

                pred = outputs
                true = batch_x

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # 将 preds 数组重塑为二维数组
        preds_2d = preds.reshape(-1, preds.shape[-1])
        trues_2d = trues.reshape(-1, trues.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, r2 = metric(preds, trues)
        print('R2:{}'.format(r2))
        print('mse:{}, mae:{}'.format(mse, mae))

        # 保存 preds_2d 为 CSV 文件
        np.savetxt(folder_path + 'preds1.csv', preds_2d, delimiter=',')
        np.savetxt(folder_path + 'trues1.csv', trues_2d, delimiter=',')

        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.ones_like(batch_x[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_x[:, :, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(dec_inp, batch_x_mark)[0]
                        else:
                            outputs = self.model(dec_inp, batch_x_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(dec_inp, batch_x_mark)[0]
                    else:
                        outputs = self.model(dec_inp, batch_x_mark)
                f_dim = 72
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)

                outputs = outputs[:, :, :f_dim]
                batch_y = batch_y[:, :, :f_dim]

                preds.append(outputs)
                trues.append(batch_y)

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # 将 preds 数组重塑为二维数组
        preds_2d = preds.reshape(-1, preds.shape[-1])
        trues_2d = trues.reshape(-1, trues.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, r2 = metric(preds, trues)
        print('R2:{}'.format(r2))
        print('mse:{}, mae:{}'.format(mse, mae))

        # np.save(folder_path + 'real_prediction.npy', preds)
        # 保存 preds_2d 为 CSV 文件
        # np.savetxt(folder_path + 'preds2.csv', preds_2d, delimiter=',')
        # np.savetxt(folder_path + 'trues2.csv', trues_2d, delimiter=',')
        return