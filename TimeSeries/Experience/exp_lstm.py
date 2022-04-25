# -*- coding: utf-8 -*-
# @Time : 2022/4/20 16:37
# @Author : li.zhang
# @File : exp_lstm.py

from utils.pyp import *

from XXDataSet.gru_dataset import GruDataset
from models.lstm import RnnModel

class ExpLSTM():
    """
    Desc:
    """

    def __init__(self, conf, data):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf = conf
        self.data = data

        self.best_score = None
        self.delta = self.conf['delta']
        self.counter = 0

    def __seed_everything__(self, seed):
        """
        Desc:
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


    def get_data(self, flag='train'):
        """
        Desc: the format of DataLoader
        """
        ds = GruDataset(self.conf, self.data, flag)
        shuffle = True
        if flag == 'train' or flag == 'val':
            shuffle = True
        else:
            shuffle = False
        dl = DataLoader(ds, batch_size=self.conf['batch_size'],
                        shuffle=shuffle, drop_last=True)
        return dl

    def get_optimizer(self, model, lr):
        """
        Desc:
        """
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)


    def get_criterion(self):
        """
        Desc:
        """
        return torch.nn.MSELoss().to(self._device)

    def save_checkpoint(self, model, val):
        """
        Desc:
        """
        torch.save({'model': model.state_dict()}, f'../checkpoints/model_name_{val}.pth')


    def early_stop(self, val_loss):
        """
        Desc:
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.conf['patient']:
                return True


    def adjust_learning_rate(self, optimizer, lr):
        """
        Desc:
        """
        optimizer.set_lr(lr)

    # def get_model(self):
    #     """
    #     Desc: define model
    #     """
    #     lstm = RnnModel(self.conf)
    #     return lstm




    def val(self, model, criterion):
        """
        Desc: validation model
        """
        dlv = self.get_data('val')
        val_loss = []
        for batch_x, batch_y in dlv:
            batch_x = batch_x.to(torch.float32)
            pre_y = model(batch_x)  # TODO

            batch_y = batch_y[:, -self.conf['output_size']:, -1:].type(torch.float64)
            pre_y = pre_y[..., -self.conf['output_size']:, -1:].type(torch.float64)

            loss = criterion(batch_y, pre_y)
            val_loss.append(loss.item())
        val_loss = np.average(val_loss)
        return val_loss


    def training(self):
        """
        Desc: train and val
        """
        # 初始化
        self.__seed_everything__(2014)

        # 获取数据
        dlt = self.get_data('train')


        # define model
        lstm = RnnModel(self.conf)

        # define optimizer and criterion
        optimizer = self.get_optimizer(lstm, self.conf['lr'])  # TODO adjust lr
        criterion = self.get_criterion()

        epoches = self.conf['epoches']

        for i in range(epoches):
            train_loss = []
            for batch_x, batch_y in dlt:
                batch_x = batch_x.to(torch.float32)
                pre_y = lstm(batch_x)

                batch_y = batch_y[:, -self.conf['output_size']:, -1:].type(torch.float64)
                pre_y = pre_y[..., -self.conf['output_size']:, -1:].type(torch.float64)

                loss = criterion(batch_y, pre_y)
                train_loss.append(loss.item())

                optimizer.step()
                optimizer.zero_grad()

            train_loss = np.average(train_loss)
            val_loss = self.val(lstm, criterion)
            print("Epoch: {}, \nTrain Loss: {}, \nValidation Loss: {}".format(i, train_loss, val_loss))

            if self.early_stop(val_loss):
                self.save_checkpoint(lstm, val_loss)











