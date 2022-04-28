# -*- coding: utf-8 -*-
# @Time : 2022/4/28 14:52
# @Author : li.zhang
# @File : exp_seq2seq.py


from utils.pyp import *
from dataset.gru_dataset import GruDataset
from models.seq2seq import Seq2Seq


class ExpSeq2Seq():
    """
    Desc: the train and val seq2seq model
    """

    def __init__(self, args, dataset):

        self.dataset = dataset
        self.args = args


    @classmethod
    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


    def get_data(self, flag='train'):
        ds = GruDataset(self.args, self.dataset, flag)
        if flag == 'train' or flag == 'val':
            shuffle = True
        else:
            shuffle = False
        dl = DataLoader(ds, batch_size=self.args['batch_size'],
                        shuffle=shuffle, drop_last=True)
        return dl



    def training(self):
        """
        Desc: train model
        """
        self.seed_everything(1024)

        dlt = self.get_data('train')

        seq2seq = Seq2Seq(self.args)

        optimizer = torch.optim.Adam(seq2seq.parameters(), lr=4e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5e-3, eta_min=1e-8, last_epoch=-1)
        criterion = torch.nn.MSELoss()


        epoches = self.args['epoches']

        for epoche in range(epoches):
            seq2seq = seq2seq.train()
            train_loss = []

            for batch_x, batch_y in dlt:
                batch_x = batch_x.to(torch.float32)

                pre_y = seq2seq(batch_x, batch_x[:, -1:, -1:])

                # 对比 和 batch_y得到新的数据
                optimizer.step()
                optimizer.zero_grad()

