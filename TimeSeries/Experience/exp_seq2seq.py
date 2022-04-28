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


    def val(self, seq2seq, criterion):
        """
        :param model:
        :param criterion:
        :return:
        """
        dlv = self.get_data('val')
        val_loss = []
        seq2seq = seq2seq.eval()
        for batch_x, batch_y in dlv:
            batch_x = batch_x.to(torch.float32).to(self.args['device'])

            pre_y = seq2seq(batch_x, batch_x[:, -1:, :])

            batch_y = batch_y[:, :, -1:].reshape((-1, 288)).to(self.args['device'])

            loss = criterion(batch_y, pre_y)
            val_loss.append(loss.to('cpu').item())
        val_loss = np.average(val_loss)
        return val_loss


    def training(self):
        """
        Desc: train model
        """
        history = dict(train=[], val=[])
        best_loss = 10000.0

        self.seed_everything(1024)

        dlt = self.get_data('train')

        seq2seq = Seq2Seq(self.args)

        optimizer = torch.optim.Adam(seq2seq.parameters(), lr=4e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5e-3, eta_min=1e-8, last_epoch=-1)
        criterion = torch.nn.MSELoss()


        epoches = self.args['epoches']

        for epoch in range(epoches):
            seq2seq = seq2seq.train()
            train_loss = []

            for batch_x, batch_y in dlt:
                batch_x = batch_x.to(torch.float32).to(self.args['device'])

                pre_y = seq2seq(batch_x, batch_x[:, -1:, :])

                batch_y = batch_y[:, :, -1:].reshape((-1, self.args['output_length'])).to(self.args['device'])

                loss = criterion(batch_y, pre_y)
                train_loss.append(loss.to('cpu').item())
                optimizer.step()
                optimizer.zero_grad()

                print("Train Loss: {}".format(loss.to('cpu').item()))
                # torch.cuda.empty_cache()  # TODO CUDA boom
            train_loss = np.average(train_loss)
            val_loss =  self.val(seq2seq, criterion)
            print("Epoch: {}, \nTrain Loss: {}, \nValidation Loss: {}".format(epoch, train_loss, val_loss))

            history['train'].append(train_loss)
            history['val'].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(seq2seq.state_dict(), './../checkpoints/seq2seq_best_model.pth')
                print("saved best model epoch:", epoch, "val loss is:", val_loss)

            scheduler.step()
            return history



