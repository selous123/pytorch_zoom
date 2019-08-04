import os
import math
from trainer import Trainer
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm

class SSL_Trainer(Trainer):

    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(SSL_Trainer, self).__init__(args, loader, my_model, my_loss, ckp)

    def train(self):

        ## 1
        [loss.step() for loss in self.loss]
        # self.loss[0].step()
        # self.loss[1].step()

        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )

        ## 2
        # self.loss[0].start_log()
        # self.loss[1].start_log()
        [loss.start_log() for loss in self.loss]

        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, labels, filename, idx_scale) in enumerate(self.loader_train):

            hr = labels[0]
            h_label = labels[1]

            lr, hr, h_label = self.prepare(lr, hr, h_label)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            sr, s_label = self.model(lr, idx_scale)

            ## white balance
            #sr = utility.postprocess_wb(sr, filename, self.args.wb_root)

            loss_SR = self.loss[0](sr, hr)
            loss_SSL = self.loss[1](s_label, h_label)

            l = loss_SR + loss_SSL
            if self.args.loss_rel is not None:

                loss_rel = self.loss[2]([hr, h_label], [sr, s_label])
                l = l + loss_rel



            l.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                if self.args.loss_rel is None:
                    self.ckp.write_log('[{}/{}]\t{}\t{}\t{:.1f}+{:.1f}s'.format(
                        (batch + 1) * self.args.batch_size,
                        len(self.loader_train.dataset),
                        self.loss[0].display_loss(batch),
                        self.loss[1].display_loss(batch),
                        timer_model.release(),
                        timer_data.release()))
                else:
                    self.ckp.write_log('[{}/{}]\t{}\t{}\t{}\t{:.1f}+{:.1f}s'.format(
                        (batch + 1) * self.args.batch_size,
                        len(self.loader_train.dataset),
                        self.loss[0].display_loss(batch),
                        self.loss[1].display_loss(batch),
                        self.loss[2].display_loss(batch),
                        timer_model.release(),
                        timer_data.release()))

            timer_data.tic()
        ## 3
        [loss.end_log(len(self.loader_train)) for loss in self.loss]
        # self.loss[0].end_log(len(self.loader_train))
        # self.loss[1].end_log(len(self.loader_train))
        self.error_last = self.loss[0].log[-1, -1]
        self.error_last = self.loss[1].log[-1, -1]
        self.optimizer.schedule()
