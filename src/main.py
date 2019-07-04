import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from ssl_trainer import SSL_Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            # print(len(loader.loader_train))
            # for d in loader.loader_train:
            #     print(d[0].shape)
            #     print(d[1].shape)
            #     print(d[2])
            #     print(d[3])
            #
            #     break;
            # exit(0)
            _model = model.Model(args, checkpoint)
            if args.model == "SSL":
                _loss = [loss.Loss(args, checkpoint), loss.Loss(args, checkpoint)] if not args.test_only else None
                t = SSL_Trainer(args, loader, _model, _loss, checkpoint)
            else:
                _loss = loss.Loss(args, checkpoint) if not args.test_only else None
                t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
