
import os
import argparse
from util import util


class TrainOptions():

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.opt = None

    def parse(self):
        parser = argparse.ArgumentParser(description='Person ReID Frame')

        # base
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # env setting
        parser.add_argument('--random_seed', type=int, default='1', help='random seed')

        # data
        parser.add_argument('--img_height', type=int, default=224, help='height of the input image')
        parser.add_argument('--img_width', type=int, default=224, help='width of the input image')
        parser.add_argument('--train_dir', type=str, default='./datasets/dogVScat/train', help='train dataset dir')
        parser.add_argument('--test_dir', type=str, default='./datasets/dogVScat/test', help='test dataset dir')
        parser.add_argument('--batch_size', default=128, type=int, help='batch size')

        # Optimizer
        parser.add_argument('--lr', type=float, default=0.1, help='learning rate')

        # train
        parser.add_argument('--start_epoch', type=int, default=1, help='start epoch')
        parser.add_argument('--epoch_num', type=int, default=1, help='start epoch')



        self.opt = parser.parse_args()
        self.print_options(self.opt)

        return self.opt 


    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            message += '{:>25}: {:<30}\n'.format(str(k), str(v))
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    