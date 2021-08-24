
import os
import argparse
from utils import util


class TrainOptions():

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.opt = None

    def parse(self):
        parser = argparse.ArgumentParser(description='Person ReID Frame')

        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

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

    