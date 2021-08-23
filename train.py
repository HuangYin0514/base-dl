from options.train_options import TrainOptions


if __name__ == '__main__':
    opt = TrainOptions().parse()
    print(opt)
