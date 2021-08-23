import torch
from options.train_options import TrainOptions

if __name__ == '__main__':
    # opt ==============================================================================
    opt = TrainOptions().parse()

    # Fix random seed==============================================================================
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    # speed up compution==============================================================================
    torch.backends.cudnn.benchmark = True

    # data============================================================================================================
    