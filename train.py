import torch
from options.train_options import TrainOptions
import torchvision.transforms as transforms

if __name__ == '__main__':
    # opt ==============================================================================
    opt = TrainOptions().parse()

    # env setting ==============================================================================
    # Fix random seed
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    # speed up compution
    torch.backends.cudnn.benchmark = True

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # data============================================================================================================
    #data Augumentation
    train_transforms =  transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])


    test_transforms = transforms.Compose([   
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ])