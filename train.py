import os
import torch

from options.train_options import TrainOptions
import torchvision.transforms as transforms
from data.cat_dog import CatAndDog

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


    import glob

    train_dir = './datasets/dogVScat/train'
    test_dir = 'datasets/dogVScat/test'
    batch_size=2

    print(os.listdir(train_dir)[:5])

    train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
    test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

    train_data = CatAndDog(train_list, transform=train_transforms)
    test_data = CatAndDog(test_list, transform=test_transforms)


    train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
    test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)
    print(train_loader)

    