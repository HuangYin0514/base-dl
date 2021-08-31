import glob
import os.path as osp

from PIL import Image
from torch.utils.data import Dataset

class Hymenoptera(Dataset):

    root_dir = ""

    def __init__(self, root="", mode="", transform=None):
        super(Hymenoptera, self).__init__()

        self.transform = transform

        self.root_dir = osp.abspath(osp.expanduser(root))
        # self.dataset_dir = osp.join(self.root_dir, mode)

        self.category2label = self._get_label()

        self._check_before_run()

        self.dataset = self._process_dir(self.root_dir)

    def __getitem__(self, index):
        img_path, label = self.dataset[index]
        img = self._read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root_dir):
            raise RuntimeError("'{}' is not available".format(self.root_dir))

    # 未来优化的地方，考虑reid网络中relabel形式
    def _get_label(self):
        # img_paths = glob.glob(osp.join(dir_path, "*", "*.jpg"))

        # category_container = set()
        # for img_path in img_paths:
        #     category = img_path.split('/')[-2]
        #     category_container.add(category)
        # category2label = {
        #     category: label for label, category in enumerate(category_container)
        # }
        category2label = {"ants": 1, "bees": 0}
        return category2label

    def _process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, "*", "*.jpg"))

        dataset = []
        for img_path in img_paths:
            category = img_path.split("/")[-2]
            label = self.category2label[category]
            dataset.append((img_path, label))

        return dataset

    def _read_image(self,img_path):
        """Keep reading image until succeed.
        This can avoid IOError incurred by heavy IO process."""
        got_img = False
        if not osp.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert("RGB")
                got_img = True
            except IOError:
                print(
                    "IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(
                        img_path
                    )
                )
                pass
        return img
