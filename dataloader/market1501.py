import glob
import os
import re


from PIL import Image
from torch.utils.data import Dataset


class Market1501(Dataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """

    dataset_dir = ""

    def __init__(self, root="", data_folder="",transform=None, relabel=False):
        super(Market1501, self).__init__()

        self.transform = transform

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.data_dir = os.path.join(self.dataset_dir, data_folder)

        self._check_before_run()

        self.dataset = self._process_dir(self.data_dir, relabel=relabel)

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
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(os.path.join(dir_path, "*.jpg"))
        pattern = re.compile(r"([-\d]+)_c(\d)")

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

    def _read_image(self, img_path):
        """Keep reading image until succeed.
        This can avoid IOError incurred by heavy IO process."""
        got_img = False
        if not os.path.exists(img_path):
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
