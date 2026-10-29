# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [MODULE DOC] Omniglot dataset support for the MAML examples: Omniglot (a map-style Dataset
# that downloads and indexes the image files), find_classes and index_classes helpers, and
# OmniglotNShot, which builds cached N-way K-shot support/query batches as tensors.
# No import-time side effects beyond imports; download and file writes happen when the
# classes are instantiated.
# These Omniglot loaders are from Jackie Loong's PyTorch MAML implementation:
#     https://github.com/dragen1860/MAML-Pytorch
#     https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot.py
#     https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglotNShot.py

import errno
import os
import os.path

import numpy as np
from PIL import Image
from torchvision import transforms

import torch
import torch.utils.data as data


class Omniglot(data.Dataset):
    # [CLASS DOC] Map-style dataset over the Omniglot character images stored under
    # root/processed. State: root, transform, target_transform, all_items (list of
    # (filename, class path, directory) tuples) and idx_classes (mapping from class path to
    # integer label). Items are (transformed image path, integer label). Class attributes give
    # the download URLs and folder names. Raises RuntimeError at construction if the data is
    # missing and download is false.
    urls = [
        "https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip",
        "https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip",
    ]
    raw_folder = "raw"
    processed_folder = "processed"
    training_file = "training.pt"
    test_file = "test.pt"

    """
    The items are (filename,category). The index of all the categories can be found in self.idx_classes
    Args:
    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    - download: need to download the dataset
    """

    def __init__(self, root, transform=None, target_transform=None, download=False):
        # [FUNCTION DOC] Stores the root and transforms, downloads the dataset if it is absent
        # and download is true (raises RuntimeError if absent and download is false), then scans
        # the processed folder to build all_items and the class-to-label mapping idx_classes.
        # Side effects: file system and network I/O, prints counts.
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            if download:
                self.download()
            else:
                raise RuntimeError(
                    "Dataset not found." + " You can use download=True to download it"
                )

        self.all_items = find_classes(os.path.join(self.root, self.processed_folder))
        self.idx_classes = index_classes(self.all_items)

    def __getitem__(self, index):
        # [FUNCTION DOC] Returns (image, target) for the item at index. The image starts as the
        # full file path string of the item and is passed through transform if set (the
        # transform is what opens the file); the target is the integer class label, passed
        # through target_transform if set.
        filename = self.all_items[index][0]
        img = str.join("/", [self.all_items[index][2], filename])

        target = self.idx_classes[self.all_items[index][1]]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        # [FUNCTION DOC] Number of image items found in the processed folder.
        return len(self.all_items)

    def _check_exists(self):
        # [FUNCTION DOC] HELPER: True if both the images_evaluation and images_background
        # folders exist under the processed folder of root.
        return os.path.exists(
            os.path.join(self.root, self.processed_folder, "images_evaluation")
        ) and os.path.exists(
            os.path.join(self.root, self.processed_folder, "images_background")
        )

    def download(self):
        # [FUNCTION DOC] Downloads the two Omniglot zip archives to root/raw and extracts them
        # into root/processed. Does nothing if the data already exists. Creates the folders
        # (an already-existing folder is tolerated; other OS errors propagate). Prints
        # progress; network and file errors propagate.
        # KNOWN ISSUE: imports urllib but uses urllib.request without importing it, which can
        # fail with AttributeError unless another module imported urllib.request earlier.
        import urllib
        import zipfile

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print("== Downloading " + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition("/")[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, "wb") as f:
                f.write(data.read())
            file_processed = os.path.join(self.root, self.processed_folder)
            print("== Unzip from " + file_path + " to " + file_processed)
            zip_ref = zipfile.ZipFile(file_path, "r")
            zip_ref.extractall(file_processed)
            zip_ref.close()
        print("Download finished.")


def find_classes(root_dir):
    # [FUNCTION DOC] HELPER: Walks root_dir recursively and returns a list of
    # (file name, class path made of the last two directory names, containing directory)
    # tuples for every file whose name ends with png. Prints the item count.
    # Assumes forward-slash separated paths.
    retour = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith("png"):
                r = root.split("/")
                lr = len(r)
                retour.append((f, r[lr - 2] + "/" + r[lr - 1], root))
    print(f"== Found {len(retour)} items ")
    return retour


def index_classes(items):
    # [FUNCTION DOC] HELPER: Assigns consecutive integer labels (in first-seen order) to the
    # distinct class paths found in the second field of each item; returns that mapping.
    # Prints the class count.
    idx = {}
    for i in items:
        if i[1] not in idx:
            idx[i[1]] = len(idx)
    print(f"== Found {len(idx)} classes")
    return idx


class OmniglotNShot:
    # [CLASS DOC] Episodic N-way K-shot sampler over Omniglot. State: x (all character images),
    # x_train (first 1200 classes) and x_test (the rest), batchsz (tasks per batch), n_way,
    # k_shot, k_query, resize, device, indexes (read position per split), datasets and
    # datasets_cache (ten pre-sampled batches per split, regenerated when exhausted).
    # Invariant: k_shot + k_query <= 20 images per class. Key methods: load_data_cache, next.
    def __init__(self, root, batchsz, n_way, k_shot, k_query, imgsz, device=None):
        # [FUNCTION DOC] Loads the cached omniglot.npy array from root, or builds it by
        # downloading the dataset, resizing images to imgsz and saving it (file and network
        # I/O). Splits classes into train (first 1200) and test, records the sampling
        # parameters, and pre-fills the batch caches for both splits. Raises AssertionError if
        # k_shot plus k_query exceeds 20. Prints status lines.
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_query:
        :param imgsz:
        """

        self.resize = imgsz
        self.device = device
        if not os.path.isfile(os.path.join(root, "omniglot.npy")):
            # if root/data.npy does not exist, just download it
            self.x = Omniglot(
                root,
                download=True,
                transform=transforms.Compose(
                    [
                        lambda x: Image.open(x).convert("L"),
                        lambda x: x.resize((imgsz, imgsz)),
                        lambda x: np.reshape(x, (imgsz, imgsz, 1)),
                        lambda x: np.transpose(x, [2, 0, 1]),
                        lambda x: x / 255.0,
                    ]
                ),
            )

            temp = {}  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
            for img, label in self.x:
                if label in temp:
                    temp[label].append(img)
                else:
                    temp[label] = [img]

            self.x = []
            for (
                label,
                imgs,
            ) in temp.items():  # labels info deserted , each label contains 20imgs
                self.x.append(np.array(imgs))

            # as different class may have different number of imgs
            self.x = np.array(self.x).astype(
                np.float64
            )  # [[20 imgs],..., 1623 classes in total]
            # each character contains 20 imgs
            print("data shape:", self.x.shape)  # [1623, 20, 84, 84, 1]
            temp = []  # Free memory
            # save all dataset into npy file.
            np.save(os.path.join(root, "omniglot.npy"), self.x)
            print("write into omniglot.npy.")
        else:
            # if data.npy exists, just load it.
            self.x = np.load(os.path.join(root, "omniglot.npy"))
            print("load from omniglot.npy.")

        # [1623, 20, 84, 84, 1]
        # TODO: can not shuffle here, we must keep training and test set distinct!
        self.x_train, self.x_test = self.x[:1200], self.x[1200:]

        # self.normalization()

        self.batchsz = batchsz
        self.n_cls = self.x.shape[0]  # 1623
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        if (k_shot + k_query) > 20:
            raise AssertionError(
                f"k_shot + k_query must be <= 20, got {k_shot + k_query}"
            )

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {
            "train": self.x_train,
            "test": self.x_test,
        }  # original data cached
        print("DB: train", self.x_train.shape, "test", self.x_test.shape)

        self.datasets_cache = {
            "train": self.load_data_cache(
                self.datasets["train"]
            ),  # current epoch data cached
            "test": self.load_data_cache(self.datasets["test"]),
        }

    def normalization(self):
        # [FUNCTION DOC] Standardizes x_train and x_test using the training mean and standard
        # deviation, then recomputes and stores mean, std, max and min of the normalized
        # training data. Mutates the instance. Not called anywhere in this file.
        """
        Normalizes our data, to have a mean of 0 and sdt of 1
        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        # print("before norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)

    # print("after norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)

    def load_data_cache(self, data_pack):
        # [FUNCTION DOC] Samples 10 batches of episodes from data_pack (classes by 20 images).
        # Each batch has batchsz tasks; each task picks n_way distinct classes and
        # k_shot + k_query distinct images per class, split into support and query sets, with
        # labels 0..n_way-1, shuffled within the set. Returns a list of 10 items, each a list
        # [support images, support labels, query images, query labels] of tensors moved to the
        # device. Uses the numpy global random state.
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes
            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set
                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)

                for j, cur_class in enumerate(selected_cls):
                    selected_img = np.random.choice(
                        20, self.k_shot + self.k_query, False
                    )

                    # meta-training and meta-test
                    x_spt.append(data_pack[cur_class][selected_img[: self.k_shot]])
                    x_qry.append(data_pack[cur_class][selected_img[self.k_shot :]])
                    y_spt.append([j for _ in range(self.k_shot)])
                    y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(
                    self.n_way * self.k_shot, 1, self.resize, self.resize
                )[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(
                    self.n_way * self.k_query, 1, self.resize, self.resize
                )[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)

            # [b, setsz, 1, 84, 84]
            x_spts = (
                np.array(x_spts)
                .astype(np.float32)
                .reshape(self.batchsz, setsz, 1, self.resize, self.resize)
            )
            y_spts = np.array(y_spts).astype(int).reshape(self.batchsz, setsz)
            # [b, qrysz, 1, 84, 84]
            x_qrys = (
                np.array(x_qrys)
                .astype(np.float32)
                .reshape(self.batchsz, querysz, 1, self.resize, self.resize)
            )
            y_qrys = np.array(y_qrys).astype(int).reshape(self.batchsz, querysz)

            x_spts, y_spts, x_qrys, y_qrys = (
                torch.from_numpy(z).to(self.device)
                for z in [x_spts, y_spts, x_qrys, y_qrys]
            )

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def next(self, mode="train"):
        # [FUNCTION DOC] Returns the next cached batch for the split named mode and advances its
        # read index; when the cache is exhausted, resets the index and resamples the cache
        # first. Raises KeyError for a mode other than train or test (val has no data).
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch
