import glob
import os
import copy
import pickle
from PIL import Image

import numpy as np
from skimage.transform import resize
import lmdb

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .utils import load_classes
from .augmentations import Augmentation


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        if h <= w:
            pad = ((pad1, pad2), (0, 0), (0, 0))
        else:
            pad = ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Subtract mean
        mean = np.array([0.5, 0.5, 0.5])
        input_img -= mean
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return img_path, input_img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    """COCO dataset"""

    def __init__(self, list_path, img_size=416):
        with open(list_path, 'r') as fdata:
            img_files = fdata.readlines()
        self.img_files = [
            path.replace('code/yolov3_origin/', '') for path in img_files
        ]
        self.label_files = [
            path.replace('images',
                         'labels').replace('.png',
                                           '.txt').replace('.jpg', '.txt')
            for path in img_files
        ]
        self.img_shape = (img_size, img_size)
        # set max object to align num of objects for different images
        # if less than, filling with 0; if more than, dropping
        self.max_objects = 70

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        # Only 3-channels image will be loaded
        while True:
            img_path = self.img_files[index % len(self.img_files)].strip()
            # (height, width, channels), 0-255 uint8, just like cv2.imread
            img = np.array(Image.open(img_path))
            if len(img.shape) != 3:
                index += 1
            else:
                break

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        if h <= w:
            pad = ((pad1, pad2), (0, 0), (0, 0))
        else:
            pad = ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Normalized by mean
        mean = np.array([0.5, 0.5, 0.5])
        input_img -= mean
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].strip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3] / 2)
            y1 = h * (labels[:, 2] - labels[:, 4] / 2)
            x2 = w * (labels[:, 1] + labels[:, 3] / 2)
            y2 = h * (labels[:, 2] + labels[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            num_gts = min(len(labels), self.max_objects)
            filled_labels[:num_gts] = copy.deepcopy(labels[:num_gts])
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)


class ImagenetDataset(Dataset):
    """Dataset Loading"""

    def __init__(self, list_path, img_size=224, transform=None):
        self.name = 'IMAGENET2012'
        self.img_size = img_size
        self.img_shape = (self.img_size, self.img_size)
        self.set_type = list_path.strip().split('/')[-1].split('.')[0]
        # Image path_prefix
        self.prefix = os.path.expanduser('~/data/imagenet2012/{}/'.format(
            self.set_type))
        # Image and labels
        with open(list_path, 'r') as fpath:
            self.img_labels = fpath.readlines()
        self.img_files = []
        self.labels = []
        for x in self.img_labels:
            if not x.strip():
                continue
            split_x = x.strip().split(' ')
            img_file, lbl = split_x[0].strip(), split_x[1].strip()
            self.img_files.append(os.path.join(self.prefix, img_file))
            self.labels.append(int(lbl))
        # Transform
        if transform is None:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(self.img_size),
                transforms.RandomHorizontalFlip()
            ])
        self.transform = transform

    def __getitem__(self, index):
        # Only RGB image will be loaded
        while True:
            img_path = self.img_files[index % len(self.img_files)].strip()
            # (height, width, (R, G, B)), 0-255 uint8
            img_pil = Image.open(img_path)
            if img_pil.mode == 'RGB':
                break
            else:
                index += 1
        # Get labels
        label = self.labels[index % len(self.img_files)]
        # ---------
        #  Image
        # ---------
        img_pil = self.transform(img_pil)
        input_img = np.array(img_pil, dtype=np.float32) / 255.
        mean = np.array([0.5, 0.5, 0.5])
        input_img -= mean
        # Channels-first for PyTorch
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()
        label = torch.LongTensor([label])
        return img_path, input_img, label

    def __len__(self):
        return len(self.img_files)


class DotaDataset(Dataset):
    """Dataset Loading"""

    def __init__(self, list_path, img_size=416, lmdb=True):
        # Set: train, val, test
        self.lmdb = lmdb
        if self.lmdb:
            self.set_type = list_path.strip().split('/')[-1].split(
                '.')[0].split('_')[0]
            self.env, self.names = self._get_names_from_lmdb(list_path)
            self.ids = self.names
            self.labels_path = os.path.expanduser(
                '~/data/dota/{}_crop/labelTxt/{}.txt')
        else:
            self.set_type = list_path.strip().split('/')[-1].split(
                '.')[0].split('_')[0]
            with open(list_path, 'r') as fpath:
                self.img_files = fpath.readlines()
            self.labels_path = os.path.expanduser(
                '~/data/dota/{}_crop/labelTxt/{}.txt')
            self.ids = [
                x.strip().split('/')[-1].split('.')[0] for x in self.img_files
            ]
            self.label_files = [
                self.labels_path.format(self.set_type, name)
                for name in self.ids
            ]
        self.img_size = img_size
        self.img_shape = (img_size, img_size)
        self.max_objects = 500  # XXX according to the training set
        self.name = 'DOTA'
        # load classes
        self.classes = load_classes(
            os.path.expanduser('~/code/yolov3_origin/data/dota/dota.names'))
        self.augs = Augmentation(self.img_size)  # data augmentations

    def _get_names_from_lmdb(self, dataroot):
        env = lmdb.open(dataroot,
                        readonly=True,
                        lock=False,
                        readahead=False,
                        meminit=False)
        keys_cache_file = os.path.join(dataroot, '_keys_cache.pkl')
        if os.path.isfile(keys_cache_file):
            with open(keys_cache_file, 'rb') as f_keys:
                keys = pickle.load(f_keys)
        else:
            raise FileNotFoundError('{} not found'.format(keys_cache_file))
        names = sorted([
            key for key in keys
            if not key.endswith('.meta') and not key.endswith('.lbl')
        ])
        return env, names

    def _get_imgs_from_lmdb(self, env, name):
        with env.begin(write=False) as txn:
            buf = txn.get(name.encode('ascii'))
            buf_meta = txn.get('.'.join([name, 'meta'
                                         ]).encode('ascii')).decode('ascii')
            buf_lbl = txn.get('.'.join([name, 'lbl']).encode('ascii'))
        img_flat = np.frombuffer(buf, dtype=np.uint8)
        H, W, C = [int(s) for s in buf_meta.split(',')]
        img = img_flat.reshape(H, W, C)
        lbl = np.frombuffer(buf_lbl).reshape(-1, 5)
        return img, lbl

    def __getitem__(self, index):
        # Only 3-channels image will be loaded
        while True:
            if self.lmdb:
                name = self.names[index % len(self.names)]
                # img, cv2 format, height, width, BGRA, 0-255 unit8
                img, lbl = self._get_imgs_from_lmdb(self.env, name)
                img = img[..., 2::-1]
                img_path = os.path.expanduser(
                    '~/data/dota/{}_crop/{}.png'.format(self.set_type, name))
            else:
                img_path = self.img_files[index % len(self.img_files)].strip()
                # (height, width, (R, G, B, A)), 0-255 uint8
                img = np.array(Image.open(img_path))
            if img.shape[2] == 4:
                img = img[..., :3]
                break
            elif img.shape[2] == 3:
                break
            elif img.shape[2] == 2:
                index += 1
                continue
            elif img.shape[2] == 1:
                img = np.concatenate((img, img, img), axis=2)
                break
        # Get labels
        if self.lmdb:
            labels = copy.deepcopy(lbl)
            #  print(labels.flags)
        else:
            label_path = self.label_files[index % len(self.img_files)].strip()
            labels = None
            boxes = None
            classes = None
            if os.path.exists(label_path):
                labels = np.loadtxt(label_path).reshape(-1, 5)
        # Shuffle Boxes!!!
        np.random.shuffle(labels)
        boxes = labels[:, 1:].reshape(-1, 4)
        classes = labels[:, 0].reshape(-1, 1)

        # augmentations
        if self.augs is not None:
            img, boxes, classes = self.augs(img, boxes, classes)

            if (boxes is not None) and (labels is not None):
                labels = np.concatenate((classes, boxes), axis=1)

        # ---------
        #  Image
        # ---------
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        if h <= w:
            pad = ((pad1, pad2), (0, 0), (0, 0))
        else:
            pad = ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Normalized by mean
        mean = np.array([0.5, 0.5, 0.5])
        input_img -= mean
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        # ---------
        #  Label
        # ---------
        if labels is not None:
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3] / 2)
            y1 = h * (labels[:, 2] - labels[:, 4] / 2)
            x2 = w * (labels[:, 1] + labels[:, 3] / 2)
            y2 = h * (labels[:, 2] + labels[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            num_gts = min(len(labels), self.max_objects)
            filled_labels[:num_gts] = copy.deepcopy(labels[:num_gts])
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels


##########################

    def another_get_item(self, index):
        img_id, img = self.get_raw_image(index)
        if img is None:
            print('Image is illegal! Accepted mode "L", "RGB", "RGBA"')
            return None, None, None
        labels = self.get_raw_labels(img_id)
        img, filled_labels = self.transforms(img, labels)
        return img, filled_labels

    def transforms(self, img, labels=None, augs=None):
        """ img and labels transforming
        np.array raw_img and labels"""
        # (height, width, (R, G, B)), 0-255 uint8
        height, width, _ = img.shape
        # Augmentations
        if augs is not None:
            classes = labels[:, 0].reshape(-1, 1)
            boxes = labels[:, 1:].reshape(-1, 4)
            img, boxes, classes = augs(img, boxes, classes)
            labels = np.concatenate((classes, boxes), axis=1)
        # Base transform
        img, labels = to_absolute(img, labels)
        img, labels = square_padding(img, labels)
        img, labels = img_resize(img, labels, self.img_size)
        img, labels = to_relative(img, labels)
        # Mean subtract
        img = img.astype(np.float32) / 255.
        img, labels = mean_subtract(img, labels)

        ###########
        # To Tensor
        ###########

        # Channels-first
        img = np.transpose(img, (2, 0, 1))
        # As pytorch tensor
        img = torch.from_numpy(img).float()

        # Fill matix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            num_gts = range(len(labels))[:self.max_objects]
            filled_labels[num_gts] = labels[:self.max_objects]
            filled_labels = torch.from_numpy(filled_labels)
        return img, filled_labels

    def get_raw_image(self, index):
        img_path = self.img_files[index % len(self.img_files)].strip()
        try:
            img = Image.open(img_path)
        except FileNotFoundError:
            print('No such file: ', img_path)
            print('You should delete the path!')
            return None, None
        img_mode = img.mode
        img = np.array(img)
        if img_mode == 'RGB':
            pass
        elif img_mode == 'RGBA':
            img = img[..., :3]
        elif img_mode == 'L':
            # greyscale
            img = np.stack((img, img, img), axis=2)
        else:
            return None, None
        img_id = self.ids[index]
        return img_id, img

    def get_raw_labels(self, img_id):
        """xywh, range 0-1"""
        label_path = self.labels_path.format(img_id)
        try:
            labels = np.loadtxt(label_path)
        except (IOError, OSError, FileNotFoundError) as e:
            print('No labels for img', img_id)
            print('It will be None as default.')
            return None
        labels = labels.reshape(-1, 5)
        np.random.shuffle(labels)
        return labels

    def basic_transform(self, img, labels):
        # Base transform
        img, labels = to_absolute(img, labels)
        img, labels = square_padding(img, labels)
        img, labels = img_resize(img, labels, self.img_size)
        img, labels = to_relative(img, labels)
        return img, labels

    def raw_transformed(self, index):
        # Only 3-channels image will be loaded
        while True:
            if self.lmdb:
                name = self.names[index % len(self.names)]
                # img, cv2 format, height, width, BGRA, 0-255 unit8
                img, lbl = self._get_imgs_from_lmdb(self.env, name)
                img = img[..., 2::-1]
                img_path = os.path.expanduser(
                    '~/data/dota/{}_crop/{}.png'.format(self.set_type, name))
            else:
                img_path = self.img_files[index % len(self.img_files)].strip()
                # (height, width, channels), 0-255 uint8, just like cv2.imread
                img = np.array(Image.open(img_path))
            if img.shape[2] == 4:
                img = img[..., :3]
                break
            elif img.shape[2] == 3:
                break
            elif img.shape[2] == 2:
                index += 1
                continue
            elif img.shape[2] == 1:
                img = np.concatenate((img, img, img), axis=2)
                break

        # Raw infomation
        raw_img = copy.deepcopy(img)
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)

        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        if h <= w:
            pad = ((pad1, pad2), (0, 0), (0, 0))
        else:
            pad = ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Normalized by mean
        mean = np.array([0.5, 0.5, 0.5])
        input_img -= mean
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()
        return img_path, raw_img, input_img

    def get_label(self, name):
        with self.env.begin(write=False) as txn:
            buf_meta = txn.get('.'.join([name, 'meta'
                                         ]).encode('ascii')).decode('ascii')
            buf_lbl = txn.get('.'.join([name, 'lbl']).encode('ascii'))
        height, width, C = [int(s) for s in buf_meta.split(',')]
        lbl = np.frombuffer(buf_lbl).reshape(-1, 5)

        label_path = self.labels_path.format(self.set_type, name)
        labels = None
        if os.path.exists(label_path):
            # c, xc, yc, w, h
            labels = copy.deepcopy(lbl)
            scale_back_labels = copy.deepcopy(labels)
            # Back to x1 y1 x2 y2
            scale_back_labels[:, 1:3] = labels[:, 1:3] - labels[:, 3:5] / 2
            scale_back_labels[:, 3:5] = labels[:, 1:3] + labels[:, 3:5] / 2
            # Back to the original shape
            scale_back_labels[:, 1:] *= (width, height, width, height)
        else:
            scale_back_labels = None
        return label_path, labels, scale_back_labels

    def __len__(self):
        if self.lmdb:
            return len(self.names)
        else:
            return len(self.img_files)


class NwpuvhrDataset(Dataset):
    """Dataset Loading"""

    def __init__(self, list_path, img_size=416):
        self.set_type = list_path.strip().split('/')[-1].split('.')[0]

        with open(list_path, 'r') as fpath:
            self.img_files = fpath.readlines()
        self.labels_path = os.path.expanduser(
            '~/code/yolov3_origin/data/nwpuvhr/labels/{}.txt')
        self.ids = [
            x.strip().split('/')[-1].split('.')[0] for x in self.img_files
        ]
        self.label_files = [self.labels_path.format(name) for name in self.ids]
        self.img_size = img_size
        self.img_shape = (img_size, img_size)
        self.max_objects = 70  # max is 67 in my trainval
        self.name = 'NWPUVHR'
        # load classes
        self.classes = load_classes(
            os.path.expanduser(
                '~/code/yolov3_origin/data/nwpuvhr/nwpuvhr.names'))
        self.augs = Augmentation(self.img_size)  # data augmentations

    def __getitem__(self, index):
        # Only 3-channels image will be loaded
        while True:
            img_path = self.img_files[index % len(self.img_files)].strip()
            # (height, width, (R, G, B)), 0-255 uint8
            img = np.array(Image.open(img_path))
            if len(img.shape) != 3:
                index += 1
            else:
                break
        # Get labels
        label_path = self.label_files[index % len(self.img_files)].strip()
        labels = None
        boxes = None
        classes = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Shuffle Boxes!!!
            np.random.shuffle(labels)
            boxes = labels[:, 1:].reshape(-1, 4)
            classes = labels[:, 0].reshape(-1, 1)

        # augmentations
        if self.augs is not None:
            img, boxes, classes = self.augs(img, boxes, classes)

            if (boxes is not None) and (labels is not None):
                labels = np.concatenate((classes, boxes), axis=1)

        # ---------
        #  Image
        # ---------
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        if h <= w:
            pad = ((pad1, pad2), (0, 0), (0, 0))
        else:
            pad = ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Normalized by mean
        mean = np.array([0.5, 0.5, 0.5])
        input_img -= mean
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        # ---------
        #  Label
        # ---------
        if labels is not None:
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3] / 2)
            y1 = h * (labels[:, 2] - labels[:, 4] / 2)
            x2 = w * (labels[:, 1] + labels[:, 3] / 2)
            y2 = h * (labels[:, 2] + labels[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            num_gts = min(len(labels), self.max_objects)
            filled_labels[:num_gts] = copy.deepcopy(labels[:num_gts])
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

##########################

    def another_get_item(self, index):
        img_id, img = self.get_raw_image(index)
        if img is None:
            print('Image is illegal! Accepted mode "L", "RGB", "RGBA"')
            return None, None, None
        labels = self.get_raw_labels(img_id)
        img, filled_labels = self.transforms(img, labels)
        return img, filled_labels

    def transforms(self, img, labels=None, augs=None):
        """ img and labels transforming
        np.array raw_img and labels"""
        # (height, width, (R, G, B)), 0-255 uint8
        height, width, _ = img.shape
        # Augmentations
        if augs is not None:
            classes = labels[:, 0].reshape(-1, 1)
            boxes = labels[:, 1:].reshape(-1, 4)
            img, boxes, classes = augs(img, boxes, classes)
            labels = np.concatenate((classes, boxes), axis=1)
        # Base transform
        img, labels = to_absolute(img, labels)
        img, labels = square_padding(img, labels)
        img, labels = img_resize(img, labels, self.img_size)
        img, labels = to_relative(img, labels)
        # Mean subtract
        img = img.astype(np.float32) / 255.
        img, labels = mean_subtract(img, labels)

        ###########
        # To Tensor
        ###########

        # Channels-first
        img = np.transpose(img, (2, 0, 1))
        # As pytorch tensor
        img = torch.from_numpy(img).float()

        # Fill matix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            num_gts = range(len(labels))[:self.max_objects]
            filled_labels[num_gts] = labels[:self.max_objects]
            filled_labels = torch.from_numpy(filled_labels)
        return img, filled_labels

    def get_raw_image(self, index):
        img_path = self.img_files[index % len(self.img_files)].strip()
        try:
            img = Image.open(img_path)
        except FileNotFoundError:
            print('No such file: ', img_path)
            print('You should delete the path!')
            return None, None
        img_mode = img.mode
        img = np.array(img)
        if img_mode == 'RGB':
            pass
        elif img_mode == 'RGBA':
            img = img[..., :3]
        elif img_mode == 'L':
            # greyscale
            img = np.stack((img, img, img), axis=2)
        else:
            return None, None
        img_id = self.ids[index]
        return img_id, img

    def get_raw_labels(self, img_id):
        """xywh, range 0-1"""
        label_path = self.labels_path.format(img_id)
        try:
            labels = np.loadtxt(label_path)
        except (IOError, OSError, FileNotFoundError) as e:
            print('No labels for img', img_id)
            print('It will be None as default.')
            return None
        labels = labels.reshape(-1, 5)
        np.random.shuffle(labels)
        return labels

    def basic_transform(self, img, labels):
        # Base transform
        img, labels = to_absolute(img, labels)
        img, labels = square_padding(img, labels)
        img, labels = img_resize(img, labels, self.img_size)
        img, labels = to_relative(img, labels)
        return img, labels

    def raw_transformed(self, index):
        # Only 3-channels image will be loaded
        while True:
            img_path = self.img_files[index % len(self.img_files)].strip()
            # (height, width, channels), 0-255 uint8, just like cv2.imread
            img = np.array(Image.open(img_path))
            if len(img.shape) == 2:  # h, w
                img = np.stack((img, img, img), axis=2)
                break
            elif len(img.shape) == 3:
                if img.shape[2] == 3:
                    break
                elif img.shape[2] == 4:  # h, w, 4
                    img = img[..., :3]
                    break
                else:
                    continue
            else:
                continue

        # Raw infomation
        raw_img = copy.deepcopy(img)
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)

        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        if h <= w:
            pad = ((pad1, pad2), (0, 0), (0, 0))
        else:
            pad = ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Normalized by mean
        mean = np.array([0.5, 0.5, 0.5])
        input_img -= mean
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()
        return img_path, raw_img, input_img

    def get_label(self, imagename):
        label_path = self.labels_path.format(imagename)
        labels = None
        if os.path.exists(label_path):
            img_path = self.img_files[self.ids.index(imagename)].strip()
            # (height, width, channels), 0-255 uint8, just like cv2.imread
            img = np.array(Image.open(img_path))
            height, width, _ = img.shape

            # c, xc, yc, w, h
            labels = np.loadtxt(label_path).reshape(-1, 5)
            scale_back_labels = copy.deepcopy(labels)
            # Back to x1 y1 x2 y2
            scale_back_labels[:, 1:3] = labels[:, 1:3] - labels[:, 3:5] / 2
            scale_back_labels[:, 3:5] = labels[:, 1:3] + labels[:, 3:5] / 2
            # Back to the original shape
            scale_back_labels[:, 1:] *= (width, height, width, height)
        else:
            scale_back_labels = None
        return label_path, labels, scale_back_labels

    def __len__(self):
        return len(self.img_files)


###########################


def to_absolute(img, labels=None):
    """ range 0-1 to range w,h"""
    height, width, _ = img.shape
    try:
        labels[:, 1:] *= (width, height, width, height)
    except TypeError as e:
        print(e)
    return img, labels


def to_relative(img, labels=None):
    """range w,h to 0-1"""
    height, width, _ = img.shape
    try:
        labels[:, 1:] /= (width, height, width, height)
    except TypeError as e:
        print(e)
    return img, labels


def to_xyxy(img, labels=None):
    """absolute"""
    try:
        labels[:, 1:3] -= labels[:, 3:] / 2
        labels[:, 3:] += (labels[:, 1:3] - 1)
    except TypeError as e:
        print(e)
    return img, labels


def to_xywh(img, labels=None):
    """absolute"""
    try:
        labels[:, 3:] -= (labels[:, 1:3] - 1)
        labels[:, 1:3] += (labels[:, 3:] / 2)
    except TypeError as e:
        print(e)
    return img, labels


#############################


def square_padding(img, labels=None):
    """np.array raw_image and labels"""
    h, w, _ = img.shape
    dim_diff = np.abs(h - w)
    # Upper (left) and lower (right) padding
    pad1 = dim_diff // 2
    pad2 = dim_diff - pad1
    # Determine padding
    if h <= w:
        pad = ((pad1, pad2), (0, 0), (0, 0))
    else:
        pad = ((0, 0), (pad1, pad2), (0, 0))
    # Add padding
    img = np.pad(img, pad, 'constant', constant_values=127)

    # labels (xywh)
    try:
        labels[:, 1:3] += (pad[1][0], pad[0][0])
    except TypeError as e:
        print(e)
    return img, labels


def img_resize(img, labels=None, input_size=416):
    # labels xywh
    # Resize and normalize
    height, width, _ = img.shape

    img = resize(img, (input_size, input_size, 3), mode='reflect')
    padded_h, padded_w, _ = img.shape

    try:
        factors = (padded_w / width, padded_h / height)
        labels[:, 1:] *= (*factors, *factors)
    except TypeError as e:
        print(e)
    return img, labels


def mean_subtract(img, labels=None, means=None):
    # img, float32, 0-1
    # Normalized by mean
    if means is None:
        means = np.array([0.5, 0.5, 0.5])
    img -= means
    return img, labels
