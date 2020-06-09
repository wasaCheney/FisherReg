'''Data augmentations from ssd'''
import copy
import numpy as np
from numpy import random
from skimage.transform import resize
import cv2
import torch


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        cv2image = tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0))
        return cv2image, boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        tensor_ = torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1)
        return tensor_, boxes, labels


#################
# Compute IOUs
#################


def intersect(box_a, box_b):
    """xyxy"""
    max_xy = np.minimum(box_a[..., 2:], box_b[..., 2:])
    min_xy = np.maximum(box_a[..., :2], box_b[..., :2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[..., 0] * inter[..., 1]  # num_a, num_b


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_b.shape[0]]
    """
    box_a_re = box_a.reshape(-1, 1, 4)
    box_b_re = box_b.reshape(1, -1, 4)

    inter = intersect(box_a_re, box_b_re)
    area_a = ((box_a_re[..., 2] - box_a_re[..., 0]) *
              (box_a_re[..., 3] - box_a_re[..., 1]))
    area_b = ((box_b_re[..., 2] - box_b_re[..., 0]) *
              (box_b_re[..., 3] - box_b_re[..., 1]))
    union = area_a + area_b - inter
    return inter / union  # (num_A, num_B)


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for trans in self.transforms:
            img, boxes, labels = trans(img, boxes, labels)
        return img, boxes, labels


###############
# augmentations
###############


# 1
class ConvertFromInts(object):
    """np.array, (height, width, (R, G, B)), unit8"""
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


# 2
class ToAbsoluteCoords(object):
    """The original boxes is relaitve (divied by height or width)
    So they need to be transformed back
    Range 0-1 to Range w,h
    """
    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        try:
            boxes *= (width, height, width, height)
        except TypeError as e:
            print(e)
        return image, boxes, labels


# 2-1
class ToXyxy(object):
    """
    w,h; xywh -> xyxy
    """
    def __call__(self, image, boxes=None, labels=None):
        try:
            boxes[:, :2] -= boxes[:, 2:] / 2
            boxes[:, 2:] += (boxes[:, :2] - 1)
        except TypeError as e:
            print(e)
        return image, boxes, labels


##################################################
# Photometric for RGB np.array with float elements
##################################################


# 3-1
class RandomBrightness(object):
    r""" In RGB space, brightness could be thought as arithmetic mean $\mu$
    of R, G, B, that is to say,
    $\mu = \frac{R + G + B}{3}$"""
    def __init__(self, delta=32):
        assert delta >= 0
        assert delta <= 255
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            # Could be negative
            delta = random.uniform(-self.delta, self.delta)
            image += delta
            image[image > 255] = 255.
            image[image < 0] = 0.
        return image, boxes, labels


# 3-x
class RandomContrast(object):
    """Hard to clarify, treat it as the difference between
    maximum and minimum"""
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
            image[image > 255.] = 255.
            image[image < 0.] = 0.
        return image, boxes, labels


# 3-x
# cv2.cvtColor, pay attention to element value's dtype and range
class ConvertColor(object):
    """np.array
    HSV is (hue, saturation, value)"""
    def __init__(self, current='RGB', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'RGB' and self.transform == 'HSV':
            # RGB to BGR, which is default order for cv2
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'HSV' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, boxes, labels


# 3-x between ConvertColor
class RandomHue(object):
    """In HSV space, hue is corresponding to angle
    if float, Hue Range [0., 360.]"""
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[..., 0] += random.uniform(-self.delta, self.delta)
            image[..., 0][image[..., 0] > 360.0] -= 360.0
            image[..., 0][image[..., 0] < 0.0] += 360.0
        return image, boxes, labels


# 3-x between ConvertColor
class RandomSaturation(object):
    """HSV image, if float,  S range [0., 1.]"""
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[..., 1] *= random.uniform(self.lower, self.upper)
            image[image[..., 1] > 1] = 1.
            image[image[..., 1] < 0] = 0.

        return image, boxes, labels


# 3-x between ConvertColor
class RandomValue(object):
    """HSV image, if float, Value range [0., 255.]"""
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[..., 2] *= random.uniform(self.lower, self.upper)
            image[image[..., 2] > 255] = 255.
            image[image[..., 2] < 0] = 0.

        return image, boxes, labels


# 3-x
class RandomLightingNoise(object):
    def __init__(self):
        self.channels = np.array([0, 1, 2])

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            temp_channels = self.channels.copy()
            random.shuffle(temp_channels)
            image = image[..., temp_channels]
        return image, boxes, labels


# 3 all
class PhotometricDistort(object):
    """(height, width, (R,G,B)) np.array, float.
    After the operation, element value could be negative or greater than 255
    """
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomHue(),
            RandomSaturation(),
            ConvertColor(current='HSV', transform='RGB'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


# 4 do not
class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)  # left padding
        top = random.uniform(0, height*ratio - height)  # top padding

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean  # padding values
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


# 5
class RandomCrop(object):
    """Crop, Range x,y; xywh
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in xyxy form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self, min_prop=0.7):
        self.min_prop = min_prop

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            pass
        else:
            origin_image = copy.deepcopy(image)
            height, width, _ = origin_image.shape
            # max try 10
            for _ in range(20):
                # Width and height
                crop_w = random.uniform(self.min_prop, 1) * width
                crop_h = random.uniform(self.min_prop, 1) * height
                # aspect ratio constraint b/t .5 & 2
                if not 0.5 <= (crop_w / crop_h) <= 2:
                    continue
                # Feasible left-top point range
                max_x = width - crop_w
                max_y = height - crop_h
                # Left-top point
                left = random.uniform(0, max_x)
                top = random.uniform(0, max_y)
                # Xyxy
                rect = (int(left), int(top), int(left+crop_w), int(top+crop_h))
                # cut the crop from the image
                image = origin_image[rect[1]:rect[3], rect[0]:rect[2], :]
                if boxes is None:
                    break
                # keep overlap with gt box IF center in sampled patch
                centers = boxes[:, :2].reshape(-1, 2)
                leftmask = rect[0] + 1 < centers[:, 0]
                rightmask = centers[:, 0] < rect[2] - 1
                maskx = leftmask * rightmask
                topmask = rect[1] + 1 < centers[:, 1]
                bottommask = centers[:, 1] < rect[3] - 1
                masky = topmask * bottommask
                mask = maskx * masky
                # Any valid boxes?
                if not mask.any():
                    continue
                # take only matching gt boxes
                boxes = boxes[mask].reshape(-1, 4)
                boxes[:, :2] -= rect[:2]  # xy
                boxes[:, 2:] = np.minimum(boxes[:, 2:], (crop_w, crop_h))
                # take only matching gt labels
                labels = labels[mask].reshape(-1, 1)
                break

        return image, boxes, labels


# 5-1
class ToXywh(object):
    """ Range w,h; xyxy -> xywh"""
    def __call__(self, image, boxes=None, classes=None):
        try:
            boxes[:, 2:] -= (boxes[:, :2] - 1)
            boxes[:, :2] += (boxes[:, 2:] / 2)
        except TypeError as e:
            print(e)
        return image, boxes, classes


# 6
class RandomMirror(object):
    # x y w h
    def __call__(self, image, boxes=None, classes=None):
        height, width, _ = image.shape
        # Horizontal
        if random.randint(2):
            image = image[:, ::-1]
            try:
                boxes[:, 0] = width - boxes[:, 0]
            except TypeError as e:
                print(e)
        # Vertical
        if random.randint(2):
            image = image[::-1]
            try:
                boxes[:, 1] = height - boxes[:, 1]
            except TypeError as e:
                print(e)
        return image, boxes, classes


# 7
class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        try:
            boxes /= (width, height, width, height)
        except TypeError as e:
            print(e)
        return image, boxes, labels


# 8
class Jitter(object):
    """Random aspect_ratio and scale; Range w,h; xywh"""
    def __init__(self, jitter=0.2, scale=0.2):
        self.jitter = jitter
        self.scale = scale

    def __call__(self, image, boxes=None, labels=None):
        # Paras
        height, width, _ = image.shape
        new_aspect_ratio = random.uniform(1 - self.jitter, 1 + self.jitter)
        new_scale = random.uniform(1 - self.scale, 1 + self.scale)
        # Image
        if random.randint(2):
            new_h = int(new_scale * height)
            new_w = int(new_h * new_aspect_ratio)
        else:
            new_w = int(new_scale * width)
            new_h = int(new_w / new_aspect_ratio)
        image /= 255.
        image = resize(image, (new_h, new_w, 3), mode='reflect')
        image *= 255.
        # Box
        try:
            boxes *= (new_w/width, new_h/height, new_w/width, new_h/height)
        except TypeError as e:
            print(e)
        return image, boxes, labels


# 9
class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class Augmentation(object):
    def __init__(self, size=416):
        #  self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            #  Expand(self.mean),
            #  Jitter(),
            #  ToXyxy(),
            #  RandomCrop(),
            #  ToXywh(),
            RandomMirror(),
            ToPercentCoords(),
            #  SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)
