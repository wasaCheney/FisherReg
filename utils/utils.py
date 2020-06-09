from __future__ import division
import os
import pickle
import time

import numpy as np
import lmdb
import torch
import cv2

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp',
    '.BMP'
]

####################
# Files & IO
####################


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def _get_paths_from_lmdb(dataroot):
    env = lmdb.open(dataroot,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False)
    keys_cache_file = os.path.join(dataroot, '_keys_cache.pkl')
    if os.path.isfile(keys_cache_file):
        keys = pickle.load(open(keys_cache_file, "rb"))
    else:
        print('File not found: {}'.format(keys_cache_file))
        exit()
    paths = sorted([
        key for key in keys
        if not key.endswith('.meta') and not key.endswith('.lbl')
    ])
    return env, paths


def get_image_paths(data_type, dataroot):
    env, paths = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            env, paths = _get_paths_from_lmdb(dataroot)
        elif data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(data_type))
    return env, paths


def _read_lmdb_img(env, path):
    with env.begin(write=False) as txn:
        buf = txn.get(path.encode('ascii'))
        buf_meta = txn.get('.'.join([path,
                                     'meta']).encode('ascii')).decode('ascii')
        buf_lbl = txn.get('.'.join([path, 'lbl']).encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    H, W, C = [int(s) for s in buf_meta.split(',')]
    img = img_flat.reshape(H, W, C)
    lbl = np.frombuffer(buf_lbl).reshape(-1, 5)
    return img, lbl


def read_img(env, path):
    # read image by cv2 or from lmdb
    # return: Numpy float32, HWC, BGR, [0,1]
    if env is None:  # img
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        img, lbl = _read_lmdb_img(env, path)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img, lbl


def create_vis_plot(viz, xlabel_, ylabel_, title_, legend_):
    num_lines = len(legend_)
    win = viz.line(X=torch.zeros((1, )).cpu(),
                   Y=torch.zeros((1, num_lines)).cpu(),
                   opts=dict(xlabel=xlabel_,
                             ylabel=ylabel_,
                             title=title_,
                             legend=legend_))
    return win


def update_vis(viz, window, xaxis, *args):
    yaxis = torch.Tensor([args]).cpu()
    viz.line(X=torch.Tensor([xaxis]).cpu(),
             Y=yaxis,
             win=window,
             update='append')


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    with open(path, 'r') as fcls:
        names = fcls.readlines()
    names = [name.strip() for name in names if name.strip()]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = (box1[..., 0] - box1[..., 2] / 2,
                        box1[..., 0] + box1[..., 2] / 2)
        b1_y1, b1_y2 = (box1[..., 1] - box1[..., 3] / 2,
                        box1[..., 1] + box1[..., 3] / 2)
        b2_x1, b2_x2 = (box2[..., 0] - box2[..., 2] / 2,
                        box2[..., 0] + box2[..., 2] / 2)
        b2_y1, b2_y2 = (box2[..., 1] - box2[..., 3] / 2,
                        box2[..., 1] + box2[..., 3] / 2)
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = (box1[..., 0], box1[..., 1], box1[..., 2],
                                      box1[..., 3])
        b2_x1, b2_y1, b2_x2, b2_y2 = (box2[..., 0], box2[..., 1], box2[..., 2],
                                      box2[..., 3])

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = (torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) *
                  torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0))
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction,
                        num_classes,
                        score_thres=0.5,
                        nms_thres=0.4):
    """
    Removes detections with lower score than 'score_thres'
    and performs
    Non-Maximum Suppression to further filter detections.
    Arguments:
        prediction, (bs, num_pred_bboxes, 5+C)
        score, is corresongding to conf_prob*cls_prob
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[..., 0] = prediction[..., 0] - prediction[..., 2] / 2
    box_corner[..., 1] = prediction[..., 1] - prediction[..., 3] / 2
    box_corner[..., 2] = prediction[..., 0] + prediction[..., 2] / 2
    box_corner[..., 3] = prediction[..., 1] + prediction[..., 3] / 2
    prediction[..., :4] = box_corner[..., :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Get scores for every box
        #  xyxy = image_pred[:, :4].view(-1, 4)
        conf_prob = image_pred[:, 4].view(-1, 1)
        cls_prob = image_pred[:, 5:].view(-1, num_classes)
        score = conf_prob * cls_prob
        # Mask
        #  conf_mask = conf_prob >= score_thres
        #  cls_prob_max = torch.max(cls_prob, 1, keepdim=True)[0]
        #  cls_mask = cls_prob_max >= score_thres
        #  score_mask = (conf_mask * cls_mask).squeeze()
        # Mask out boxes with lower score
        score_max = torch.max(score, 1, keepdim=False)[0]
        score_mask = (score_max >= score_thres)
        image_pred = image_pred[score_mask]
        score = score[score_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_score, class_pred = torch.max(score, 1, keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_prob, cls_score, cls_pred)
        # entries must be the same type in a Tensor
        detections = torch.cat(
            (image_pred[:, :5], class_score.float(), class_pred.float()), 1)

        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by
            # score (p(obj)*p(cls|obj)) rather than cls_prob!!!
            _, conf_sort_index = torch.sort(detections_class[:, 5],
                                            descending=True)
            detections_class = detections_class[conf_sort_index]

            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data

            # Add max detections to outputs
            if output[image_i] is None:
                output[image_i] = max_detections
            else:
                output[image_i] = torch.cat((output[image_i], max_detections))

    return output  # Its entry could be None or torch.Tensor


def build_targets(pred_boxes, target, anchors, all_anchors, num_anchors,
                  num_classes, dim, ignore_thres, img_dim):
    """
    Arguments:
        pred_boxes, scaled relative to the output size
        target, (_, xc, yc, w, h), relative values to the input size
        anchors, scaled anchors shape relative to the output size
        all_anchors, scaled all_anchors shape relative to the output size
        num_anchors, len(anchors)
        dim, the output size
        ignore_thres, boxes with IOU>ignore_thres will be ingnored
        img_dim, the input size"""
    nB = target.size(0)  # batch size
    nA = num_anchors  # num_anchors with different shapes
    nC = num_classes
    dim = dim  # g_dim, how many x_center (or y_center) to predict
    mask = torch.zeros(nB, nA, dim, dim)  # positive bbox
    # not be ignored bbox
    conf_mask = torch.ones(nB, nA, dim, dim)
    tx = torch.zeros(nB, nA, dim, dim)
    ty = torch.zeros(nB, nA, dim, dim)
    tw = torch.zeros(nB, nA, dim, dim)
    th = torch.zeros(nB, nA, dim, dim)
    tconf = torch.zeros(nB, nA, dim, dim)
    tcls = torch.zeros(nB, nA, dim, dim, nC)
    # to balance losses of boxes wih different size
    #  box_loss_scale = torch.zeros(nB, nA, dim, dim)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            #  time_begin = time.time()
            if target[b, t, 1:].sum() == 0:
                continue
            nGT += 1
            # get the area of GT for box_loss_scale
            #  box_area = target[b, t, 3] * target[b, t, 4]
            # Convert to position relative to the output size
            #  print(target[b, t])
            gx = target[b, t, 1] * dim
            gy = target[b, t, 2] * dim
            gw = target[b, t, 3] * dim
            gh = target[b, t, 4] * dim
            # Get grid box indices
            # Transform might make index out of range
            gi = min(int(gx), dim - 1)
            gj = min(int(gy), dim - 1)
            gt_shape = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # Get shape of all anchor box
            all_A = len(all_anchors)
            all_anchor_shapes = (np.zeros((all_A, 2)), np.array(all_anchors))
            all_anchor_shapes = torch.FloatTensor(
                np.concatenate(all_anchor_shapes, 1))
            all_anch_ious = bbox_iou(gt_shape, all_anchor_shapes)
            all_best_n = np.argmax(all_anch_ious)
            best_shape = all_anchors[all_best_n]

            shape_inds = [(best_shape[0] == ele[0]) * (best_shape[1] == ele[1])
                          for ele in anchors]
            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            # Get the interior point of the GT
            top = int(max(0, gy - gh / 2))
            bottom = int(min(dim, gy + gh / 2 + 1))
            left = int(max(0, gx - gw / 2))
            right = int(min(dim, gi + gw / 2 + 1))
            #  top = 0
            #  bottom = dim
            #  left = 0
            #  right = dim
            # Calculate IOUs of interior bboxes and the GT
            i_bboxes = pred_boxes[b, :, top:bottom, left:right]
            i_ious = bbox_iou(gt_box, i_bboxes, x1y1x2y2=False)
            ignore_index = i_ious >= ignore_thres
            # Mask out boxes with IOU>ignore_thres
            # but the positive one may not be excluded
            conf_mask[b, :, top:bottom, left:right][ignore_index] = 0

            if any(shape_inds):
                # best shape in this level
                best_n = np.argmax(shape_inds)
                # Get the best prediction
                pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
                # GT Masks
                mask[b, best_n, gj, gi] = 1
                # Coordinates
                # It is possible that two different objs have the same
                # best_n and gj, gi, but very rarely
                tx[b, best_n, gj, gi] = gx - gi
                ty[b, best_n, gj, gi] = gy - gj
                # Width and height
                tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][0] +
                                                  1e-16)
                th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][1] +
                                                  1e-16)
                # box loss scale
                #  box_loss_scale[b, best_n, gj, gi] = max(
                #  1, 1.5 - box_area**0.5)
                # One-hot encoding of label
                tcls[b, best_n, gj, gi, int(target[b, t, 0])] = 1
                # Objectness
                tconf[b, best_n, gj, gi] = 1
                # Calculate iou between ground truth and the best matching
                iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
                if iou > 0.5:
                    nCorrect += 1
            #  time_end = time.time()
            #  print(t, '====== matching time (s): ====',
            #  time_end - time_begin)
    conf_mask = (1 - mask) * conf_mask  # real negatives

    return (nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls
            )  # , box_loss_scale)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.from_numpy(np.eye(num_classes, dtype='uint8')[y])
