"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
    Adapter: Cheney zhou
"""

import os
from shutil import rmtree
import time
import argparse
import pickle
#  import concurrent.futures

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import visdom

from models import Darknet
from utils.datasets import NwpuvhrDataset, DotaDataset
from utils.utils import non_max_suppression
from utils.utils import create_vis_plot, update_vis
#  from utils.parse_config import parse_model_config


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        return self.diff


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


####################################
# Write predictions cls by cls, .txt
####################################


def get_voc_results_file_template(dataset, cls, output_dir):
    """Filename to save predictions cls by cls"""
    # output_dir/results/det_test_airplane.txt
    filename = 'det_{:s}_{:s}.txt'.format(dataset.set_type, cls)
    filedir = os.path.join(output_dir, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset, output_dir):
    # all_boxes, list, (num_classes, num_images, xxx, 5)
    # output_dir/results/det_test_clsname.txt:
    # image_name, scores, xmin, ymin, xmax, ymax
    for cls_ind, cls in enumerate(dataset.classes):
        print('Writing {:s} {} results file'.format(cls, dataset.name))
        filename = get_voc_results_file_template(dataset, cls, output_dir)
        with open(filename, 'w') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                        index, dets[k, -1], dets[k, 0] + 1, dets[k, 1] + 1,
                        dets[k, 2] + 1, dets[k, 3] + 1))


###############################################
# Get predictions from .txt and sorted
###############################################


def get_preds(det_file):
    """Det_file contains bboxes with the same class
    Returns:
        image_ids, str
        BB, bboxes sorted by confidence"""
    with open(det_file, 'r') as fdet:
        lines = fdet.readlines()
    if any(lines) == 1:
        splitlines = [x.strip().split(' ') for x in lines]  # split with space!
        # Get imagename, confidence, and bbox for cls
        image_ids = []
        confidence = []
        BB = []
        for x in splitlines:
            # imagename
            image_ids.append(x[0])
            # Cls_confidence
            confidence.append(float(x[1]))
            # Cls_bobx array shaped as (xxx, 4)
            BB.append([float(z) for z in x[2:]])
        confidence = np.array(confidence)
        BB = np.array(BB)

        # sort by confidence, descending order
        sorted_ind = np.argsort(confidence)[::-1]
        #  sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
    else:
        image_ids = None
        BB = None
    return image_ids, BB


#########################
# Get GTs from annots.pkl
#########################


def get_annots(cachedir, dataset):
    """ Get GTs from a cache file (if not exists, write)
    Arguments:
        cachedir, path to store cache file
    Returns:
        recs, dict of np.array shaped (-1, 5)
        recs[imagename] could be None!
    """
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')

    imagenames = dataset.ids
    # load annots if exists else save
    # recs: {image_name: np.array shaped (-1, 5)}
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            _, _, recs[imagename] = dataset.get_label(imagename)
            if i % 20 == 0:
                pass
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)
    return recs, imagenames


################
# Get GTs of cls
################


def cls_gts(recs, cls, dataset):
    """Extract gt objects for a given class
    Arguments:
        recs,
    class_recs, dict of dict"""

    cls_ind = dataset.classes.index(cls)

    eps = 0.1

    class_recs = {}
    npos = 0
    for imagename in dataset.ids:
        labels = recs[imagename]
        # If no labels for this image
        if labels is None:
            class_recs[imagename] = {
                'bbox': np.array([]),
                'difficult': np.array([]),
                'det': np.array([])
            }
            continue
        # bbox, array shaped as (xxx_t, 4)
        cls_selected_ind = np.abs(labels[:, 0] - cls_ind) < eps
        bbox = (labels[:, 1:][cls_selected_ind]).reshape(-1, 4)

        # The GT is postive or not (difficult = 0 means positive)
        difficult = np.zeros(len(bbox)).astype(np.bool)

        det = [False] * len(labels)  # Whether a GT is matched or not
        npos = npos + sum(~difficult)  # num_positive for this cls
        class_recs[imagename] = {
            'bbox': bbox,
            'difficult': difficult,
            'det': det
        }
    return class_recs, npos


##########################################
# Calculate VOC-style AP for a given class
##########################################


def voc_ap(rec, prec, use_07_metric=True):
    """For a given class, ap = voc_ap(rec, prec, [use_07_metric])
    Args:
        rec, shape (num_splitlines,)
        prec, shape (num_splitlines,)
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detfile,
             cachedir,
             classname,
             dataset,
             ovthresh=0.5,
             use_07_metric=True):
    """For a given class, do voc_eval
    rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
       detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
       annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5), VOC style but COCO
    AP use 0.5:0.95:0.05
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
       (default True)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # Load GTs
    recs, _ = get_annots(cachedir, dataset)
    class_recs, npos = cls_gts(recs, classname, dataset)
    # Load predictions of cls from .txt
    image_ids, BB = get_preds(detfile)

    if BB is None:
        rec = -1.
        prec = -1.
        ap = -1.
        rec_nodup = -1.
        prec_nodup = -1.
        ap_nodup = -1.
        rec_nofp = -1.
        prec_nofp = -1.
        ap_nofp = -1.
    else:
        ###################
        # Marks TPs and FPs
        ###################

        nd = len(image_ids)  # num of pred_bbox for this cls
        tp = np.zeros(nd)
        dup = np.zeros(nd)  # duplicated detections
        sim_oth_bg = np.zeros(nd)  # fps with any(iou) < thres for this cls.
        fp = np.zeros(nd)

        for d in range(nd):
            R = class_recs[image_ids[d]]  # dict
            BBGT = R['bbox'].astype(float)  # target, (xxx, 4)

            bb = BB[d, :].astype(float)  # predicted, (4,)

            ovmax = -np.inf
            if BBGT.size > 0:
                # compute overlaps and find the  one
                # Intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                # Union
                areabb = (bb[2] - bb[0]) * (bb[3] - bb[1])
                areaBBGT = ((BBGT[:, 2] - BBGT[:, 0]) *
                            (BBGT[:, 3] - BBGT[:, 1]))
                uni = (areabb + areaBBGT) - inters
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1  # mark a matched true box
                    else:
                        dup[d] = 1.
                        fp[d] = 1.
            else:
                sim_oth_bg[d] = 1.
                fp[d] = 1.
        ###########
        # APovthres
        ###########
        fp_thres = np.cumsum(fp)  # (No. of splitlines,) sorted by conf
        tp_thres = np.cumsum(tp)  # (No. of splitlines,) sorted by conf
        rec = tp_thres / float(npos)
        prec = tp_thres / np.maximum(tp_thres + fp_thres,
                                     np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        ###########
        # AP after removing duplicated detections
        ###########
        fp_nodup = np.cumsum(fp[dup == 0.])
        tp_nodup = np.cumsum(tp[dup == 0.])
        rec_nodup = tp_nodup / float(npos)
        prec_nodup = tp_nodup / np.maximum(tp_nodup + fp_nodup,
                                           np.finfo(np.float64).eps)
        ap_nodup = voc_ap(rec_nodup, prec_nodup, use_07_metric)
        ###########
        # AP after removing all FPs
        ###########
        # all 0
        fp_nofp = np.cumsum(fp[(dup == 0.) * (sim_oth_bg == 0.)])
        # all 1
        tp_nofp = np.cumsum(tp[(dup == 0.) * (sim_oth_bg == 0.)])
        rec_nofp = tp_nofp / float(npos)
        prec_nofp = tp_nofp / np.maximum(tp_nofp + fp_nofp,
                                         np.finfo(np.float64).eps)
        ap_nofp = voc_ap(rec_nofp, prec_nofp, use_07_metric)
    return (rec, prec, ap, rec_nodup, prec_nodup, ap_nodup, rec_nofp,
            prec_nofp, ap_nofp)


############################################
# Load GTs and Predictions for mAP computing
# Then Write to disk cls by cls and plot PR
############################################


def do_python_eval(dataset,
                   output_dir='eval',
                   use_07=True,
                   iou_threses=(0.75, 0.5, 0.1)):
    cachedir = os.path.join(output_dir, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    # Get prs
    pr_dict = {}
    for cls_ind, cls in enumerate(dataset.classes):
        filename = get_voc_results_file_template(dataset, cls, output_dir)
        pr_dict[cls] = {}
        for iou_thres in iou_threses:
            (rec, prec, ap, rec_nodup, prec_nodup, ap_nodup, rec_nofp,
             prec_nofp, ap_nofp) = voc_eval(filename,
                                            cachedir,
                                            cls,
                                            dataset,
                                            ovthresh=iou_thres,
                                            use_07_metric=use_07_metric)
            if iou_thres == 0.1:
                pr_dict[cls][iou_thres] = {
                    'rec': rec,
                    'prec': prec,
                    'ap': ap,
                    'rec_nodup': rec_nodup,
                    'prec_nodup': prec_nodup,
                    'ap_nodup': ap_nodup,
                    'rec_nofp': rec_nofp,
                    'prec_nofp': prec_nofp,
                    'ap_nofp': ap_nofp
                }
            else:
                pr_dict[cls][iou_thres] = {'rec': rec, 'prec': prec, 'ap': ap}
                if iou_thres == 0.5:
                    aps += [ap]
                    print('AP{:n} for {} = {:.4f}'.format(
                        100 * iou_thres, cls, ap))
    # Write to disk
    with open(os.path.join(output_dir, 'pr_cls_thres.pkl'), 'wb') as fpr:
        pickle.dump(pr_dict, fpr)
    # Compute mAP
    mean_ap = np.mean(aps)
    # Plot pr_curve for each cls
    num_classes = len(dataset.classes)
    pr_curve(pr_dict, output_dir, dataset.set_type, num_classes, mean_ap)
    print('Mean AP = {:.4f}'.format(mean_ap))
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')
    return aps, mean_ap


#############
# PR Plotting
#############


def pr_curve(pr_dict, output_dir, set_type, num_classes, mean_ap):
    def cls_pr_curve(cls, recs, precs, lbls, clrs, ax):
        ax.set_title(cls)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        # FN
        ax.fill_between(np.linspace(0, 1, 101, endpoint=True),
                        0,
                        1,
                        label='FN: 1.00',
                        alpha=1,
                        linestyle='-',
                        facecolor=clrs[-1])
        # AP75, AP50, Loc, Dup, BG
        for rec, prec, lbl, clr in zip(recs[::-1], precs[::-1], lbls[::-1],
                                       clrs):
            try:
                ax.fill_between(rec,
                                0,
                                prec,
                                label=lbl,
                                linestyle='-',
                                alpha=1,
                                facecolor=clr)
            except:
                continue
        ax.legend(loc='lower left')

    def get_prlbl(cls_dict):
        lbls = []
        recs = []
        precs = []
        # Order is important
        for thres, prs in cls_dict.items():
            recs.append(prs['rec'])
            precs.append(prs['prec'])
            if thres == 0.1:
                # Loc
                lbls.append('Loc: {:.2f}'.format(prs['ap']))
                # Dup
                lbls.append('Dup: {:.2f}'.format(prs['ap_nodup']))
                recs.append(prs['rec_nodup'])
                precs.append(prs['prec_nodup'])
                # BG
                lbls.append('BG: {:.2f}'.format(prs['ap_nofp']))
                recs.append(prs['rec_nofp'])
                precs.append(prs['prec_nofp'])
            else:
                lbls.append('AP{:n}: {:.2f}'.format(thres * 100, prs['ap']))
        return recs, precs, lbls

    # Adjust accoding to num_cls please, better multiples of 5
    rows = int(num_classes // 5)
    fig, axes = plt.subplots(rows, 5,
                             figsize=(20, rows * 4))  # 2000*xxx pixels
    fill_colors = tuple('bgrcmy')
    for cls_ind, (cls, cls_dict) in enumerate(pr_dict.items()):
        ax = axes[cls_ind // 5, cls_ind % 5]
        recs, precs, lbls = get_prlbl(cls_dict)
        cls_pr_curve(cls, recs, precs, lbls, fill_colors, ax)
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir,
                     'prec_rec_{}_{:.6}.png'.format(set_type, mean_ap)))
    plt.close()


################
# Evaluation steps
################


def test_net(output_dir,
             net,
             cuda,
             dataset,
             score_min,
             nms_max,
             use_07_eval=True,
             iou_thres=(0.1, 0.5, 0.75)):
    num_images = len(dataset)
    num_classes = len(dataset.classes)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    [x1, y1, x2, y2, score]
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    #  output_dir = get_output_dir('ssd300_120000', set_type)

    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        _, raw_im, im = dataset.raw_transformed(i)
        h, w, _ = raw_im.shape

        x = Variable(im.unsqueeze(0))  # 1, channels, height, width
        if cuda:
            x = x.cuda()
        _t['im_detect'].tic()

        # When ssd.phase == 'test', net_output is
        # shaped as (batch_size, num_classes, top_k, 5)
        # and the last dim order is (score, xmin, ymin, xmax, ymax)
        detections = net(x)
        # Detection time, averaged
        detect_time = _t['im_detect'].toc(average=True)

        detections = non_max_suppression(detections,
                                         num_classes,
                                         score_thres=score_min,
                                         nms_thres=nms_max)

        # No background cls
        for j in range(num_classes):
            if detections[0] is None:
                continue
            # for class j, shape (xxx, 7)
            # (x1, y1, x2, y2, obj_conf, cls_conf, cls)
            dets = detections[0][detections[0][:, -1] == j]

            # select boxes with score > 0
            #  mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            #  dets = torch.masked_select(dets, mask).view(-1, 5)

            if dets.size(0) == 0:
                continue

            boxes = dets[:, :4]  # (xxx, 4)

            #############################################
            # Transform boxes back according to raw image
            #############################################
            # Unresize
            max_hw = max(h, w)
            scale_hw = max_hw / dataset.img_shape[0]
            boxes *= scale_hw
            # Unpadding
            diff = abs(h - w)
            half_diff = diff // 2
            if h <= w:
                pad = (0, half_diff, 0, diff - half_diff)
            else:
                pad = (half_diff, 0, diff - half_diff, 0)
            boxes -= torch.Tensor(pad).type_as(boxes)

            scores = dets[:, -2].cpu().numpy()

            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets  # (xxx, 5) with location and scores
    # store the predicted boxes and labels as .pkl
    with open(det_file, 'wb') as fdet:
        pickle.dump(all_boxes, fdet, pickle.HIGHEST_PROTOCOL)

    print('Write predictions to disk')  # cls by cls, .txt
    write_voc_results_file(all_boxes, dataset, output_dir)
    print('Evaluating detections')
    aps, mAP = do_python_eval(dataset, output_dir, use_07_eval, iou_thres)
    return aps, mAP


def all_args():
    '''Args that user could set'''
    # args
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Evaluation')
    parser.add_argument('--dataset',
                        default='nwpuvhr',
                        type=str,
                        help='Dataset, [nwpuvhr, dota]')
    parser.add_argument('--set_type',
                        default='test',
                        type=str,
                        help='set type, {train, val, trainval, test}')
    parser.add_argument('--img_size',
                        default=416,
                        type=int,
                        help='Size of image input to network')
    parser.add_argument('--arc',
                        default='yolov3.pytorch',
                        type=str,
                        help='Net arcitecture')
    parser.add_argument('--model_config_path',
                        default='config/yolov3_nwpuvhr.cfg',
                        type=str,
                        help='Path to model config')
    parser.add_argument('--trained_model',
                        default=90,
                        type=int,
                        help='Epochs for trained model')
    parser.add_argument('--epochs',
                        default=500,
                        type=int,
                        help='Training epochs')
    parser.add_argument('--save_folder',
                        default='eval/',
                        type=str,
                        help='File path to save results')
    # score is conf_prob*cls_prob
    parser.add_argument('--score_thres',
                        default=0.3,
                        type=float,
                        help='Keep predictions with score >= score_thres')
    parser.add_argument('--nms_thres',
                        default=0.4,
                        type=float,
                        help='IOU thres in NMS')
    parser.add_argument('--use_cuda',
                        default=True,
                        type=str2bool,
                        help='Use cuda to train model')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and args.use_cuda
    args.iou_thres = (0.75, 0.5, 0.1)  # AP75, AP50, AP10
    return args


def get_data(args):
    '''Get data'''
    # Data path prefix
    args.dataroot = os.path.join('data', args.dataset)
    # Data path suffix
    if args.dataset == 'nwpuvhr':
        middle = ''
        DataSet = NwpuvhrDataset
    elif args.dataset == 'dota':
        middle = '_crop'
        DataSet = DotaDataset
    suffix = '{:s}{}.txt'.format(args.set_type, middle)
    # Data path
    if args.dataset == 'dota':
        test_path = '/home/cheney/data/dota/val_crop.lmdb'
    elif args.dataset == 'nwpuvhr':
        test_path = os.path.join(args.dataroot, suffix)
    # Dataset
    dataset = DataSet(test_path, img_size=args.img_size)
    return dataset


def init_viz(args, init_style, para_part, reg, alpha, dataset):
    '''Initialize visdom'''
    viz = visdom.Visdom()
    vis_title = 'YOLOv3.PyTorch_{}_{}_{}_{}_{}'.format(args.dataset,
                                                       init_style, para_part,
                                                       reg, alpha)
    vis_legend = dataset.classes + ['mAP']
    iter_aps = create_vis_plot(viz, 'Epoch', '{} APs'.format(dataset.set_type),
                               vis_title, vis_legend)
    return viz, iter_aps


def get_ckpts(args, init_style, para_part, reg, alpha, ablation_type):
    '''All ckpts in ckpt_path in order'''

    def my_order(path_name):
        parts = path_name.split('.')
        epoch = int(parts[1])
        batch_idx = int(parts[2])
        return 5e3 * epoch + batch_idx

    ckpt_prefix = 'checkpoints'
    ckpt_suffix = '_'.join(
        [args.dataset, init_style, para_part, reg,
         str(alpha), ablation_type])
    #  ckpt_suffix = '{}_{}_{}_{}_{}_{}'.format(args.dataset, init_style,
    #  para_part, reg, alpha,
    #  ablation_type)
    ckpt_path = os.path.join(ckpt_prefix, ckpt_suffix)
    # Remove last one first
    last_item = args.dataset + '.last.weights'
    ckpts_nolast = os.listdir(ckpt_path)
    ckpts_nolast.remove(last_item)
    # Sort increase
    ckpts = sorted(ckpts_nolast, key=my_order) + [last_item]
    return ckpt_path, ckpts


def eval_flowchart(init_style, para_part, reg, alpha, ablation_type):
    '''Main body for evaluation'''
    args = all_args()
    # Storage path 'eval/'
    os.makedirs(args.save_folder, exist_ok=True)
    # Dataset
    dataset = get_data(args)
    # Load net
    net = Darknet(args.model_config_path, img_size=args.img_size)
    # Visdom
    viz, epoch_aps = init_viz(args, init_style, para_part, reg, alpha, dataset)
    # Evaluate
    ckpt_path, ckpts = get_ckpts(args, init_style, para_part, reg, alpha,
                                 ablation_type)

    mAP_max = 0
    for ckpt_idx, ckpt in enumerate(ckpts):
        # sample one for hyperparameter adjustment
        if ckpt_idx < 120:
            continue
        # Make output dir
        dir_name = '_'.join([
            ablation_type, args.arc, args.dataset, args.set_type, init_style,
            para_part, reg,
            str(alpha), ckpt,
            str(ckpt_idx)
        ])
        output_dir = get_output_dir(args.save_folder, dir_name)
        # Load weight
        args.weight_path = os.path.join(ckpt_path, ckpt)
        #  assert os.path.isfile(args.weight_path)
        try:
            net.load_weights(args.weight_path)
        except FileNotFoundError as err:
            print(err)
        # Cuda or not
        if args.cuda:
            net = net.cuda()
            cudnn.benchmark = True
        net.eval()
        print('Finished loading model!')
        # Evaluation, use_07_eval False
        aps, mAP = test_net(output_dir,
                            net,
                            args.cuda,
                            dataset,
                            args.score_thres,
                            args.nms_thres,
                            use_07_eval=False,
                            iou_thres=args.iou_thres)
        # If not greater than before, delete
        if mAP_max >= mAP:
            rmtree(output_dir)
        else:
            mAP_max = mAP
        # Visdom
        update_vis(viz, epoch_aps, ckpt_idx + 1, *aps, mAP)


#  with concurrent.futures.ProcessPoolExecutor() as executor:
#  executor.map(eval_loop, para_parts)
if __name__ == '__main__':
    PARA_PARTS = 'all'  # 'all' is the best
    INIT_STYLES = {
        'random': None,
        'imagenet': 'darknet53.conv.74',
        'coco': 'yolov3.weights'
    }
    #  PARA_PARTS = ('head', 'body', 'bb1', 'bb2', 'all')  # 'all' is the best
    REG_STYLES = ('None', 'WD', 'diagonal', 'keynode')
    ALPHA = (1e2, 1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5)  # hyper-para
    for init_style, pretrained_w in INIT_STYLES.items():
        if init_style == 'random':
            continue
            #  regs = REG_STYLES[:2]
            #  alphas = ALPHA[-5:]
        elif init_style in ('imagenet', 'coco'):
            if init_style == 'coco':
                continue
            regs = REG_STYLES
        else:
            print('Initialization style is not known!')
            exit()
        for reg in regs:
            if reg == 'None':
                #  continue
                alphas = (0, )
            elif reg == 'WD':
                continue
                #  alphas = (1e-1, )
            elif reg in ('diagonal', 'keynode'):
                #  if reg == 'diagonal':
                continue
                alphas = (ALPHA[4], )
            for alpha in alphas:
                #  for ablation_type in [
                #  'ablation_yolov3', 'ablation_smthl1',
                #  'ablation_balance', 'ablation_4anchors'
                #  ]:
                ablation_type = 'ablation_4anchors'
                eval_flowchart(init_style, PARA_PARTS, reg, alpha,
                               ablation_type)
