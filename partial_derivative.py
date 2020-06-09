'''Compute and save partial derivative'''
import os
import argparse
import pickle
import copy

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from models import Darknet53, Darknet
from utils.utils import weights_init_normal
from utils.datasets import ListDataset, ImagenetDataset
from utils.parse_config import parse_data_config  # , parse_model_config


def get_parser(dataname):
    '''User interface'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        help='names of dataset [imagenet | coco]')
    parser.add_argument('--model_config_path', type=str,
                        default='config/yolov3.cfg',
                        help='path to model config file')
    parser.add_argument('--data_config_path', type=str,
                        default='config/coco.data',
                        help='path to data config file')
    parser.add_argument('--weights_path', type=str,
                        default='weights',
                        help='path to pretrained weights')
    parser.add_argument('--pretrained_w', type=str,
                        default='yolov3.weights',
                        help='pretrained weights')
    parser.add_argument('--n_cpu', type=int, default=4,
                        help='number of cpu threads to use')
    parser.add_argument('--img_size', type=int, default=224,
                        help='size of each image dimension')
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help='whether to use cuda if available')
    opt = parser.parse_args()
    if dataname == 'imagenet':
        opt.model_config_path = 'config/darknet53.cfg'
        opt.data_config_path = 'config/imagenet1k.data'
        opt.pretrained_w = 'darknet53.weights'
        opt.img_size = 224
    elif dataname == 'coco':
        opt.model_config_path = 'config/yolov3.cfg'
        opt.data_config_path = 'config/coco.data'
        opt.pretrained_w = 'yolov3.weights'
        opt.img_size = 416
    opt.cuda = torch.cuda.is_available() and opt.use_cuda
    cudnn.benchmark = True if opt.cuda else False
    print(opt)
    return opt


def get_data(opt, num_imgs, n_cpu):
    '''Get data'''
    data_cfg = parse_data_config(opt.data_config_path)
    dataset_path = data_cfg['train']
    if opt.dataset == 'coco':
        opt.img_size = 416
        Dataname = ListDataset
    elif opt.dataset == 'imagenet':
        opt.img_size = 224
        Dataname = ImagenetDataset
    dataset = Dataname(dataset_path, img_size=opt.img_size)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=num_imgs, shuffle=True,
        num_workers=n_cpu, pin_memory=True)
    return dataset, dataloader


def get_model(opt, pretrained=None, trn=True):
    '''Getting model and initializing.
    Args:
        pretrained, None or path to pretrained model weights
        trn, True for training and False for evaluating'''
    if opt.dataset == 'imagenet':
        model_name = Darknet53
    elif opt.dataset == 'coco':
        model_name = Darknet
    model = model_name(opt.model_config_path, opt.img_size, None, 0)
    print(model)
    # Initialization
    model.apply(weights_init_normal)
    coco_weights = True if pretrained == 'weights/yolov3.weights' else False
    try:
        model.load_weights(pretrained, use_coco=coco_weights)
    except TypeError:
        pass
    # Cuda or not
    if opt.cuda:
        model = model.cuda()
        cudnn.benchmark = True
    if trn:
        model.train()
    else:
        model.eval()
    return model


def get_weights_FIMs(dataname='coco', style='diagonal'):
    """Get the basic information of FIMs
    Args:
        dataname, [imagenet, coco]
        style, [keynode, diagonal]"""
    conds = ((dataname in ('imagenet', 'coco')) and
             (style in ('diagonal', 'keynode')))
    assert conds, 'Wrong combination: {} and {}'.format(dataname, style)
    if dataname == 'imagenet':
        if style == 'diagonal':
            all_idx = tuple(range(156))
        elif style == 'keynode':
            all_idx = (75, 76, 77, 126, 127, 128, 153, 154, 155)
    elif dataname == 'coco':
        if style == 'diagonal':
            all_idx = (tuple(range(171)) +
                       tuple(range(176, 194)) +
                       tuple(range(199, 217)))
        elif style == 'keynode':
            all_idx = (168, 169, 170, 191, 192, 193, 214, 215, 216)
    weights_FIMs = {
        'heads': (dataname, style),
        'inds': all_idx,
        'weights': [],
        'FIMs': []}
    return weights_FIMs


def flowchart(dataname, style):
    '''Compute the partial derivative'''
    #  assert para_part in ('head', 'body', 'bb1', 'bb2', 'all')
    opt = get_parser(dataname)
    opt.dataset = dataname

    save_dir = '{}/{}.FIMsnew.{}'.format(opt.weights_path, opt.dataset, style)
    batch_size = 1  # must be 1!

    if opt.cuda:
        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor
    else:
        FloatTensor = torch.FloatTensor
        LongTensor = torch.LongTensor

    # Data
    dataset, dataloader = get_data(opt, batch_size, opt.n_cpu)
    # Weights path
    try:
        pretrained_path = os.path.join(opt.weights_path, opt.pretrained_w)
    except TypeError:
        pretrained_path = None
    # Model
    model = get_model(opt, pretrained=pretrained_path, trn=True)
    params = list(model.named_parameters())
    # Paras collector, FIM is too big, only the diagonals are computed
    weights_FIMs = get_weights_FIMs(dataname, style)
    # Collecting weights
    for ind in weights_FIMs['inds']:
        print('===Collecting weights===')
        print(ind, params[ind][0])  # para name
        weights_FIMs['weights'].append(params[ind][1])
        #  print(params[ind][1].is_cuda)  # GPU
    #  for para_ind, para in enumerate(model.named_parameters()):
        #  print(para_ind, para[0])
    #  exit()
    for epoch in range(1):
        img_seen = 0
        for batch_i, (img_path, imgs, targets) in enumerate(dataloader):
            # Batch data
            imgs = Variable(imgs.type(FloatTensor))
            if dataname == 'imagenet':
                targets = Variable(targets.type(LongTensor), requires_grad=False)
            elif dataname == 'coco':
                targets = Variable(targets.type(FloatTensor), requires_grad=False)
            # Forward
            loss = model(imgs, targets)
            # Backward
            model.zero_grad()
            loss.backward()
            # Collecting FIMs
            for num, ind in enumerate(weights_FIMs['inds']):
                # Old FIM
                try:
                    expect_H = copy.deepcopy(weights_FIMs['FIMs'][num])
                except IndexError:
                    expect_H = 0
                # This FIM
                print('{}===Collecting FIMs==='.format(batch_i))
                print(ind, params[ind][0])
                this_grad = params[ind][1].grad
                # FIM has too many elements, so only keep the diagonals!
                # The complete one should be E[\nabla * \nabla^T]
                this_H = this_grad * this_grad
                # New FIM
                # Prevent overflow or underflow, increment-style
                coeff = 1 / (1 + img_seen/batch_size)
                expect_H = expect_H * (1 - coeff) + this_H * coeff
                # Update into the dict
                try:
                    weights_FIMs['FIMs'][num] = copy.deepcopy(expect_H)
                except IndexError:
                    weights_FIMs['FIMs'].append(expect_H)
            img_seen += batch_size
    # Save FIMs
    with open(save_dir, 'wb') as fFIMs:
        pickle.dump(weights_FIMs, fFIMs)


if __name__ == '__main__':
    DATANAME = ('imagenet', 'coco')
    STYLE = ('diagonal', 'keynode')
    for dataname in DATANAME:
        if dataname == 'coco':
            continue
        for style in STYLE:
            if style == 'keynode':
                continue
            flowchart(dataname, style)
    exit()
    # Only for checking
    with open('weights/imagenet.FIMs', 'rb') as fFIMs:
        FIMs = pickle.load(fFIMs)
    print(type(FIMs))
    for info in FIMs.values():
        for ind in range(3):
            print(info['weights'][ind].size(),
                  info['FIMs'][ind].size())
