'''YOLOv3'''
import os
import time
import argparse
import pickle

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim

import visdom

from models import Darknet
from utils.utils import weights_init_normal
from utils.utils import create_vis_plot, update_vis
from utils.datasets import ListDataset, NwpuvhrDataset, DotaDataset
from utils.parse_config import parse_data_config  # , parse_model_config


def get_parser():
    '''User interface'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        default='coco',
                        help='names of dataset [coco | nwpuvhr]')
    parser.add_argument('--model_config_path',
                        type=str,
                        default='config/yolov3.cfg',
                        help='path to model config file')
    parser.add_argument('--data_config_path',
                        type=str,
                        default='config/coco.data',
                        help='path to data config file')
    parser.add_argument('--weights_path',
                        type=str,
                        default='weights',
                        help='path to pretrained weights')
    parser.add_argument('--n_cpu',
                        type=int,
                        default=4,
                        help='number of cpu threads to use')
    parser.add_argument('--img_size',
                        type=int,
                        default=416,
                        help='size of each image dimension')
    parser.add_argument('--checkpoint_interval',
                        type=int,
                        default=1,
                        help='interval between saving model weights')
    parser.add_argument('--print_interval',
                        type=int,
                        default=10,
                        help='interval between saving model weights')
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default='checkpoints',
                        help='directory where model checkpoints are saved')
    parser.add_argument('--use_cuda',
                        type=bool,
                        default=True,
                        help='whether to use cuda if available')
    parser.add_argument('--visdom',
                        type=bool,
                        default=True,
                        help='whether to use visdom')
    opt = parser.parse_args()
    opt.cuda = torch.cuda.is_available() and opt.use_cuda
    print(opt)
    return opt


class Paras(object):
    '''Collecting paras'''

    def __init__(self, dataname):
        if dataname == 'nwpuvhr':
            self.data_path = 'data/nwpuvhr/trainval.txt'
            self.batch_size = 12
            self.epochs = 500
            self.lr_init = 1e-4
            self.lr_adjust = {450: 1e-5}
            self.decay = 5e-4
        elif dataname == 'dota':
            self.data_path = 'data/dota/train_crop.txt'
            self.batch_size = 12
            self.epochs = 150
            self.lr_init = 1e-4
            self.lr_adjust = {130: 1e-5}
            self.decay = 5e-4


def get_para(opt):
    '''Training paras'''
    parameters = Paras(opt.dataset)
    # Get data path
    data_config = parse_data_config(opt.data_config_path)
    if opt.dataset == 'dota':
        parameters.data_path = '/home/cheney/data/dota/train_crop.lmdb'
    elif opt.dataset == 'nwpuvhr':
        parameters.data_path = data_config['train']
    # Get hyper parameters
    #  hyperparams = parse_model_config(opt.model_config_path)[0]
    #  para.epochs = int(hyperparams['max_epoch'])
    #  para.lr = float(hyperparams['learning_rate'])
    #  para.lr_steps = list(map(int, hyperparams['steps'].strip().split(',')))
    #  lr_scales = map(float, hyperparams['scales'].strip().split(','))
    #  momentum = float(hyperparams['momentum'])
    #  decay = float(hyperparams['decay'])
    return parameters


def get_data(dataset_name, dataset_path, num_imgs, n_cpu):
    '''Get data'''
    if dataset_name == 'coco':
        dataset = ListDataset(dataset_path)
    elif dataset_name == 'nwpuvhr':
        dataset = NwpuvhrDataset(dataset_path)
    elif dataset_name == 'dota':
        dataset = DotaDataset(dataset_path)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=num_imgs,
                                             shuffle=True,
                                             num_workers=n_cpu)
    return dataset, dataloader


def get_model(opt, pretrained=None, trn=True, weights_FIMs=None, alpha=1.):
    '''Getting model and initializing.
    Args:
        pretrained, None or path to pretrained model weights
        trn, True for training and False for evaluating'''
    # Model structure
    model = Darknet(opt.model_config_path, opt.img_size, weights_FIMs, alpha)
    print(model)
    # Initialize
    model.apply(weights_init_normal)
    # Pretrained or not
    coco_weights = True if pretrained == 'weights/yolov3.weights' else False
    try:
        model.load_weights(pretrained, use_coco=coco_weights)
    except TypeError:
        pass
    # Cuda or not
    if opt.cuda:
        model = model.cuda()
        cudnn.benchmark = True
    # Mode = train or eval
    if trn:
        model.train()
    else:
        model.eval()
    return model


def get_optimizer(model, learning_rate, reg, alpha):
    '''YOLOv3 optimizer
    Args:
        start_ind, paras from here to the end will be trained'''
    # len of model.parameters() is 222, 0-155 is pretrained on ImageNet
    # bb1 is body + backbone's heads for 13*13 level
    # bb2 is bb1 + backbone's heads for 26*26 level
    head_inds = ((171, 176), (194, 199), (217, 222))  # (include, exclude)
    part_inds = {
        'head': head_inds,
        'body': 156,
        'bb1': 129,
        'bb2': 78,
        'all': 0
    }
    optims = {}
    for part, ind in part_inds.items():
        if part == 'head':
            trained_para = []
            for start, end in ind:
                temp_seq = list(model.parameters())[start:end]
                #  print(len(temp_seq))
                trained_para += temp_seq  # [ele for ele in temp_seq]
            trained_para = tuple(trained_para)
        else:
            trained_para = (ele for ele in list(model.parameters())[ind:])
        # weight decay
        if reg == 'WD':
            decay = alpha
        else:
            decay = 0
        optimizer = optim.Adam(trained_para,
                               lr=learning_rate,
                               weight_decay=decay)
        optims[part] = optimizer
    return optims


def init_viz(opt, init_style, para_part, reg, alpha):
    '''Initialize visdom'''
    viz = visdom.Visdom()
    vis_title = 'YOLOv3.PyTorch_{}_{}_{}_{}_{}'.format(opt.dataset, init_style,
                                                       para_part, reg, alpha)
    vis_legend = ['XY Loss', 'WH Loss', 'Conf Loss', 'Cls Loss', 'Reg Loss']
    iter_plot = create_vis_plot(viz, 'Iteration', 'Loss', vis_title,
                                vis_legend)
    epoch_plot = create_vis_plot(viz, 'Epoch', 'Loss', vis_title, vis_legend)
    lr_plot = viz.line(X=torch.zeros((1, )).cpu(),
                       Y=torch.zeros((1, )).cpu(),
                       opts=dict(xlabel='Iteration',
                                 ylabel='lr',
                                 title=vis_title))
    return viz, iter_plot, epoch_plot, lr_plot


def load_weights_FIMs(opt, init_style, reg):
    """Load pretrained weights and FIMs"""
    # Load FIMs
    weights_FIMs = None
    if init_style in ('imagenet', 'coco') and reg in ('diagonal', 'keynode'):
        FIM_path = '{}/{}.FIMs.{}'.format(opt.weights_path, init_style, reg)
        with open(FIM_path, 'rb') as fFIMs:
            weights_FIMs = pickle.load(fFIMs)
    return weights_FIMs


def flowchart(init_style, pretrained_w, para_part, reg, alpha=1):
    '''YOLOv3 training'''
    assert para_part in ('head', 'body', 'bb1', 'bb2', 'all')
    opt = get_parser()
    # Init vis
    if opt.visdom:
        viz, iter_plot, epoch_plot, lr_plot = init_viz(opt, init_style,
                                                       para_part, reg, alpha)
    # Checkpoint dir
    save_dir = os.path.join(
        opt.checkpoint_dir, '{}_{}_{}_{}_{}'.format(opt.dataset, init_style,
                                                    para_part, reg, alpha))
    os.makedirs(save_dir, exist_ok=True)

    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor

    paras = get_para(opt)

    dataset, dataloader = get_data(opt.dataset, paras.data_path,
                                   paras.batch_size, opt.n_cpu)
    # Pretrained weights path
    try:
        pretrained_path = os.path.join(opt.weights_path, pretrained_w)
    except TypeError:
        pretrained_path = None
    # Load FIMs
    weights_FIMs = load_weights_FIMs(opt, init_style, reg)
    # Model
    model = get_model(opt, pretrained_path, True, weights_FIMs, alpha)
    #  for para_ind, para in enumerate(model.named_parameters()):
    #  print(para_ind, para[0])
    #  exit()
    # Optimizer
    lr_runtime = paras.lr_init
    optims = get_optimizer(model, lr_runtime, reg, alpha)
    optimizer = optims[para_part]
    for epoch in range(paras.epochs):
        # adjust lr
        if epoch + 1 in paras.lr_adjust:
            lr_runtime = paras.lr_adjust[epoch + 1]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_runtime
        # Init epoch plot
        if opt.visdom:
            xy_epoch = 0
            wh_epoch = 0
            conf_epoch = 0
            cls_epoch = 0
            reg_epoch = 0
        # Batch training
        end_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            #  print('img_path', img_path)
            # Data time
            data_time = time.time() - end_time
            # Batch data
            imgs = Variable(imgs.type(Tensor))
            targets = Variable(targets.type(Tensor), requires_grad=False)
            # Forward
            loss = model(imgs, targets)
            # Backward
            optimizer.zero_grad()
            loss.backward()
            # Params update
            optimizer.step()
            # return
            # Batch time
            batch_time = time.time() - end_time
            end_time = time.time()
            # Loss
            xy_loss = model.losses['x'] + model.losses['y']
            wh_loss = model.losses['w'] + model.losses['h']
            loc_loss = xy_loss + wh_loss
            # Print to screen
            if batch_i % opt.print_interval == 0:
                print('[Epoch {:d}/{:d}, Batch {:d}/{:d}] '
                      '[Time: batch {:.4f} data {:.4f}] '
                      '[Losses: loc {:.4f}, conf {:.4f}, '
                      'cls {:.4f}, reg {:.4f}, '
                      'total: {:.4f}]'.format(epoch, paras.epochs, batch_i,
                                              len(dataloader), batch_time,
                                              data_time, loc_loss,
                                              model.losses['conf'],
                                              model.losses['cls'],
                                              model.losses['reg'],
                                              loss.item()))
            # Update batch viz
            if opt.visdom:
                iteration = batch_i + 1 + epoch * len(dataloader)
                update_vis(viz, lr_plot, iteration, lr_runtime)
                update_vis(viz, iter_plot, iteration, xy_loss, wh_loss,
                           model.losses['conf'], model.losses['cls'],
                           model.losses['reg'])
                xy_epoch += xy_loss * paras.batch_size
                wh_epoch += wh_loss * paras.batch_size
                conf_epoch += model.losses['conf'] * paras.batch_size
                cls_epoch += model.losses['cls'] * paras.batch_size
                reg_epoch += model.losses['reg'] * paras.batch_size
            # Storage
            if opt.dataset == 'dota':
                iterations = batch_i + 1 + epoch * len(dataloader)
                if (epoch + 1 >= 80) and (iterations %
                                          opt.checkpoint_interval) == 0:
                    file_name = '{}.{:n}.{:n}.weights'.format(
                        opt.dataset, epoch, batch_i)
                    model.save_weights(os.path.join(save_dir, file_name))
            # Model head
            model.seen += imgs.size(0)
        # Update epoch viz
        if opt.visdom:
            xy_epoch /= len(dataset)
            wh_epoch /= len(dataset)
            conf_epoch /= len(dataset)
            cls_epoch /= len(dataset)
            reg_epoch /= len(dataset)
            update_vis(viz, epoch_plot, epoch + 1, xy_epoch, wh_epoch,
                       conf_epoch, cls_epoch, reg_epoch)
        # Storage
        if opt.dataset == 'nwpuvhr':
            if (epoch + 1 >=
                    300) and (epoch + 1) % opt.checkpoint_interval == 0:
                file_name = '{}.{:n}.1.weights'.format(opt.dataset, epoch)
                model.save_weights(os.path.join(save_dir, file_name))
    # Storage
    file_name = '{}.last.weights'.format(opt.dataset)
    model.save_weights(os.path.join(save_dir, file_name))


if __name__ == '__main__':
    PARA_PARTS = 'all'  # 'all' is the best
    INIT_STYLES = {
        'random': None,
        'imagenet': 'darknet53.conv.74',
        'coco': 'yolov3.weights'
    }
    #  PARA_PARTS = ('head', 'body', 'bb1', 'bb2', 'all')  # 'all' is the best
    REG_STYLES = ('None', 'WD', 'diagonal', 'keynode')
    ALPHA = (1e2, 1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5)  # hyper-para for WD
    for init_style, pretrained_w in INIT_STYLES.items():
        if init_style == 'random':
            continue
            regs = REG_STYLES[:2]
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
                alphas = ALPHA[-5:]
            elif reg in ('diagonal', 'keynode'):
                #  if reg == 'diagognal':
                continue
                alphas = [1]
            for alpha in alphas:
                flowchart(init_style, pretrained_w, PARA_PARTS, reg, alpha)
