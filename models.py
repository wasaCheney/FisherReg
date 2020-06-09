from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from utils.parse_config import parse_model_config
from utils.utils import build_targets


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module
    configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module(
                'conv_%d' % i,
                nn.Conv2d(in_channels=output_filters[-1],
                          out_channels=filters,
                          kernel_size=kernel_size,
                          stride=int(module_def['stride']),
                          padding=pad,
                          bias=not bn))
            if bn:
                modules.add_module('batch_norm_%d' % i,
                                   nn.BatchNorm2d(filters))
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

        elif module_def['type'] == 'upsample':
            upsample = nn.Upsample(scale_factor=int(module_def['stride']),
                                   mode='nearest')
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def["layers"].split(',')]
            # How many filters (feature maps) totally
            filters = sum([output_filters[layer_i] for layer_i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())

        # Global Average pooling
        elif module_def['type'] == 'avgpool':
            global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
            modules.add_module("avgpool_%d" % i, global_avgpool)

        # softmax output
        elif module_def['type'] == 'softmax':
            num_classes = int(module_def['classes'])
            softmax_layer = SoftmaxLayer(num_classes)
            modules.add_module('softmax_%d' % i, softmax_layer)

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            all_anchors = [(anchors[i], anchors[i + 1])
                           for i in range(0, len(anchors), 2)]
            anchors = [all_anchors[i] for i in anchor_idxs]
            num_classes = int(module_def['classes'])
            # width = height for network input 416
            img_height = int(hyperparams['height'])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_height,
                                   all_anchors)
            modules.add_module('yolo_%d' % i, yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class SoftmaxLayer(nn.Module):
    """Classification layer. Input x is the linear output,
    then passed into LogSoftmax and finally NLLLoss."""

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.nll_loss = nn.NLLLoss()  # reduce=True as default

    def forward(self, x, targets=None):
        # x, (bs, num_classes, 1, 1)
        assert targets is not None, 'Labels are necessary to calculate loss!'
        bs = x.size(0)
        # Tensors for cuda support
        if x.is_cuda:
            #  FloatTensor = torch.cuda.FloatTensor
            LongTensor = torch.cuda.LongTensor
        else:
            #  FloatTensor = torch.FloatTensor
            LongTensor = torch.LongTensor
        prediction = x.view(bs, self.num_classes).contiguous()
        log_softmax = self.log_softmax(prediction)
        # Variable?
        targets = LongTensor(targets.view(bs))
        loss = self.nll_loss(log_softmax, targets)
        return log_softmax, loss, bs


class YOLOLayer(nn.Module):
    """Detection layer
    Note: loss is modified by Cheney
        loc_loss and cls_loss are averaged,
        conf_loss is averaged by used bboxes (pos, neg),
        then the final loss is averaged by batch_size
    """

    def __init__(self,
                 anchors,
                 num_classes,
                 img_dim,
                 all_anchors,
                 ignore_thres=0.5):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_dim = img_dim
        self.all_anchors = all_anchors
        # negatives with IOUs > ignore_thres do not contribute to loss
        self.ignore_thres = ignore_thres
        # Hyperparam, it should adjust according to GT's size, w*h
        # It helps net treating boxes with different size equally
        self.alpha = 5

        self.mse_loss = nn.MSELoss(reduce=False)
        self.smthl1_loss = nn.SmoothL1Loss(reduce=False)
        self.bce_loss = nn.BCELoss(reduce=False)

    def forward(self, x, targets=None):
        # x is the output with linear activation before yolo layer
        # x.size() (bs, num_anchors*(5+num_classes), g_dim, g_dim)
        bs = x.size(0)
        g_dim = x.size(2)
        stride = self.img_dim / g_dim
        # Tensors for cuda support
        if x.is_cuda:
            FloatTensor = torch.cuda.FloatTensor
            LongTensor = torch.cuda.LongTensor
        else:
            FloatTensor = torch.FloatTensor
            LongTensor = torch.LongTensor

        prediction = x.view(bs, self.num_anchors, self.bbox_attrs, g_dim,
                            g_dim).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs (offset)
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # Calculate offsets for each grid
        grid = torch.linspace(0, g_dim - 1, g_dim).repeat(g_dim, 1)
        grid_x = grid.repeat(bs * self.num_anchors, 1,
                             1).view(x.shape).type(FloatTensor)
        grid_y = grid.t().repeat(bs * self.num_anchors, 1,
                                 1).view(y.shape).type(FloatTensor)
        scaled_anchors = [(a_w / stride, a_h / stride)
                          for a_w, a_h in self.anchors]
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1,
                                                 g_dim * g_dim).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1,
                                                 g_dim * g_dim).view(h.shape)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # Training
        if targets is not None:

            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()

            scaled_all_anchors = [(a_w / stride, a_h / stride)
                                  for a_w, a_h in self.all_anchors]
            (nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf,
             tcls) = build_targets(pred_boxes.cpu().data,
                                   targets.cpu().data, scaled_anchors,
                                   scaled_all_anchors, self.num_anchors,
                                   self.num_classes, g_dim, self.ignore_thres,
                                   self.img_dim)

            #  nProposals = int((conf > 0.25).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1

            # Handle masks
            mask = Variable(mask.type(FloatTensor))  # loc
            cls_mask = Variable(
                mask.unsqueeze(-1).repeat(
                    1, 1, 1, 1, self.num_classes).type(FloatTensor))  # cls
            conf_mask = Variable(conf_mask.type(FloatTensor))  # neg conf

            # number of positives is less than that of negatives
            # so the loss need to be balanced
            # For loc_loss, cls_loss, should be 1/num_pos
            # For conf_loss, should be 1/(num_pos + num_neg)
            # Ignored boxes does not trigger any loss
            balanced = False
            num_positive_box = torch.sum(mask.view(bs, -1), -1).view(
                bs, 1, 1, 1) + 1e-16
            num_negative_box = torch.sum(conf_mask.view(bs, -1), -1).view(
                bs, 1, 1, 1) + 1e-16

            # Handle target variables
            # (nB, nA, dim, dim)
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            # (nB, nA, dim, dim)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            # (nB, nA, dim, dim)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            # (nB, nA, dim, dim)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            # (nB, nA, dim, dim)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            # (nB, nA, dim, dim, nC)
            tcls = Variable(tcls.type(FloatTensor), requires_grad=False)
            #  box_loss_scale  = Variable(box_loss_scale.type(FloatTensor),
            #  requires_grad=False)

            # loc loss
            loss_x = torch.sum(
                (1 / num_positive_box * self.mse_loss(x, tx))[mask == 1]) / bs
            loss_y = torch.sum(
                (1 / num_positive_box * self.mse_loss(y, ty))[mask == 1]) / bs
            # width height loss, mse (vanilla yolov3) or smthl1 (ours)
            loss_w = torch.sum((
                1 / num_positive_box *  # box_loss_scale *
                self.mse_loss(w, tw))[mask == 1]) / bs
            loss_h = torch.sum((
                1 / num_positive_box *  # box_loss_scale *
                self.mse_loss(h, th))[mask == 1]) / bs
            loss_x *= 1  # self.alpha
            loss_y *= 1  # self.alpha
            loss_w *= 1  # self.alpha
            loss_h *= 1  # self.alpha

            # cls loss
            num_cls_each_box = torch.zeros(bs, self.num_anchors, g_dim,
                                           g_dim).type(FloatTensor) + 1e-16
            # bs, nBoxes, nC
            if balanced:
                num_ref = torch.sum(tcls.reshape(bs, -1, self.num_classes), 1)
                for bs_ind in range(bs):
                    for cls_ind in range(self.num_classes):
                        boxes_ = (tcls[bs_ind][..., cls_ind] == 1)
                        num_cls_each_box[bs_ind][boxes_] = num_ref[bs_ind,
                                                                   cls_ind]
                num_cls_each_box = num_cls_each_box.unsqueeze(-1)
                loss_cls = torch.sum((1 / num_cls_each_box * self.bce_loss(
                    pred_cls, tcls))[cls_mask == 1]) / (bs * self.num_classes)
            else:
                if cls_mask.max().item() == 0.:
                    loss_cls = torch.sum(
                        self.bce_loss(pred_cls, tcls)[cls_mask == 1])
                else:
                    loss_cls = torch.mean(
                        self.bce_loss(pred_cls, tcls)[cls_mask == 1])
            # conf loss
            if balanced:
                conf_balance = [num_positive_box, num_negative_box]
            else:
                conf_balance = [
                    num_positive_box + num_negative_box,
                    num_positive_box + num_negative_box
                ]
            loss_conf_all = self.bce_loss(conf, tconf)
            loss_conf_pos = torch.sum(
                (1 / conf_balance[0] * loss_conf_all)[mask == 1])
            loss_conf_neg = torch.sum(
                (1 / conf_balance[1] * loss_conf_all)[conf_mask == 1])
            loss_conf = (loss_conf_pos + loss_conf_neg) / bs

            loss = loss_x + loss_y + loss_w + loss_h + loss_cls + loss_conf

            return (loss, loss_x.item(), loss_y.item(), loss_w.item(),
                    loss_h.item(), loss_conf.item(), loss_cls.item(), recall)

        else:
            # If not in training phase return predictions
            output = torch.cat(
                (pred_boxes.view(bs, -1, 4) * stride, conf.view(
                    bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
            return output.data


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416, weights_FIMs=None, alpha=1.):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ['x', 'y', 'w', 'h', 'conf', 'cls', 'recall']
        self.weights_FIMs = weights_FIMs  # For Regularization
        self.alpha = alpha

    def forward(self, x, targets=None):
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []
        for i, (module_def,
                module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample', 'avgpool']:
                x = module(x)
            elif module_def['type'] == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo':
                # Train phase: get loss
                if is_training:
                    # module is a Sequential, but only contains YOLOLayer
                    x, *losses = module[0](x, targets)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module(x)
                output.append(x)
            layer_outputs.append(x)
        #  self.losses['recall'] /= 3
        # Bayes TL
        loss_reg = 0
        if self.weights_FIMs:
            loss_reg = self.bayes_tl(loss_reg)
            self.losses['reg'] = loss_reg  # Record reg loss
        return loss_reg + sum(output) if is_training else torch.cat(output, 1)

    def parainds2moduleinds(self):
        """Map para inds to module inds"""
        dataname, style = self.weights_FIMs['heads']
        if dataname == 'imagenet':
            #  range(0,4 5,8 9,11 12,15, 16,18, 19,21 22,24, 25,27 28,30 31,33 34,36
            #  37,40 41,43 44,46 47,49 50,52 53,55)
            if style == 'diagonal':
                module_inds = range(75)
            elif style == 'keynode':
                module_inds = (35, 60, 73)
        elif dataname == 'coco':
            if style == 'diagonal':
                skip_inds = set((80, 81, 82, 92, 93, 94, 104, 105, 106))
                module_inds = set(range(107)).difference(skip_inds)
            elif style == 'keynode':
                module_inds = (79, 91, 103)
        return tuple(module_inds)

    def my_mse(self, x, y, w, alpha=1.):
        """
        Args:
            x, para value
            y, pretrianed value, the same size with x
            w, FIMs, the same size with x
            alpha, hyperparameter"""
        return alpha * 0.5 * torch.sum((x - y)**2 * w)

    def bayes_tl(self, loss_reg=0):
        """BayesTL Regularization"""
        module_inds = self.parainds2moduleinds()
        # Get loss_reg
        num = 0
        for i, (module_def,
                module) in enumerate(zip(self.module_defs, self.module_list)):
            if i in module_inds and module_def['type'] == 'convolutional':
                params = list(module.named_parameters())
                for _, param in params:
                    # print(i, num)  # This for test and remove when testing
                    # print(len(self.weights_FIMs['weights']), len(self.weights_FIMs['FIMs']))
                    loss_reg += self.my_mse(param,
                                            self.weights_FIMs['weights'][num],
                                            self.weights_FIMs['FIMs'][num],
                                            self.alpha)
                    num += 1
        '''
        for loc, ind in module_inds.items():
            info = self.weights_FIMs[loc]
            params = list(self.module_list[ind].named_parameters())
            for num, (_, param) in enumerate(params):
                loss_reg += self.my_mse(param,
                                        info['weights'][num],
                                        info['FIMs'][num],
                                        self.alpha)
        '''
        return loss_reg

    def load_weights(self, weights_path, use_coco=False):
        """Parses and loads the weights stored in 'weights_path'"""
        # Open the weights file
        fp = open(weights_path, "rb")
        header = np.fromfile(fp, dtype=np.int32,
                             count=5)  # First five are header values

        # Needed to write header when saving weights
        self.header_info = header

        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
        len_weights = weights.size
        fp.close()

        ptr = 0
        # These layers' weights would not be loaded
        yolo_head = (80, 81, 92, 93, 104, 105) if use_coco else ()
        for i, (module_def,
                module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                if module_def['batch_normalize']:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        bn_layer.bias)
                    if i not in yolo_head:
                        bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        bn_layer.weight)
                    if i not in yolo_head:
                        bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        bn_layer.running_mean)
                    if i not in yolo_head:
                        bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        bn_layer.running_var)
                    if i not in yolo_head:
                        bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                    if i not in yolo_head:
                        conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(
                    conv_layer.weight)
                if i not in yolo_head:
                    conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
                if ptr >= len_weights:
                    break

    def save_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff
                              (cutoff = -1 -> all are saved)
        """

        fp = open(path, 'wb')
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(
                zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()


class Darknet53(nn.Module):
    """Imagenet classification model"""

    def __init__(self, config_path, img_size=416, weights_FIMs=None, alpha=1.):
        super().__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ['NLL_loss']
        self.weights_FIMs = weights_FIMs
        self.alpha = alpha

    def forward(self, x, targets=None):
        assert targets is not None, 'Labels are necessary!'
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []
        for i, (module_def,
                module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample', 'avgpool']:
                x = module(x)
            elif module_def['type'] == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'softmax':
                # Train phase: get loss
                if is_training:
                    # module is a Sequential, but only contains SoftmaxLayer
                    log_softmax, x, bs = module[0](x, targets)
                    losses = [x]
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss

                output.append(x)
            layer_outputs.append(x)

        return sum(output)

    def load_weights(self, weights_path, use_coco=False):
        """Parses and loads the weights stored in 'weights_path'"""
        # Open the weights file
        fp = open(weights_path, "rb")
        header = np.fromfile(fp, dtype=np.int32,
                             count=5)  # First five are header values

        # Needed to write header when saving weights
        self.header_info = header

        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
        len_weights = weights.size
        fp.close()

        ptr = 0
        # These layers' weights would not be loaded
        yolo_head = (80, 81, 92, 93, 104, 105) if use_coco else ()
        for i, (module_def,
                module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                if module_def['batch_normalize']:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        bn_layer.bias)
                    if i not in yolo_head:
                        bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        bn_layer.weight)
                    if i not in yolo_head:
                        bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        bn_layer.running_mean)
                    if i not in yolo_head:
                        bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        bn_layer.running_var)
                    if i not in yolo_head:
                        bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                    if i not in yolo_head:
                        conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(
                    conv_layer.weight)
                if i not in yolo_head:
                    conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
                if ptr >= len_weights:
                    break

    def save_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff
                              (cutoff = -1 -> all are saved)
        """

        fp = open(path, 'wb')
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(
                zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
