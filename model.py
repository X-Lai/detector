import torch
import torch.nn.functional as F
from utils import *
from torch import nn
from torchvision.models.resnet import *


class Backbone(nn.Module):
    def __init__(self, type='ResNet50', pretrained=True, progress=True):
        super(Backbone, self).__init__()
        self.model = None
        if type == 'ResNet50':
            self.model = resnet50(pretrained=pretrained, progress=progress)
        else:
            raise Exception("No type {} backbone".format(type))

    def forward(self, x):
        '''
        use backbone to extract features from images x, like ResNet-50
        :param x: Tensor, [N, C_in, H_in, W_in]
        :return: tuple[Tensor], each Tensor has the shape as [N, C_out, H_out, W_out],
        meaning different levels of feature maps.
        '''
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        conv2 = self.model.layer1(x)
        conv3 = self.model.layer2(conv2)
        conv4 = self.model.layer3(conv3)
        conv5 = self.model.layer4(conv4)

        return conv2, conv3, conv4, conv5


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(in_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class Neck(nn.Module):
    def __init__(self, num_outs=5, num_inputs=4, channels_in=[256, 512, 1024, 2048],
                 channels_out=256, type='FPN'):
        super(Neck, self).__init__()
        if type not in ["FPN"]:
            raise Exception("No type {} neck".format(type))
        self.num_inputs = num_inputs
        self.num_outs = num_outs
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.laterals = [ConvModule(in_channels=channels_in[i], out_channels=channels_out,
                                    kernel_size=1) for i in range(num_inputs)]
        self.convs = [ConvModule(in_channels=channels_out, out_channels=channels_out,
                                 kernel_size=3, padding=1) for _ in range(num_inputs)]
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x):
        '''
        use features to build feature pyramids, like FPN
        :param x: tuple[Tensor], means different levels of feature maps
        :return: tuple[Tensor], means different levels of feature maps
        '''
        laterals_outs = [lateral(x[i]) for i, lateral in enumerate(self.laterals)]
        for i in range(self.num_inputs-1, 0, -1):
            laterals_outs[i-1] = laterals_outs[i-1] + \
                                 F.interpolate(laterals_outs[i], scale_factor=2)
        outs = [conv(laterals_outs[i]) for i, conv in enumerate(self.convs)]
        for _ in range(self.num_outs - self.num_inputs):
            # ??? Why kernel_size = 1 ???
            outs.append(F.max_pool2d(outs[-1], kernel_size=1, stride=2))
        return outs


class RPN(nn.Module):
    def __init__(self, channels_in=256, channels_out=256, strides=[4, 8, 16, 32, 64],
                 anchor_ratios=[0.5, 1., 2.], anchor_scales=[8],
                 out_activation='sigmoid'):
        super(RPN, self).__init__()
        self.num_anchors = len(anchor_ratios) * len(anchor_scales)
        self.num_levels = len(strides)
        self.strides = strides
        self.anchor_generators = [AnchorGenerator(strides[i], anchor_ratios, anchor_scales)
                                  for i in range(self.num_levels)]
        self.rpn_conv = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1,
                                  padding=1)
        self.cls = None
        self.loss_fn = None
        if out_activation == 'sigmoid':
            self.cls = nn.Conv2d(channels_out, self.num_anchors, kernel_size=1, stride=1,
                                 padding=0)
            self.loss_fn = torch.nn.BCELoss()
        elif out_activation == 'softmax':
            self.cls = nn.Conv2d(channels_out, self.num_anchors * 2, kernel_size=1,
                                 stride=1, padding=0)
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            raise Exception("No type {} in classification activation".format(out_activation))
        self.reg = nn.Conv2d(channels_out, self.num_anchors * 4, kernel_size=1, stride=1,
                             padding=0)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.rpn_conv.weight.data, std=0.01)
        nn.init.normal_(self.cls.weight.data, std=0.01)
        nn.init.normal_(self.reg.weight.data, std=0.01)

    def forward(self, x):
        '''
        use feature pyramids to predict class score and bbox coordinates
        :param x: tuple[Tensor], means different levels of feature maps
        :return: y1, y2. y1 is a list of Tensor of shape [N, num_anchors*1, H, W]
        when the activation function is sigmoid, or [N, num_anchors*2, H, W]
        when the activation function is softmax, meaning the class score for different
        levels of feature maps. y2 is a list of Tensor of shape [N, num_anchors*4, H, W],
        meaning the prediction for bbox for different levels of feature maps.
        '''
        outs_cls, outs_reg = [], []
        for feat in x:
            feat = self.rpn_conv(feat)
            outs_cls.append(self.cls(feat))
            outs_reg.append(self.reg(feat))
        return outs_cls, outs_reg


    def loss(self, rpn_cls_scores, rpn_loc_preds, gt_clses, gt_bboxes, imgs_meta):
        '''
        compute loss from RPN
        :param rpn_cls_scores: list[Tensor]. A list of Tensor of shape [N, num_anchors*1, H, W]
        when the activation function is sigmoid, or [N, num_anchors*2, H, W] when the
        activation function is softmax, meaning the class score for different levels of
        feature maps.
        :param rpn_loc_preds: list[Tensor]. A list of Tensor of shape [N, num_anchors*4, H, W],
        meaning the prediction for bbox for different levels of feature maps.
        :param gt_clses: list[Tensor], each item means ground-truth labels for all the
        bboxes in each image, having the shape as [num_bboxes, ].
        :param gt_bboxes: list[Tensor], each item means ground-truth bboxes coordinates
        (the top-left corner (x_0, y_0), and the bottom-right corner (x_1, y_1)) for all
        the bboxes in each image, having the shape as [num_bboxes, 4]
        :return: rpn_loss: Scalar.
        '''

        rpn_loss = 0

        mlv_sizes = [rpn_cls_scores[i].size()[-2:] for i in range(self.num_levels)]
        img_shape = imgs_meta[0]['img_shape']
        mlv_anchors = [self.anchor_generators[i].grid_anchors(mlv_sizes[i], self.strides[i], img_shape) for i in range(self.num_levels)]

        # merge anchors of different levels into a single Tensor
        mlv_valid_masks = torch.cat([anchors[1] for anchors in mlv_anchors])
        mlv_anchors = torch.cat([anchors[0] for anchors in mlv_anchors])

        labels_lists, targets_lists = self.rpn_assign_and_sample(mlv_sizes, mlv_anchors,
                                                                 mlv_valid_masks, gt_clses, gt_bboxes)
        for i, labels in enumerate(labels_lists):
            targets = targets_lists[i]
            rpn_loss += self.loss_single(rpn_cls_scores[i], rpn_loc_preds[i],
                                         labels, targets)
        return rpn_loss

    def rpn_assign_and_sample_single(self, mlv_sizes, mlv_anchors, mlv_valid_masks, gt_cls, gt_bbox):
        '''
        assign and sample anchors for a single image
        :return: labels: list[Tensor], each of Tensor has shape [1, num_anchors, H, W]
        targets: list[Tensor], each of Tensor has shape [1, num_anchors*4, H, W]
        '''

        labels, targets = assign(mlv_anchors, mlv_valid_masks, gt_cls, gt_bbox, self.cfg)

        sample_labels = sample(labels, self.cfg)
        sample_targets = targets * (sample_labels == 1).to(dtype=torch.float)

        labels = unmap(sample_labels, self.num_anchors, mlv_sizes)
        targets = unmap(sample_targets, self.num_anchors*4, mlv_sizes)

        return labels, targets

    def rpn_assign_and_sample(self, mlv_sizes, mlv_anchors, mlv_valid_masks, gt_clses, gt_bboxes):
        '''
        Get ground-truth labels and targets
        :param mlv_sizes: list[tuple], means feature map size for different levels
        :param mlv_anchors: Tensor, means [num_all_anchors, 4] for different levels
        :param mlv_valid_masks: Tensor, means [num_all_anchors, ] for different levels
        :param gt_clses: list[Tensor], [num_bboxes, ], ground-truth class for all bboxes for all levels.
        :param gt_bboxes: list[Tenosr], [num_bboxes, 4], ground-truth bboxes for all bboxes for all levels.
        :return: labels: list[Tensor]. Ground-truth class labels.
        targets: list[Tensor]. Corresponding ground-truth bboxes.
        '''

        # split the inputs into each image, and get the final targets and labels for all images
        num_imgs = len(gt_clses)
        labels_lists, targets_lists = [], []
        for i in range(num_imgs):
            labels, targets = self.rpn_assign_and_sample_single(mlv_sizes, mlv_anchors, mlv_valid_masks,
                                                           gt_clses[i], gt_bboxes[i])
            labels_lists.append(labels)
            targets_lists.append(targets)
        labels, targets = [], []
        for i in range(self.num_levels):
            labels.append(torch.cat([labels_lists[j][i] for j in range(num_imgs)]))
            targets.append(torch.cat([targets_lists[j][i] for j in range(num_imgs)]))
        return labels, targets

    def loss_single(self, rpn_cls_score, rpn_loc_pred, labels, targets):
        '''
        compute rpn loss from a single level
        :param rpn_cls_score: Tensor, has shape as [N, num_anchors, H, W] if sigmoid,
        [N, num_anchors*2, H, W] if softmax, represents predicted class scores in a single level.
        :param rpn_loc_pred: Tensor, has shape as [N, num_anchors*4, H, W]
        :param labels: Tensor, has the same shape as rpn_cls_score
        :param targets: Tensor, the same shape as rpn_loc_pred
        :return: loss_rpn
        '''

        # rpn_loss = 0
        # mlv_anchors = [self.anchor_generator.grid_anchors(rpn_cls_scores[i].size()[-2:])
        #                for i in range(len(rpn_cls_scores))]
        #
        # # merge anchors of different levels into a single Tensor
        # mlv_valid_masks = torch.cat([anchors[1].to(dtype=torch.uint8) for anchors in mlv_anchors])
        # mlv_anchors = torch.cat([anchors[0] for anchors in mlv_anchors])
        #
        # pos_inds, gt_bbox_inds, neg_inds = rpn_assign_and_sample(mlv_anchors, mlv_valid_masks,
        #                                                gt_cls, gt_bbox)
        # gt_targets = bbox2loc(gt_bbox)
        #
        # # cls loss
        # loss_cls =
        #
        # # loc loss
        # gt_pos_targets = [gt_targets[gt_bbox_inds[i].to(dtype=torch.uint8)]
        #                   for i in range(self.num_levels)]
        # pos_targets = [rpn_loc_preds[i].permute(1,2,0).view(-1, 4)[pos_inds[i]]
        #                for i in range(self.num_levels)]
        # loss_loc =


    def get_proposals(self, rpn_cls_scores, rpn_bboxes):
        '''
        use RPN prediction results to form proposals
        :param rpn_cls_scores: the same as self.loss()
        :param rpn_bboxes: the same as self.loss()
        :return: proposals: list[Tensor]. Each Tensor means all the bboxes for each image.
        '''

class Head(nn.Module):
    def __init__(self):

    def forward(self, x):
        '''
        use RoI features to classify and regress
        :param x: Tensor. [num_samples, C, H, W], RoI features.
        :return: y1, y2. y1 is a Tensor of shape [num_samples, C], meaning the scores of all
        classes for all samples. y2 is a Tensor of shape [num_samples, 4], meaning the
        prediction targets for all samples.
        '''

    def loss(self, head_cls_scores, head_loc_preds, sample_gt_clses, sample_gt_bboxes):
        '''
        Given the predicted class scores and localizations, compute the loss from Head.
        :param head_cls_scores: Tensor, [num_samples, ]
        :param head_loc_preds: Tensor, [num_samples, 4]
        :param sample_gt_clses: Tensor, [num_samples, ]
        :param sample_gt_bboxes: Tensor, [num_samples, 4]
        :return: loss_head: Scalar
        '''


class FasterRCNN(nn.Module):
    def __init__(self):
        super(FasterRCNN, self).__init__()
        self.backbone = Backbone()
        self.neck = Neck()
        self.rpn = RPN()
        self.cfg = {}


    def forward(self, x, gt_clses, gt_bboxes, imgs_meta):
        '''
        use images x and ground-truth information to obtain loss
        :param x: Tensor, [N, 3, H_in, W_in], input images
        :param gt_clses: list[Tensor], each item means ground-truth labels for all the
        bboxes in each image, having the shape as [num_bboxes, ].
        :param gt_bboxes: list[Tensor], each item means ground-truth bboxes coordinates
        (the top-left corner (x_0, y_0), and the bottom-right corner (x_1, y_1)) for all
        the bboxes in each image, having the shape as [num_bboxes, 4]
        :param imgs_meta: list[dict], each item is a python dictionary including the
        information of each image.
        :return: loss: Scalar. It includes two parts: 1) loss from RPN, 2) loss from head.
        '''

        x = self.backbone(x)
        feats = self.neck(x)

        rpn_cls_scores, rpn_loc_preds = self.rpn(feats)

        loss_rpn = self.rpn.loss(rpn_cls_scores, rpn_loc_preds, gt_clses, gt_bboxes, imgs_meta)

        proposals = self.rpn.get_proposals(rpn_cls_scores, rpn_loc_preds)
        sample_bboxes, sample_gt_bboxes, sample_gt_clses = assign_and_sample(
            proposals, gt_bboxes, gt_clses, self.cfg)

        rois = feat_extract(feats, sample_bboxes, self.cfg)
        head_cls_scores, head_loc_preds = self.head(rois)
        loss_head = self.head.loss(head_cls_scores, head_loc_preds, sample_gt_clses,
                                   sample_gt_bboxes)

        loss = self.cfg.lambda_rpn * loss_rpn + self.cfg.lambda_head * loss_head
        return loss
