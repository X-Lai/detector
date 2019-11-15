import torch
import math

def assign_and_sample(proposals, gt_bboxes, gt_clses, cfg):
    '''
    Assign class index for the given proposal candidates, and then sample them for Head
    to process.
    :param proposals: list[Tensor]. Each Tensor is of shape [num_proposals, 5], and means
    the proposed bboxes' coordinates and class scores for each image. For each item, the
    first 4 values are (x_0, y_0, x_1, y_1), and the last value is the class score within
    [0,1].
    :param gt_bboxes: list[Tensor]. Each Tensor is of the shape [num_bboxes, 4], and means
    the coordinates of all ground-truth bboxes for each image.
    :param gt_clses: list[Tensor], each item means ground-truth labels for all the
    bboxes in each image, having the shape as [num_bboxes, ].
    :return: sample_bboxes: list[Tensor], each Tensor is of the shape [num_samples, 4],
    and represents the sampled bboxes' coordinates for each image.
    sample_gt_bboxes: list[Tensor], each Tensor is of the shape [num_samples, 4], and
    represents the coordinates of the corresponding ground-truth bboxes for each image.
    sample_gt_clses: list[Tensor], each Tensor is of the shape [num_samples, ], and
    represents the class index of the corresponding ground-truth bboxes for each image.
    '''


def feat_extract(feats, sample_bboxes, cfg):
    '''
    Map each sample bbox to a level of feature map, and then extract the features of
    a fixed size.
    :param feats: list[Tensor], means different levels of feature maps, each Tensor is of
    the shape [N, C, H, W] for each level.
    :param sample_bboxes: list[Tensor], each Tensor is of the shape [num_samples, 4],
    and represents the sampled bboxes' coordinates for each image.
    :return: rois: Tensor, [num_samples, C, H, W]
    '''

def assign(mlv_anchors, mlv_valid_masks, gt_cls, gt_bbox, cfg):
    '''
    1. find those positive(set 1) anchors with overlaps with any gt bbox > pos_thresh
    2. find those positive(set 1) anchors which has the most overlaps with each gt bbox
    3. find those negative(set -1) anchors with overlaps with any gt bbox < neg_thresh
    4. the rest anchors are ignored(set 0), which are not considered in the loss function.
    :param mlv_anchors: Tensor, [num_all_anchors, 4]
    :param mlv_valid_masks: Tensor, [num_all_anchors, ]
    :param gt_cls: Tensor, [num_bboxes, ]
    :param gt_bbox: Tensor, [num_bboxes, 4]
    :param cfg:
    :return: labels: Tensor, [h*w*num_anchors, ], indices of gt_bboxes.
        Positive samples: 1; Negative samples: -1; Ignored samples: 0
    targets: Tensor, [h*w*num_anchors, 4]
    '''





def sample(labels, cfg):
    '''
    Get both positive and negative samples
    :param labels: Tensor, [h*w*num_anchors, ]
    :param cfg:
    :return: labels. Positive samples: 1; Negative samples: -1; Ignored samples: 0
    '''


def unmap(sample_labels, num_anchors, mlv_sizes):



class AnchorGenerator(object):
    def __init__(self, base_size, anchor_ratios, anchor_scales):
        self.base_size = base_size
        self.num_anchors = len(anchor_scales) * len(anchor_ratios)
        self.anchor_ratios = torch.tensor(anchor_ratios)
        self.anchor_scales = torch.tensor(anchor_scales)
        self.anchor_bases = self.get_anchor_bases()

    def get_anchor_bases(self):
        '''
        use self.anchor_ratios and self.anchor_scales to generate anchor bases, i.e. the
        anchors coordinates in one grid.
        :return: anchor_bases: Tensor. [num_anchors, 4], i.e. [[x_0,x_1,y_1,y2], ...]
        '''
        ws = torch.sqrt(self.anchor_ratios)[None, :] * self.anchor_scales[:, None] * self.base_size
        hs = 1.0 / torch.sqrt(self.anchor_ratios)[None, :] * self.anchor_scales[:, None] * self.base_size
        center = 0.5 * (self.base_size - 1)
        anchor_bases = torch.stack([
            center - 0.5 * (ws - 1), center - 0.5 * (hs - 1),
            center + 0.5 * (ws - 1), center + 0.5 * (hs - 1)
        ], dim=-1)
        return anchor_bases

    def grid_anchors(self, feat_size, stride, img_shape, device='cuda'):
        '''
        obtain anchors in the feature maps with the size of feat_size
        :param feat_size: tuple. (h, w)
        :return: anchors: Tensor. [h*w*num_anchors, 4].
        valid_mask: Tensor. [h*w*num_anchors]
        '''
        h, w = feat_size
        img_x, img_y = img_shape

        base_anchors = self.anchor_bases.to(device)
        shift_x = torch.arange(w) * stride
        shift_y = torch.arange(h) * stride
        shift_xx = shift_x.repeat(h, 1).view(-1)
        shift_yy = shift_y.view(-1, 1).repeat(1, w).view(-1)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        anchors = (base_anchors[None, :, :] + shifts[:, None]).view(-1, 4)

        valid_mask = torch.ones(h*w*self.num_anchors, dtype=torch.uint8, device=device)
        valid_mask[anchors[:, 0]<0] = 0
        valid_mask[anchors[:, 1]<0] = 0
        valid_mask[anchors[:, 2]>img_x] = 0
        valid_mask[anchors[:, 3]>img_y] = 0

        return anchors, valid_mask

