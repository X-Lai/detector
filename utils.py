import torch


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

    :param mlv_anchors:
    :param mlv_valid_masks:
    :param gt_cls:
    :param gt_bbox:
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
    def __init__(self, anchor_ratios, anchor_scales, anchor_strides):
        self.anchor_ratios = anchor_ratios
        self.anchor_scales = anchor_scales
        self.anchor_strides = anchor_strides
        self.anchor_bases = self.get_anchor_bases()

    def get_anchor_bases(self):
        '''
        use self.anchor_ratios and self.anchor_scales to generate anchor bases, i.e. the
        anchors coordinates in one grid.
        :return: anchor_bases: Tensor. [num_anchors, 4], i.e. [[x_0,x_1,y_1,y2], ...]
        '''

    def grid_anchors(self, feat_size):
        '''
        obtain anchors in the feature maps with the size of feat_size
        :param feat_size: tuple. (h, w)
        :return: anchors: Tensor. [h*w*num_anchors, 4].
        valid_mask: Tensor. [h*w*num_anchors]
        '''

