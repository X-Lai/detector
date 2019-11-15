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

def bbox2loc(bboxes):
    '''

    :param bboxes: Tensor, [h*w*num_anchors, 4], i.e. [x_0, y_0, x_1, y_1]
    :return: targets: Tensor, [h*w*num_anchors, 4], i.e. [x, y, w, h]
    '''
    return torch.stack([(bboxes[:, 2] + bboxes[:, 0]) / 2,
                        (bboxes[:, 3] + bboxes[:, 1]) / 2,
                        bboxes[:, 2] - bboxes[:, 0],
                        bboxes[:, 3] - bboxes[:, 1]])


def assign(mlv_anchors, mlv_valid_masks, gt_cls, gt_bbox, cfg):
    '''
    1. find those positive(set 1) anchors with overlaps with any gt bbox > assign_pos_thresh
    2. find those positive(set 1) anchors which has the most overlaps with each gt bbox
    3. find those negative(set -1) anchors with overlaps with any gt bbox < assign_neg_thresh
    4. the rest anchors are ignored(set 0), which are not considered in the loss function.
    :param mlv_anchors: Tensor, [num_all_anchors, 4]
    :param mlv_valid_masks: Tensor, [num_all_anchors, ]
    :param gt_cls: Tensor, [num_bboxes, ]
    :param gt_bbox: Tensor, [num_bboxes, 4]
    :param cfg:
    :return: labels: Tensor, [h*w*num_anchors, ], indices of gt_bboxes.
        Positive samples: 1; Negative samples: -1; Ignored samples: 0
    targets: Tensor, [h*w*num_anchors, 4],
        t_x = (x-x_a)/w_a, t_y = (y-y_a)/h_a, t_w = log(w/w_a), t_h = log(h/h_a)
    '''

    overlaps = get_overlaps(mlv_anchors, gt_bbox)
    areas_anchors = get_areas(mlv_anchors)
    areas_gt_bboxes = get_areas(gt_bbox)
    overlaps = overlaps / (areas_anchors[None, :] + areas_gt_bboxes[:, None] - overlaps)

    labels = torch.zeros_like(mlv_valid_masks, dtype=torch.int8)

    values, indices = torch.max(overlaps, dim=0)

    # get targets
    gt_bboxes = gt_bbox[indices]
    gt_loc = bbox2loc(gt_bboxes)
    anchors_loc = bbox2loc(mlv_anchors)
    targets = torch.stack([(gt_loc[:, 0]-anchors_loc[:, 0])/anchors_loc[:, 2],
                           (gt_loc[:, 1]-anchors_loc[:, 1])/anchors_loc[:, 3],
                           torch.log(gt_loc[:, 2]/anchors_loc[:, 2]),
                           torch.log(gt_loc[:, 3]/anchors_loc[:, 3])], dim=-1)

    # The first step
    labels[values >= cfg.assign_pos_thresh] = 1

    # The second step
    max_indices_per_gt_bbox = torch.argmax(overlaps, dim=1)
    labels[max_indices_per_gt_bbox] = 1

    # The third step
    labels[values < cfg.assign_neg_thresh] = -1

    return labels, targets


def get_overlaps(anchors, gt_bbox):
    '''
    get areas of overlapping region
    :param anchors:
    :param gt_bbox:
    :return: overlaps: [num_gt_bboxes, num_all_anchors]
    '''
    xs_lt = torch.max(anchors[None, :, 0], gt_bbox[:, None, 0])
    ys_lt = torch.max(anchors[None, :, 1], gt_bbox[:, None, 1])
    xs_rb = torch.min(anchors[None, :, 2], gt_bbox[:, None, 2])
    ys_rb = torch.min(anchors[None, :, 3], gt_bbox[:, None, 3])
    return (ys_rb - ys_lt) * (xs_rb - xs_lt)


def get_areas(bboxes):
    return (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])


def sample(labels, cfg):
    '''
    Randomly select positive and negative samples
    :param labels: Tensor, [h*w*num_anchors, ]
    :param cfg: num_samples, pos_ratio
    :return: labels. Positive samples: 1; Negative samples: -1; Ignored samples: 0
    '''
    num_pos_samples = cfg.num_samples * cfg.pos_ratio
    num_neg_samples = cfg.num_samples - num_pos_samples
    pos_labels = (labels == 1).to(dtype=torch.float16)
    max_num_pos_samples = torch.sum(pos_labels).to(dtype=torch.int)
    if num_pos_samples > max_num_pos_samples:
        num_pos_samples = max_num_pos_samples
    pos_inds = torch.multinomial(pos_labels, num_pos_samples)

    neg_labels = (labels == -1).to(dtype=torch.float16)
    max_num_neg_samples = torch.sum(neg_labels).to(dtype=torch.int)
    if num_neg_samples > max_num_neg_samples:
        num_neg_samples = max_num_neg_samples
    neg_inds = torch.multinomial(neg_labels, num_neg_samples)

    sample_labels = torch.zeros_like(labels)
    sample_labels[pos_inds] = 1
    sample_labels[neg_inds] = -1

    return sample_labels


def unmap(sample_labels, num_anchors, mlv_sizes):
    segments = [h*w*num_anchors for h, w in mlv_sizes]
    assert sum(segments) == sample_labels.size(0)
    labels = []
    start = 0
    for i, seg in enumerate(segments):
        h, w = mlv_sizes[i]
        labels.append(sample_labels[start:start+seg].view(1, h, w, num_anchors).permute(0, 3, 1, 2))
        start += seg
    return labels


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
        valid_mask[anchors[:, 0] < 0] = 0
        valid_mask[anchors[:, 1] < 0] = 0
        valid_mask[anchors[:, 2] > img_x] = 0
        valid_mask[anchors[:, 3] > img_y] = 0

        return anchors, valid_mask
