import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json, os, copy
import collections

from detectron2.structures import Boxes, BoxMode, pairwise_iou
import torch


# https://bitbucket.org/visionsystemsinc/geogenx/src/e07cb38be109a2b0831baff0c47f74e9a8a88206/ \
#              internal/python/geogenx/eval_utils.py
# see also https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py
def match_gt(detections, annotations, iou_thresh=0.0, convert_to_json_ids=True):
    # detections - predicted detections
    # annotations - ground-truth annotations
    #
    # NOTE category labels are ignored in proposal scoring

    gt_boxes = [
            BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            for obj in annotations
        ]
    gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
    gt_boxes = Boxes(gt_boxes)
    gt_areas = torch.as_tensor([obj["area"] for obj in annotations if obj["iscrowd"] == 0])

    proposal_boxes = [
            BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            for obj in detections
    ]
    proposal_boxes = torch.as_tensor(proposal_boxes).reshape(-1,4)
    proposal_boxes = Boxes(proposal_boxes)

    overlaps = pairwise_iou(proposal_boxes, gt_boxes)
    box_inds = np.argsort([-obj["score"] for obj in detections])
    matches = [[] for _ in annotations]
    best_matches_gt = [-1 for _ in annotations]
    best_matches_dt = [-1 for _ in detections]


    for box_ind in box_inds: # in the order of box confidence
        m = -1
        curr_thresh = iou_thresh
        for gt_ind in range(0, len(gt_boxes)):
            if overlaps[box_ind, gt_ind] > 0:
                matches[gt_ind].append(box_ind) # some overlap, add to the list
                #print(gt_ind, box_ind, overlaps[box_ind, gt_ind], curr_thresh, best_matches_gt[gt_ind])
            if best_matches_gt[gt_ind] > -1:
                continue # already matched to some other proposal
            if  overlaps[box_ind, gt_ind] > curr_thresh:
                curr_thresh = overlaps[box_ind, gt_ind]
                m = gt_ind
        if m == -1: # no match with any ground truth for this box
            continue
        best_matches_gt[m] = box_ind # set the best matching box at box_ind for ground truth m
        best_matches_dt[box_ind] = m # set ground truth m for the box at box_ind

    # overlaps - [len(detections), len(annotations)] np.array
    # matches - [detections[j] for j in matches[i]] are the possible confusers
    #           for annotation[i]; can be []
    # len(matches) == len(annotations)
    # best_matches_gt - annotation[i] matches detections[best_matches_gt[i]] if
    #                   best_matches_gt != -1
    #                   NOTE the best match for an annotation can be a detection
    #                   from a different category...
    # len(best_matches_gt) == len(annotations)
    # best_matches_dt - detections[i] matches annotations[best_matches_dt[i]]
    #                   if best_matches_dt != -1
    #                   NOTE the best match for a detection can be an annotation
    #                   from a different category...
    # len(best_matches_dt) == len(detections)
    return overlaps, matches, best_matches_gt, best_matches_dt


def overlap(bbox1, bbox2): # IoU
    min_x1, min_y1, width1, height1 = bbox1
    min_x2, min_y2, width2, height2 = bbox2
    max_x1, max_y1 = min_x1 + width1, min_y1 + height1
    max_x2, max_y2 = min_x2 + width2, min_y2 + height2

    inter_x1 = max(min_x2, min_x1)
    inter_y1 = max(min_y2, min_y1)
    inter_x2 = min(max_x2, max_x1)
    inter_y2 = min(max_y2, max_y1)
    width  = inter_x2-inter_x1
    height = inter_y2-inter_y1
    # no intersection
    if width <= 0.0 or height <= 0.0:
        return 0.0
    inter_area = width*height # area of intersection

    area1 = width1*height1
    area2 = width2*height2
    union_area = area1 + area2 - inter_area # area of union

    iou = inter_area / float(union_area)

    return iou

def overlay_annotations(im, bboxes, color):
    im_height, im_width = im.shape[:2]
    for bbox in bboxes:
        min_x, min_y, width, height = bbox
        min_x, min_y, width, height = int(min_x), int(min_y), int(width), int(height)
        # mark bbox
        im[min_y, min_x:min_x+width, :] = color
        im[min_y:min_y+height, min_x, :] = color
        im[min(min_y+height, im_height-1), min_x:min_x+width, :] = color
        im[min_y:min_y+height, min(min_x+width, im_width-1), :] = color

def dilate_bbox(bbox, scale):
    min_x, min_y, width, height = bbox

    min_x, min_y = min_x - width*(scale-1)/2, min_y - height*(scale-1)/2
    width, height = width*scale, height*scale
    dilated = (min_x, min_y, width, height)
    return dilated

def dilate_square_bbox(bbox, scale=None, size=None):
    assert (scale is None) != (size is None) # xor
    min_x, min_y, width, height = bbox

    length = max(width, height)*scale if scale is not None else size

    # dilate square bbox
    center_x, center_y = min_x + width/2, min_y + height/2
    min_x, min_y = center_x - length/2, center_y - length/2
    dilated = (min_x, min_y, length, length)
    return dilated

def clip(bbox, im_width, im_height):
    min_x, min_y, width, height = bbox

    min_x, min_y = max(0, min_x), max(0, min_y)
    width, height = min(im_width, width), min(im_height, height)
    clipped = (min_x, min_y, width, height)
    return clipped

def to_pixel(bbox):
    bbox = map(np.round, bbox)
    bbox = map(int, bbox)
    return list(bbox)

def chip_detection(im, bbox):
    min_x, min_y, width, height = bbox
    chip = im[min_y:min_y+height, min_x:min_x+width, :]
    return chip

def pad_to_square_detection(im, *args, **kwargs):
    height, width = im.shape[:2]
    length = max(width, height)
    horizontal_padding = int((length - width) / 2)
    vertical_padding = int((length - height) / 2)
    padding = ((vertical_padding,), (horizontal_padding,), (0,))
    im1 = np.pad(im, padding, *args, **kwargs)
    if im1.shape[2] != 3: assert False
    return im1

def dist_squared_to_bbox_center(ref, bboxes):
    min_x, min_y, width, height = ref
    center_x, center_y = min_x + width//2, min_y + height//2

    bboxes = np.array(bboxes)
    offsets = np.vstack([
        (bboxes[:, 0] + bboxes[:, 2]/2) - center_x,
        (bboxes[:, 1] + bboxes[:, 3]/2) - center_y
    ])
    dist_squared = np.sum(offsets**2.0, 0)
    return dist_squared

def add_txt_annotation(ax, im, bbox, annotations, categories, txt_offset, label, color):
    # can't just pass in annotations=[detection] because detection can be [], which becomes [[]]
    if not isinstance(annotations, collections.abc.Sequence):
        annotations = [annotations]
    category_by_id = {cat['id']: cat for cat in categories}

    overlaps = [a for a in annotations if overlap(a['bbox'], bbox) > 0]
    if overlaps == []:
        return txt_offset

    im_height, im_width = im.shape[:2]
    x_txt, txt_height = 1.05*im_width, 0.1*im_height

    # sort these by distance from the center
    # alternatively, compute the IoU for each bbox with the primary annotation,
    # (if avaliable), which would have to be added as a parameter
    dist_squared = dist_squared_to_bbox_center(bbox, [a['bbox'] for a in overlaps]) # bbox is not clipped
    overlaps = [overlaps[i] for i in np.argsort(dist_squared)]

    y_txt = txt_height*(txt_offset+1)
    for i,intersect in enumerate(overlaps):
        trimmed_category = category_by_id[intersect['category_id']]['name'].split(':')[1]
        txt = f"{label} - {trimmed_category}" + (f": {intersect['score']:0.2f}" if 'score' in intersect else '')
        ax.text(x_txt, y_txt+txt_height*i, txt, fontsize=15, color=color)
    return txt_offset+len(overlaps)

def save_chip_from_image(im, detection, detections, annotation, annotations, categories, outfile):
    assert (detection != []) or (annotation != [])

    reference = annotation if annotation != [] else detection
    im_height, im_width = im.shape[:2]
    dilated_bbox = dilate_square_bbox(reference['bbox'], 4.0)
    dilated_bbox = to_pixel(clip(dilated_bbox, im_width, im_height))

    # these other annotations are interesting in the case of false alarms (i.e., a miss-categorization)
    overlay_annotations(im, [a['bbox'] for a in annotations], [0,0,128])  # all annotations
    overlay_annotations(im, [a['bbox'] for a in detections],  [0,128,0])  # all detections
    overlay_annotations(im, [a['bbox'] for a in [annotation] if a != []], [0,0,255])  # matched annotation
    overlay_annotations(im, [a['bbox'] for a in [detection]  if a != []], [0,255,0])  # this detection

    chip = chip_detection(im, dilated_bbox)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    #Image.fromarray(chip).save(outfile)
    #
    #with open(os.path.splitext(outfile)[0] + '.txt', 'w') as fd:
    #    if annotation != []:
    #        fd.write(f"*annotation: {category_by_id[annotation['category_id']]['name']}\n")
    #    if detection != []:
    #        fd.write(f"*detection: {category_by_id[detection['category_id']]['name']} {detection['score']:0.2f}\n")
    #    for intersect in [a for a in annotations if overlap(a['bbox'], dilated_bbox) > 0]:
    #        fd.write(f"annotation: {category_by_id[intersect['category_id']]['name']}\n")
    #    for intersect in [d for d in detections if overlap(d['bbox'], dilated_bbox) > 0]:
    #        fd.write(f"detection: {category_by_id[intersect['category_id']]['name']} {intersect['score']:0.2f}\n")

    # add the categories and scores to the side
    # this is much slower than saving out the images and txt files
    plt.ioff()
    fig,ax = plt.subplots(1,1, dpi=90)
    plt.imshow(chip); plt.axis('off')

    # could write this in a single block of text, but then i'd loose the ability to color each line
    # colors match the colors above (0-1 instead of 0-255)
    _ = add_txt_annotation(ax, chip, dilated_bbox, annotation, categories, 0, '*annotation', [0,0,1])
    _ = add_txt_annotation(ax, chip, dilated_bbox, detection, categories, 1, '*detection', [0,1,0])
    # annotations
    offset = add_txt_annotation(ax, chip, dilated_bbox, [a for a in annotations if a != annotation], categories,
            2, 'annotation', [0,0,0.5])
    # detections/confusers
    _ = add_txt_annotation(ax, chip, dilated_bbox, [d for d in detections if d != detection], categories,
            offset, 'detection', [0,0.5,0])

    plt.savefig(outfile, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    plt.ion()
