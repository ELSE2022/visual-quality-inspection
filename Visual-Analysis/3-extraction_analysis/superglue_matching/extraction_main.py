#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Apr 2020

@author: Adrien
"""

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import json
import cv2
import os


from perception_info import Perception
from roi_utils import (extract_rois, vis_rois, shift, change_mask,
                       vis_defect, update_perception, update_zone_infos)

from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, process_resize)



parser = argparse.ArgumentParser(description='Compares ideal product with the provided'
                                             ' template.')
parser.add_argument('--ideal', type=str, help='Path to the ideal image')
parser.add_argument('--target', type=str, help='Path to the target image')
parser.add_argument('--perception', type=str, help='Path to the perception')
parser.add_argument('--mask', type=str, help='Path to the ideal product mask')
parser.add_argument('--defect', choices={'stitches', 'holes'}, help='defect type')
parser.add_argument('--resize', type=int, nargs='+', required=False,
                    help='Resize the input images to (w, h), before analysis')
parser.add_argument('--mirror', action='store_true', help='Mirror the target image')
parser.add_argument('--resize_float', action='store_true',
                    help='Resize the image after casting uint8 to float')
parser.add_argument('--force_cpu', action='store_true',
                    help='Force pytorch to run in CPU mode.')
parser.add_argument('--nms_radius', type=int, default=4,
                    help='SuperPoint Non Maximum Suppression (NMS) radius'
                    ' (Must be positive)')
parser.add_argument('--keypoint_threshold', type=float, default=0.005,
                    help='SuperPoint keypoint detector confidence threshold')
parser.add_argument('--max_keypoints', type=int, default=1024,
                    help='Maximum number of keypoints detected by Superpoint'
                    ' (\'-1\' keeps all keypoints)')
parser.add_argument('--superglue', choices={'indoor'}, default='indoor',
                    help='SuperGlue weights')
parser.add_argument('--sinkhorn_iterations', type=int, default=20,
                    help='Number of Sinkhorn iterations performed by SuperGlue')
parser.add_argument('--match_threshold', type=float, default=0.80,
                    help='SuperGlue match threshold')
parser.add_argument('--show_keypoints', action='store_true',
                    help='Plot the keypoints in addition to the matches')
parser.add_argument('--fast_viz', action='store_true',
                    help='Use faster image visualization with OpenCV'
                    'instead of Matplotlib')
parser.add_argument('--viz_extension', type=str, default='png', choices=['png', 'pdf'],
                    help='Visualization file extension. Use pdf for highest-quality.')
parser.add_argument('--opencv_display', action='store_true',
                    help='Visualize via OpenCV before saving output images')
parser.add_argument('--input_dir', type=str, default='images/',
                    help='Path to the directory that contains the images')
parser.add_argument('--output_dir', type=str, default='output/',
                    help='Path to the directory in which the .npz results,'
                    'and the visualization images are written')


args = parser.parse_args()
print(args)


################################################################################

image_name = args.ideal

image_name1 = args.target

defect_type = args.defect

visual_info = args.perception

mask = args.mask

size = args.resize

inv = args.mirror

################################################################################





if size is not None:
    resize = size
else:
    resize = [-1]  # do not resize the images for superglue matching




################################################################################


torch.set_grad_enabled(False)


assert not (args.opencv_display and not args.fast_viz), 'Cannot use --opencv_display without --fast_viz'
assert not (args.fast_viz and args.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'


if len(resize) == 2 and resize[1] == -1:
    resize = resize[0:1]
if len(resize) == 2:
    print('Will resize to {}x{} (WxH)'.format(
        resize[0], resize[1]))
elif len(resize) == 1 and resize[0] > 0:
    print('Will resize max dimension to {}'.format(resize[0]))
elif len(resize) == 1:
    print('Will not resize images')
else:
    raise ValueError('Cannot specify more than two integers for --resize')


img_name0 = os.path.split(image_name)[1]
img_name1 = os.path.split(image_name1)[1]

pairs = [[img_name0, img_name1]]

# Load the SuperPoint and SuperGlue models.
device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'

print('Running inference on device \"{}\"'.format(device))
config = {
    'superpoint': {
        'nms_radius': args.nms_radius,
        'keypoint_threshold': args.keypoint_threshold,
        'max_keypoints': args.max_keypoints
    },
    'superglue': {
        'weights': args.superglue,
        'sinkhorn_iterations': args.sinkhorn_iterations,
        'match_threshold': args.match_threshold,
    }
}


matching = Matching(config).eval().to(device)

# Create the output directories if they do not exist already.
input_dir = Path(args.input_dir)
print('Looking for data in directory \"{}\"'.format(input_dir))
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)
print('Will write matches to directory \"{}\"'.format(output_dir))



print('Will write visualization images to',
	'directory \"{}\"'.format(output_dir))




timer = AverageTimer(newline=True)
for i, pair in enumerate(pairs):
    name0, name1 = pair[:3]
    stem0, stem1 = Path(name0).stem, Path(name1).stem
    matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
    viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, args.viz_extension)
    

    # Handle --cache logic.
    do_match = True
    do_viz = True
    
    

    # If a rotation integer is provided (e.g. from EXIF data), use it:
    if len(pair) >= 5:
        rot0, rot1 = int(pair[2]), int(pair[3])
    else:
        rot0, rot1 = 0, 0

    # Load the image pair.
    image0, inp0, scales0 = read_image(
        input_dir / name0, device, resize, rot0, args.resize_float)
    image1, inp1, scales1 = read_image(
        input_dir / name1, device, resize, rot1, args.resize_float)
    
    if image0 is None or image1 is None:
        print('Problem reading image pair: {} {}'.format(
            input_dir/name0, input_dir/name1))
        exit(1)
    timer.update('load_image')

    if do_match:
        # Perform the matching.
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        timer.update('matcher')

        # Write the matches to disk.
        out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                       'matches': matches, 'match_confidence': conf}
        np.savez(str(matches_path), **out_matches)

    # Keep the matching keypoints.
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]

    if do_viz:
        # Visualize the matches.
        color = cm.jet(mconf)
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0)),
        ]
        if rot0 != 0 or rot1 != 0:
            text.append('Rotation: {}:{}'.format(rot0, rot1))

        # Display extra parameter info.
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {}:{}'.format(stem0, stem1),
        ]

        make_matching_plot(
            image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
            text, viz_path, args.show_keypoints,
            args.fast_viz, args.opencv_display, 'Matches', small_text)

        timer.update('viz_match')

    

    timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))


stem0, stem1 = Path(img_name0).stem, Path(img_name1).stem
path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)




######### .npz file result analysis #########

npz = np.load(path)

kp0, kp1 = npz['keypoints0'], npz['keypoints1']
matches = npz['matches']
confidence = npz['match_confidence']

# For each keypoint in `keypoints0`, the `matches` array indicates the index
# of the matching keypoint in `keypoints1`, or `-1` if the keypoint is unmatched.

# rows & columns we have to shift
# the ROI bboxes to adapt on new images
shifted = shift(kp0, kp1, matches, confidence)


######### upload the perception data #########

with open(visual_info, 'r') as f:
    perception = json.load(f)


p = Perception(perception)

shoe_infos = p.shoe_infos

if defect_type not in shoe_infos["defects"]:
    raise ValueError("The specified defect is not present in shoe's model.")


angle_infos = p.angle_infos
zone_infos = p.zone_infos
shoe_name = p.shoe_name
### dict recording the mask colors per zone
zone_colors = p.zone_colors
### record the defects per visible zones, and the defect types per zone
inner_defects, inner_zone_defect = p.inner()
inner_vis_zones = p.inner_vis_zones


# read and resize the input images
ideal = cv2.imread(image_name)
orig_shape = ideal.shape
target = cv2.imread(image_name1)
w, h = orig_shape[1], orig_shape[0]
w_new, h_new = process_resize(w, h, resize)

ideal = cv2.resize(ideal, (w_new, h_new))
target = cv2.resize(target, (w_new, h_new))



######### Ideal/target product ROIs extraction and analysis #########

# - the list of ideal/target products ROIs pair as tuples
# - the list of updated bboxes infos
# - the list of ROIs name
inner_rois, inner_bboxes, inner_rois_names = extract_rois(ideal, target,
                                                          inner_vis_zones,
                                                          inner_defects,
                                                          inner_zone_defect,
                                                          orig_shape,
                                                          shift=shifted)



# initialize zone_infos for digital trasformation
zone_infos = update_zone_infos(zone_infos, shoe_infos)

img = target.copy()

for zone in inner_rois.keys():
    rois = inner_rois[zone]
    bboxes = inner_bboxes[zone]
    rois_name = inner_rois_names[zone]
    for i in range(len(rois)):
        ### here for testing, we generate the voting ensemble output array with casual values
        voting = np.random.randint(0, 2, 6)
        img = vis_defect(img, shoe_name, voting, bboxes[i], rois_name[i], defect_type)
        perception["zones"] = update_perception(voting, rois_name[i], zone_infos,
                                                angle="inner", defect=defect_type)


with open('perception_updated.json', 'w') as f:
        json.dump(perception, f, indent=4)


print("\n", "Press 'Q' to close image windows!")
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()