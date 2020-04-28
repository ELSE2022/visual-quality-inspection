#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Apr 2020

@author: Adrien
"""

import argparse
import cv2
import numpy as np
import os
import json

from match import Match
from image_info import Image, Perception
from utils import (extract_rois, vis_rois, vis_defect,
				   update_perception, update_zone_infos, change_mask)



parser = argparse.ArgumentParser(description='Compares ideal product with the provided'
                                             ' template.')
parser.add_argument('--ideal', type=str, help='Path to the ideal image')
parser.add_argument('--target', type=str, help='Path to the target image')
parser.add_argument('--perception', type=str, help='Path to the perception')
parser.add_argument('--mask', type=str, help='Path to the ideal product mask')
parser.add_argument('--mirror', action='store_true', help='Mirror the target image')
parser.add_argument('--defect', choices={'stitches', 'holes'}, help='defect type')

args = parser.parse_args()
print(args)


################################################################################
image_name = args.ideal

image_name1 = args.target

visual_info = args.perception

mask = args.mask

defect_type = args.defect

inv = args.mirror
################################################################################


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


######### upload ideal and target images, and features matching #########


INFO = Image(image_name)
INFO1 = Image(image_name1, inv=inv)

mask = cv2.imread(mask)

INFO.mask_image(mask)

m = Match(INFO, INFO1)
m.get_mask()
m.vis()
shift = (m.axe0, m.axe1) # specify how many rows & columns we have
						 # to shift the ROI bboxes or mask to adapt on new images


######### Ideal/target product ROIs extraction and analysis #########

# - the list of ideal/target products ROIs pair as tuples
# - the list of updated bboxes infos
# - the list of ROIs name
inner_rois, inner_bboxes, inner_rois_names = extract_rois(INFO.image, INFO1.image,
														  inner_vis_zones,
														  inner_defects,
														  inner_zone_defect,
														  shift=shift)


# initialize zone_infos for digital trasformation
zone_infos = update_zone_infos(zone_infos, shoe_infos)

img = INFO1.image.copy()

for zone in inner_rois.keys():
	rois = inner_rois[zone]
	bboxes = inner_bboxes[zone]
	rois_name = inner_rois_names[zone]
	for i in range(len(rois)):
		### here for testing, we generate the voting ensemble output array with casual values
		voting = np.random.randint(0, 2, 6)
		img = vis_defect(img, shoe_name, voting, bboxes[i], rois_name[i], defect_type, ratio=0.2)
		perception["zones"] = update_perception(voting, rois_name[i], zone_infos,
												angle="inner", defect=defect_type)


with open('perception_updated.json', 'w') as f:
        json.dump(perception, f, indent=4)

print("\n", "Press 'Q' to close image windows!")
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()