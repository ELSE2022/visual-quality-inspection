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
INFO1.mask_image(m.new_mask)
shift = (m.axe0, m.axe1) # specify how many rows & columns we have
						 # to shift the ROI bboxes or mask to adapt on new images



######### Ideal mask superimposition over the target product #########

new_mask = change_mask(mask, shift)
cv2.imwrite("target_mask.png", new_mask)
cv2.imwrite('masked_target.png', INFO1.masked_image)

vis_rois(INFO1.image, inner_vis_zones, inner_defects, inner_zone_defect,
		 mask=new_mask, zones=zone_infos, zone_colors=zone_colors,
		 shoe_name=shoe_name, ratio=0.2, shift=shift)


print("\n", "Press 'Q' to close image windows!")
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()