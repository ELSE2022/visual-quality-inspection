import argparse
import cv2
import numpy as np
import os
import time
from utils import zoneColors, angleDefects


class Image:
    def __init__(self, image_name, image=None, size=None, inv=False):
        self.image_name = image_name
        if image is None:
            self.image = cv2.imread(self.image_name)
        else:
            self.image = image
        if size is not None:
            self.image = cv2.resize(self.image, size,
                                    interpolation=cv2.INTER_AREA)
        if inv:
            self.image = self.image[:, ::-1]

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.inverted = cv2.bitwise_not(self.image)
        self.masked_inv = self.inverted.copy()
        self.mask = np.zeros(self.image.shape)
        self.original_mask = np.zeros(self.image.shape)
        self.masked_image = self.image.copy()
        self._edit_image_name()

    def _edit_image_name(self):
        self.image_name = self.image_name[-self.image_name[::-1].find('/'):]
        self.image_name = self.image_name[:-4]

    def mask_image(self, _mask):
        self.original_mask = _mask
        self.mask = _mask
        assert self.mask.shape[0] > 0, "Mask is not defined correctly"
        if len(np.unique(self.mask)) > 2:
            self._change_mask()
        self.masked_image = np.ma.masked_array(data=self.image[:, :, :3],
                                               mask=self.mask,
                                               fill_value=0).filled()
        self.masked_inv = np.ma.masked_array(data=self.inverted[:, :, :3],
                                             mask=self.mask,
                                             fill_value=0).filled()

    def _change_mask(self):
        assert self.mask.shape[0] > 0, "Mask is not defined"
        self.mask = self.mask[:, :, 0] < 1
        self.mask = self.mask * 255
        self.mask = self.mask.reshape((self.mask.shape[0],
                                       self.mask.shape[1], 1))
        self.mask = np.concatenate((self.mask, self.mask, self.mask), axis=2)
        #cv2.imwrite('mask.png', self.mask)





class Perception:
    def __init__(self, perception):
        self.shoe_infos = perception["product"]
        self.angle_infos = perception["angles"]
        self.zone_infos = perception["zones"]
        self.shoe_name = self.shoe_infos["name"]
        self.zone_colors = zoneColors(self.zone_infos)
        self.inner_vis_zones = []
        self.back_vis_zones = []
        self.outer_vis_zones = []
        self.front_vis_zones = []


    def inner(self):
        ### read the angle's infos
        inner_angle = self.angle_infos["inner"]
        ### record angle's visible zones in list
        self.inner_vis_zones = inner_angle["visible_zones"]
        ### mask and image names, for both real & synthetic data
        inner_synth_mask = inner_angle["synth_mask"]
        inner_synth_img = inner_angle["synth_image"]
        inner_real_mask = inner_angle["real_mask"]
        inner_real_img = inner_angle["real_image"]

        inner_defects, zone_defect_type = angleDefects(self.inner_vis_zones,
                                                       self.zone_infos,
                                                       angle="inner")
        return inner_defects, zone_defect_type


    def back(self):
        ### read the angle's infos
        back_angle = self.angle_infos["back"]
        ### record angle's visible zones in list
        self.back_vis_zones = back_angle["visible_zones"]
        ### mask and image names, for both real & synthetic data
        back_synth_mask = back_angle["synth_mask"]
        back_synth_img = back_angle["synth_image"]
        back_real_mask = back_angle["real_mask"]
        back_real_img = back_angle["real_image"]

        back_defects, zone_defect_type = angleDefects(self.back_vis_zones,
                                                       self.zone_infos,
                                                       angle="back")
        return back_defects, zone_defect_type


    def outer(self):
        ### read the angle's infos
        outer_angle = self.angle_infos["outer"]
        ### record angle's visible zones in list
        self.outer_vis_zones = outer_angle["visible_zones"]
        ### mask and image names, for both real & synthetic data
        outer_synth_mask = outer_angle["synth_mask"]
        outer_synth_img = outer_angle["synth_image"]
        outer_real_mask = outer_angle["real_mask"]
        outer_real_img = outer_angle["real_image"]

        outer_defects, zone_defect_type = angleDefects(self.outer_vis_zones,
                                                       self.zone_infos,
                                                       angle="outer")
        return outer_defects, zone_defect_type


    def front(self):
        ### read the angle's infos
        front_angle = self.angle_infos["front"]
        ### record angle's visible zones in list
        self.front_vis_zones = front_angle["visible_zones"]
        ### mask and image names, for both real & synthetic data
        front_synth_mask = front_angle["synth_mask"]
        front_synth_img = front_angle["synth_image"]
        front_real_mask = front_angle["real_mask"]
        front_real_img = front_angle["real_image"]

        front_defects, zone_defect_type = angleDefects(self.front_vis_zones,
                                                       self.zone_infos,
                                                       angle="front")
        return front_defects, zone_defect_type

