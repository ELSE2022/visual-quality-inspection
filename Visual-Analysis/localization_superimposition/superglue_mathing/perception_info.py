

from roi_utils import zoneColors, angleDefects

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

