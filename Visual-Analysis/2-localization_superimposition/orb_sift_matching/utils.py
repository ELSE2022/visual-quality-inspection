

import cv2
import numpy as np


def multiDimenDist(point1, point2):
    # type: (point1) -> list
    # type: point2 -> list
    # returns length of the vector
    deltas = [point2[dimension] - point1[dimension] for dimension in range(
        len(point1))]
    length = 0
    for coord in deltas:
        length += coord**2
    return length**(1/2)



def change_mask(mask, shift):
    """
    function to update the mask to match the target image
    """
    new_mask = mask.copy()
    new_mask = np.roll(new_mask, shift[0], axis=0)
    new_mask = np.roll(new_mask, shift[1], axis=1)

    return new_mask


def zoneColors(zones):
    """
    zones: dict containing all the zones infos, available in the perception
    return -> a dict recording the mask colors for each zone
    """
    zone_colors = {}
    for zone in zones.keys():
        mask_color = zones[zone]["mask_color"]
        zone_colors[zone] = mask_color

    return zone_colors




def angleDefects(angle_vis_zones, zones, angle):
    """
    angle_vis_zones: a list recording angle's visible zones
    zones: dict containing all the zones infos, available in the perception
    angle: string, specify the angle view

    return: 
    -> angle_defects: dict recording the angle's defects per visible zones
    -> zone_defect_type : dict recording the defect types per zone

    """

    angle_defects = {}
    zone_defect_type = {}

    ### loop on inner angle zones and record relevant data
    for zone in angle_vis_zones:
        perception_zone = zones[zone]
        if "defects" not in perception_zone.keys():
            continue

        if zone not in zone_defect_type.keys():
                zone_defect_type[zone] = []
        defects = perception_zone["defects"]
        angle_defects[zone] = {}        # dictionary to record defect name and infos

        for defect in defects:
            if angle not in defect["roi"].keys():
                continue
            defect_name = defect["name"]
            defect_prob = defect["probability"]
            defect_weight = defect["weight"]
            defect_roi = defect["roi"][angle]
            defect_infos = {"roi":defect_roi, "prob":defect_prob, "weight":defect_weight}
            angle_defects[zone][defect_name] = defect_infos

            if defect_name not in zone_defect_type[zone]:
                zone_defect_type[zone].append(defect_name)

    return angle_defects, zone_defect_type




def extract_rois(ideal, target, angle_vis_zones, angle_defects,
                 zone_defect_type, shift, save=False):
    """
    Function to extract ROIs on input images.

    ideal: array, image of the ideal product to use for ROIs extraction
    target: array, image of the target product to use for ROIs extraction
    angle_vis_zones: a list recording angle's visible zones
    zone_defect_type : dict recording the defect types per zone
    angle_defects: dict recording the angle's defects per visible zones
    shift: tuple, specify how many rows & columns we have to shift the ROI bboxes to adapt on new images
    save: bool, indicate whether to locally save ROIs extracted from both input images

    return:
    -> angle_rois: dict recording the list of ROIs pair as tuples, for each zone
    -> angle_bbox: dict recording the rois bbox info, for each zone
    -> rois_names: dict recording the rois names, for each zone
    """

    angle_rois = {}
    angle_bbox = {}
    rois_names = {}
    
    for zone in angle_vis_zones:
        if zone not in zone_defect_type.keys():
            continue

        angle_rois[zone] = []
        angle_bbox[zone] = []
        rois_names[zone] = []
        defects = zone_defect_type[zone]
        for defect in defects:
            rois = angle_defects[zone][defect]["roi"]
            prob = angle_defects[zone][defect]["prob"]
            weight = angle_defects[zone][defect]["weight"]
            i = 0
            for roi in rois:
                x0, y0 = roi[0]["x"], roi[0]["y"]
                x1, y1 = roi[1]["x"], roi[1]["y"]
                crop_ideal = ideal[y0:y1, x0:x1]
                crop_target = target[y0+shift[0]:y1+shift[0], x0+shift[1]:x1+shift[1]]
                angle_rois[zone].append((crop_ideal, crop_target))
                angle_bbox[zone].append((x0+shift[1], y0+shift[0], x1+shift[1], y1+shift[0]))
                rois_names[zone].append(zone + "_roi-" + str(i))

                if save:
                    cv2.imwrite("output/" + zone + "_roi_" + str(i) + "_ideal.jpg", crop_ideal)
                    cv2.imwrite("output/" + zone + "_roi_" + str(i) + "_target.jpg", crop_target)
                i += 1

    return angle_rois, angle_bbox, rois_names



def vis_rois(image, angle_vis_zones, angle_defects,
             zone_defect_type, shoe_name, shift,
             mask=None, zones=None, zone_colors=None,
             ratio=1., save=False):
    """
    Function to visualize ROIs on input image.

    image: array, input image for ROIs visualization
    mask: array, the image's mask
    angle_vis_zones: a list recording angle's visible zones
    zone_defect_type : dict recording the defect types per zone
    angle_defects: dict recording the angle's defects per visible zones
    zones: dict containing all the zones infos, available in the perception
    zone_colors: dict, specify the mask colors for each zone
    shoe_name : shoe's name to display
    shift: tuple, specify how many rows & columns we have to shift the ROI bboxes to adapt on image
    ratio: float, [0, 1] specify the image resizing ratio to use during visualization
    save: bool, indicate whether to locally save visual output
    """


    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dsize = (width, height)

    for zone in angle_vis_zones:
        if zone not in zone_defect_type.keys():
            continue

        defects = zone_defect_type[zone]
        
        img = image.copy()
        if (mask is not None) and (zones is not None and zone_colors is not None):
            zone_color = zone_colors[zone]
            r, g, b = bytes.fromhex(zone_color)
            zone_color = (b, g, r)
            comp = (mask[:, :, 0] == b) & (mask[:, :, 1] == g) & (mask[:, :, 2] == r)
            bgr = (mask != 0)
            zone_mask = mask.copy()

            zone_mask[bgr] = 0
            zone_mask[comp] = zone_color
            cv2.addWeighted(img, 1, zone_mask, 0.3, 0, img)
            
            add_zones = zones[zone]["defects"][0]["additional_zones"]
            for additional in add_zones:
                add_mask = mask.copy()
                additional_color = zone_colors[additional]
                r1, g1, b1 = bytes.fromhex(additional_color)
                additional_color = (b1, g1, r1)
                comp1 = (mask[:, :, 0] == b1) & (mask[:, :, 1] == g1) & (mask[:, :, 2] == r1)
                add_mask[bgr] = 0
                add_mask[comp1] = additional_color
                cv2.addWeighted(img, 1, add_mask, 0.3, 0, img)

        for defect in defects:
            color = (255,0,0)
            rois = angle_defects[zone][defect]["roi"]
            prob = angle_defects[zone][defect]["prob"]
            weight = angle_defects[zone][defect]["weight"]
            i = 0
            for roi in rois:
                x0, y0 = roi[0]["x"] + shift[1], roi[0]["y"] + shift[0]
                x1, y1 = roi[1]["x"] + shift[1], roi[1]["y"] + shift[0]
                #print("working on zone", zone + ": roi" + str(i))
                cv2.rectangle(img, (x0, y0), (x1, y1), color, 3)
                imgS = cv2.resize(img, dsize)
                cv2.rectangle(imgS, (5, 5), (190, 90), color, 1)
                cv2.putText(imgS, "-Shoe name: " + shoe_name, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(imgS, "-Angle: inner", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(imgS, "-ROIs: " + defect, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(imgS, "-Zone: ", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if (mask is not None) and (zones is not None and zone_colors is not None):
                    cv2.putText(imgS, zone, (75, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 1)
                else:
                    cv2.putText(imgS, zone, (75, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                i += 1
                cv2.imshow("ROI", imgS)
                if len(rois) != i:
                    cv2.waitKey(60)
                else:
                    cv2.waitKey(3000)
                
        if save:
            cv2.imwrite("output/angle/" + zone + "_defect.jpg", img)







def vis_defect(image, shoe_name, voting, bboxes, rois_name, defect,
               ratio=1., threshold=0.7, save=False):
    """
    Function to visualize ROIs on input image.

    image: array, input image for ROIs visualization
    voting: array, voting ensemble output
    bboxes : dict, rois bboxes info
    rois_name: dict, rois name
    shoe_name : shoe's name to display
    defect: str, defect type
    ratio: float, [0, 1] specify the image resizing ratio to use during visualization
    threshold: float, percentage threshold for decision making
    save: bool, indicate whether to locally save visual output

    Return: image with updated defect bboxes
    """


    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dsize = (width, height)

    color = (255,0,0)
    zone = rois_name.split("_")[0]

    img = image.copy()
    (x0, y0, x1, y1) = bboxes
    c = (255,255,255)

    for i in range(2):
        cv2.rectangle(img, (x0, y0), (x1, y1), c, 3)
        if c == (0,0,255):
            image = img.copy()

        imgS = cv2.resize(img, dsize)
        cv2.rectangle(imgS, (5, 5), (190, 90), color, 1)
        cv2.putText(imgS, "-Shoe name: " + shoe_name, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(imgS, "-Angle: inner", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(imgS, "-ROIs: " + defect, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(imgS, "-Zone: ", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(imgS, zone, (75, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imshow("ROI", imgS)
        cv2.waitKey(60)

        if np.sum(voting) > threshold*len(voting):
            c = (0,0,255)

    imgS = cv2.resize(image.copy(), dsize)
    cv2.rectangle(imgS, (5, 5), (190, 90), color, 1)
    cv2.putText(imgS, "-Shoe name: " + shoe_name, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(imgS, "-Angle: inner", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(imgS, "-ROIs: " + defect, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(imgS, "-Zone: ", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(imgS, zone, (75, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.imshow("ROI", imgS)
    cv2.waitKey(60)

    if save:
        cv2.imwrite("output/angle/" + zone + "_defect.jpg", img)

    return image



def update_perception(voting, rois_name, zone_infos, defect, angle="inner", threshold=0.7):
    """
    Function to update perception.

    voting: array, voting output
    rois_name: list, name of the ROIs
    bboxes: list, bboxes of the ROIs
    zone_infos : dict, containing all the zones infos, available in the perception
    defect: str, defect type
    angle: str, angle view
    threshold: float, percentage threshold for decision making
    
    Return:
    zone_infos: dict, updated zones infos
    """


    zone = rois_name.split("_")[0]
    roi_name = rois_name.split("_")[-1]
    
    for dft in zone_infos[zone]["defects"]:
        if dft["name"] != defect:
            continue

        if np.sum(voting) > threshold*len(voting):
            dft["defected_roi"][angle].append(roi_name)


    return zone_infos



def update_zone_infos(zone_infos, shoe_infos):
    """
    function to initialize the zone_infos for update
    zone_infos : dict, containing all the zones infos, available in the perception
    """
    for zone in shoe_infos["zones"]:
        if "defects" not in zone_infos[zone].keys():
            continue

        for defect in zone_infos[zone]["defects"]:
            defect["defected_roi"] = {}
            defect["defected_roi"]["inner"] = []
            defect["defected_roi"]["back"] = []
            defect["defected_roi"]["outer"] = []
            defect["defected_roi"]["front"] = []

    return zone_infos