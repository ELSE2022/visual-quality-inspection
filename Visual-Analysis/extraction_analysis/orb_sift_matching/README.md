## Dependencies
* Python 3
* OpenCV >= 3.4 (4.1.2.30 recommended)
* NumPy >= 1.18


## Contents
The repo contains a top-level script `extraction_main.py` that runs the ORB/SIFT Features matching from image pair (ideal and target products) and subsequent rois extraction and analysis.

## Running
```sh
./extraction_main.py --ideal images/img.png --target images/img.png --mask path/to/mask.png --perception path/to/file.json --defect defect_type
```




