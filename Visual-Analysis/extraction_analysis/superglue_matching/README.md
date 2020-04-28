## Dependencies
* Python 3
* PyTorch >= 1.1
* OpenCV >= 3.4
* Matplotlib >= 3.1
* NumPy >= 1.18

## Contents
The repo contains a top-level script `extraction_main.py` that runs the matching from image pair (ideal and target products) using the SuperGlue model (@ Magic Leap (CVPR 2020, Oral)), and subsequent mask superimposition over target product.

## Running
```sh
./extraction_main.py --ideal images/img.png --target images/img.png --mask path/to/mask.png --perception path/to/file.json --defect defect_type
```

Ideal and target images should be in the directory specified by the `--input_dir` flag (default: 'images/')

The `--resize` flag can be used to resize the input image in three ways:

1. `--resize` `width` `height` : will resize to exact `width` x `height` dimensions
2. `--resize` `max_dimension` : will resize largest input image dimension to `max_dimension`
3. `--resize` `-1` : will not resize (default)


The `--output_dir` flag (default: 'output/') is the directory to save the output matches. The matches are colored by their predicted confidence in a jet colormap (Red: more confident, Blue: less confident).



