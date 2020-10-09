# cVAE: Conditional Variational Auto-Encoder

Training a Variational Auto-Encoder conditioned on the input classes.

## Requirement
- numpy==1.16.5
- opencv-python==4.2.0.32
- keras==2.3.1
- tensorflow-gpu==1.15
- matplotlib==3.1.1

## Training

Please arrange the files following data structure below.
```
input
└───Class1
    └───001.jpg
    └───002.jpg
    └───003.jpg
    ...
└───Class2
    └───001.jpg
    └───002.jpg
    └───003.jpg
    ...
└───Class3
    └───001.jpg
    └───002.jpg
    └───003.jpg
    ...
```
In this script our classes are ['Ked_high', 'Ked_low', 'Slip_on'].

### Training command

```bash
python VAE_cond_main.py --directory input --epochs <num of epochs> --batch_size <batch size>
```

## Inference
- Generate and save latent space vectors for our training images. Here the latent space are 20-dimensional vectors.

```bash
python image2latent.py --directory input --weights <path/to/encoder_weight.h5>
```

- Create a trackbar for label-based conditioned realtime latent space vector representation analysis.

```bash
python latent_trackbar.py --weights <path/to/decoder_weight.h5>
```
