# SGAN: Semi-Supervided GAN

An extension of the GAN architecture that involves the simultaneous training of a generator model, an unsupervised discriminator, and a supervised discriminator (classifier). The result is a supervised classification model that generalizes well to unseen examples and a generator model that outputs plausible examples of images from the domain.

## Requirement
- numpy==1.16.5
- opencv-python==4.2.0.32
- keras==2.3.1
- tensorflow-gpu==1.15
- matplotlib==3.1.1
- scikit-learn==0.21.3

## Training

Please arrange the files following data structure below.
```
input/train
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
python GAN_cl_main.py --train input/train
```

## Inference
Please arrange the files following data structure below.
```
input/test
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
Then run the command below for classification inference.
```bash
python GAN_Classifier_batch.py --test input/test --model_path <path/to/model_classifier.h5>
```
