# CycleGAN: Cycle Generative Adversarial Network

Approach for training a deep convolutional neural network for unsupervised image-to-image translation tasks (domain adaptation).

## Requirement
- numpy==1.16.5
- opencv-python==4.2.0.32
- keras==2.3.1
- keras-contrib (github.com/keras-team/keras-contrib.git)
- tensorflow-gpu==1.15
- matplotlib==3.1.1

### Data preparation

Please arrange the files following data structure below.
```
input/real_and_synth
└───real
    └───001.jpg
    └───002.jpg
    └───003.jpg
    ...
└───synth
    └───001.jpg
    └───002.jpg
    └───003.jpg
    ...
```
And then run the script below

```bash
python data_prep.py --path input/real_and_synth
```

### Training command

```bash
python CycleGANs_main.py
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
Then run the command below.
In this script our classes are ['Ked_high', 'Ked_low', 'Slip_on'].
```bash
python cycleGAN_inference_batch.py --path input/test --model <path/to/saved_model.h5>
```
