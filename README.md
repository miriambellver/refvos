# RefVOS


In this repository there is the code of the method for referring image segmentation and language-guided video object segmentation presented in the work: [RefVOS: A Closer Look At Referring Expressions for Video Object Segmentation](https://arxiv.org/pdf/2010.00263.pdf). We also provide category labels for the A2D and DAVIS 2017 datasets, and augmented REs for A2D, which are in the folder *datasets*. In the paper you can find the details about our proposed categorization and extension of A2D referring expressions.


### Datasets

We used our method for referring image segmentation with RefCOCO and RefCOCO+. Both datasets can be downloaded in the [official site](https://github.com/lichengunc/refer).

For language-guided video object segmentation (L-VOS) we used [DAVIS 2017](https://davischallenge.org) and [A2D datasets](https://web.eecs.umich.edu/~jjcorso/r/a2d/), which can be found in their official sites. The language queries used in the experiments can be found in DAVIS 2017 website. Regarding A2D, the language queries can be found in this [project site](https://kgavrilyuk.github.io/publication/actor_action/).


### Installation instructions

The requisites to use this code are to install the following packages:

```bash
pip3 install torch
pip3 install transformers
pip3 install torchvision
pip3 install pycocotools
pip3 install scipy
pip3 install scikit-image
pip3 install h5py 
```

Following, *refer* has to be installed as well. You should substitute the original refer.py file with our custom one, as it is adapted for python3. 

```bash
git clone https://github.com/lichengunc/refer.git
cp ./refer_utils/refer.py ./refer/refer.py
make
```

### RefVOS Pre-trained weights

Download the pre-trained weights for our models using the following links:

[Model trained on RefCOCO](https://drive.google.com/file/d/1VI2TixrkjDORirkGGi3eXDJj35CRoEVr/view?usp=sharing)

[Model trained on RefCOCO+](https://drive.google.com/file/d/1HIM3xHkL2Z1rCnnA6OF3r8SmroL7EoQy/view?usp=sharing)

[Model trained on A2D](https://drive.google.com/file/d/1CqwYTwcD0lQ0VHJMJJ9iOmRGvjj3Eiuf/view?usp=sharing)

[Model trained on RefCOCO and then A2D](https://drive.google.com/open?id=1Y4sclYO4wViw-gH2nrLZ-XmLq0_FXpWf)

[Model trained on DAVIS](https://drive.google.com/open?id=1H3S4bZQChIlJNAM5mttbjTAr7pnOk1PL)

We recommend to create a folder in the main directory named *checkpoints* and save these checkpoints there.

### BERT Pre-trained weights

As explained in our article, we first train BERT with RefCOCO or RefCOCO+ expressions. These pre-trained models are in the following links:

[Bert trained on RefCOCO](https://drive.google.com/file/d/1-hpF7UwA-cza2MpT75fyHEsKLe7xgpGc/view?usp=sharing)

[Bert trained on RefCOCO+](https://drive.google.com/file/d/1FmDjRj66YXG4Hv8X1nagwmaSxhVC7Y7_/view?usp=sharing)

We recommend to create a folder in the main directory named *checkpoints_bert*, and to unzip these two files there. You will see two folders: bert_pretrained_refcoco and bert_pretrained_refcoco+.

### Training

For training the models with the different datasets, the command is the following:
```bash
python3 train.py --dataset refcoco  --model_id model
```
 
with the argument --dataset as 'refcoco', 'refcoco+', 'a2d' or 'davis' and in the argument --model_id the model name you want to set. Check args.py for more arguments. You may need to add arguments to indicate paths to the datasets.

Other possible configurations to train models are the following:

- In our work we explain how we first train BERT with the referring expressions given in the RefCOCO or RefCOCO+ sets. Then we plug these pre-trained models to RefVOS and start the training for the referring image segmentation task. If you want to leverage such pre-trained models for BERT, you can first download the BERT pre-trained weights, and use the following command:

```bash
python3 train.py --dataset refcoco  --model_id model_pretrained_bert --ck_bert ./checkpoints_bert/bert_pretrained_refcoco
```

The same applies for RefCOCO+ but changing refcoco for refcoco+.

- If you want to train a model starting from pre-trained weights (for instance on RefCOCO), the command you can use is the following:

```bash
python3 train.py --dataset a2d  --model_id a2d_model --pretrained_refvos --ck_pretrained_refvos ./checkpoints/model_refcoco.pth
```

where --pretrained_refvos indicates to load the checkpoint indicated in --ck_pretrained_refvos and pre-trained RefVOS with it.

### Testing

For testing the models with the different datasets, the command is the following:

```bash
python3 test.py  --dataset refcoco --resume ./checkpoints/model_refcoco.pth --split val
```

with the --dataset argument with 'refcoco', 'refcoco+', 'a2d' or 'davis' and in --resume argument, the checkpoint you want to evaluate. The --split argument indicates which split to evaluate. In RefCOCO and RefCOCO+ there are the val, testA and testB splits. In case you want to save the results, add the argument --display. Check args.py for more arguments. 

### Training Details

We use SGD optimizer with 0.9 of momentum and 1e-6 of weight decay. We train our model with batch size of 7 and 480x480 of resolution. The learning rate scheduler for RefCOCO and RefCOCO+ consists in first starting with learning rate of 0.01 and decrease it 0.0004 every epoch until reaching 3e-3 of learning rate. Then we increase the learning rate again to 6e-3 and decrease it 2.5e-4 every epoch until reaching 1e-3. Increasing the learning rate after some iterations has been proven to increase performance in previous works. Finally we set a final state with fixed learning rate of 3e-5. To later fine-tune on DAVIS, we set an initial learning rate of 1e-3 and decrease 1e-5 every epoch for 20 epochs. To train the model with A2D we set a learning rate of 0.1 and decrease it 0.004 for 15 epochs. We used a single Tesla V100 GPU to train the models. The average time for training the model with RefCOCO is about 3 days, 2 days for A2D and 1 day for DAVIS-2017 with our machines. The approximate number of parameters of our model, considering also the BERT encoder, is 170M.
