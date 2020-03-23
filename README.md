# RefVOS

This code is for referring image segmentation and language-guided video object segmentation.

### Datasets

We used our method for referring image segmentation with RefCOCO and RefCOCO+. Both datasets can be downloaded in the [official site](https://github.com/lichengunc/refer).

For language-guided video object segmentation (L-VOS) we used [DAVIS 2017](https://davischallenge.org) and [A2D datasets](https://web.eecs.umich.edu/~jjcorso/r/a2d/), which can be respectively be found in their official sites. The language queries used in the experiments can be found in Khoreva et al. work and are found in DAVIS 2017 website. Regarding A2D, the language queries are from Gavrilyuk et al. and are found in their [project site](https://kgavrilyuk.github.io/publication/actor_action/).


### Installation instructions

The requisites to use this code are to have installed the following:

pip3 install torch
pip3 install transformers
pip3 install torchvision
pip3 install pycocotools
pip3 install scipy
pip3 install scikit-image

Following refer has to be installed as well, you should substitute the original refer.py file with our custom one, as it is adapted for python3. 

git clone https://github.com/lichengunc/refer.git
cp ./refer_utils/refer.py ./refer/refer.py
make


### Pre-trained weights

Download the pre-trained weights for our models using the following links:

Model trained on RefCOCO

Model trained on RefCOCO+

[Model trained on A2D](https://drive.google.com/file/d/1CqwYTwcD0lQ0VHJMJJ9iOmRGvjj3Eiuf/view?usp=sharing)

[Model trained on RefCOCO and then A2D](https://drive.google.com/open?id=1Y4sclYO4wViw-gH2nrLZ-XmLq0_FXpWf)

[Model trained on DAVIS](https://drive.google.com/open?id=1H3S4bZQChIlJNAM5mttbjTAr7pnOk1PL)

### Training

For training the models with the different datasets, the command is the following:
```bash
python3 train.py --dataset refcoco  --model_id model
```
  
with the argument of the dataset as 'refcoco', 'refcoco+', 'a2d' or 'davis' and in the argument of model_id, the model name you want to set. Check args.py for more arguments. 

### Testing

For testing the models with the different datasets, the command is the following:

```bash
python3 test.py  --dataset refcoco --resume ./checkpoints/model_refcoco.pth 
``` 

with the argument of the dataset as 'refcoco', 'refcoco+', 'a2d' or 'davis' and in the argument of resume, the chekpoint you want to evaluate. Check args.py for more arguments. 
