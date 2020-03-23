# RefVOS

This code is referring image segmentation and language-guided video object segmentation.

### Datasets

We used our method for referring image segmentation with RefCOCO and RefCOCO+. Both datasets can be downloaded in the [official site](https://github.com/lichengunc/refer).

For language-guided video object segmentation (L-VOS) we used [DAVIS 2017](https://davischallenge.org) and [A2D datasets](https://web.eecs.umich.edu/~jjcorso/r/a2d/), which can be respectively be found in their official sites. The language queries used in the experiments can be found in Khoreva et al. work and are found in DAVIS 2017 website. Regarding A2D, the language queries are from Gavrilyuk et al. and are found in their [project site](https://kgavrilyuk.github.io/publication/actor_action/).


### Installation instructions

### Training

For training the models with the different datasets, the command is the following:

  python3 train.py --dataset refcoco  --model_id model
  
with the argument of the dataset as 'refcoco', 'refcoco+', 'a2d' or 'davis' and in the argument of model_id, the model name you want to set. Check args.py for more arguments. 

### Testing

For testing the models with the different datasets, the command is the following:

  python3 test.py  --dataset refcoco --resume ./checkpoints/model_refcoco.pth 
  
with the argument of the dataset as 'refcoco', 'refcoco+', 'a2d' or 'davis' and in the argument of resume, the chekpoint you want to evaluate. Check args.py for more arguments. 
