DeepGUM: Learning Deep Robust Regression with a Gaussian-Uniform Mixture Model

## Introduction.

This is a Keras implementation of the work :
DeepGUM: Learning Deep Robust Regression with a Gaussian-Uniform Mixture Model
Stéphane Lathuilière, Pablo Mesejo, Xavier Alameda-Pineda, Radu Horaud, ECCV 2018

For more details [pdf](https://arxiv.org/abs/1808.09211)

Tested with keras 1.1.0 with theano backend and python 2.7.12
Requieres the installation of scikit-learn.

------------------
## How to run:

trainingAnnotations.txt must contain the list of the training images followed by the targets:
```
img_name_1.jpg y1 y2 y3
img_name_2.jpg y1 y2 y3 
...
```

testAnnotations.txt must contain the list of the test images with the same format

Download the [VGG16 weights](https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view)

Run the following command:
```shell
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,exception_verbosity='high' python deepGUM.py  $rootpathData trainingAnnotations.txt testAnnotations.txt $DIM $JOB_ID $OPTIONS
```

where JOB_ID is a job id used to save the network weights. You can give any number. $rootpathData is the path to your dataset folder. The file vgg16_weights.h5 must be moved in the $rootpathData folder.

DIM is the output space dimension

**OPTIONS:**
* probability model:
  * -i : Each dimension of the samples are treated separately.
  * -p : When predictions are landmarks, we consider that the pair (x,y) can be an outlier and not x or y only.
  * else: the full sample is an inlier or an outlier
  * -d : diagonal covriances
  * -iso : isotropic
* -u: update the uniform parameter distribution in the EM
* Validation mode:
  * -rnTra: the mixture parameters computed on the training set are used to compute the validation loss. In that case many samples may be considered as outlier since the variance of the validation is usually larger thatn for the training set. 
  * -rnHard: Same but hard decision to compute the loss (rn<0.5 -> rn=0 otherwise rn=1)
  * -rnEqui: we discard from the validation the same proportion od outlier than in the training set. It avoids the problem of rnTra


## Support

For any question, please contact [Stéphane Lathuilière](https://team.inria.fr/perception/team-members/stephane-lathuiliere/).
