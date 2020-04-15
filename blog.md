---
mathjax: true
---
<header class="post-header">
    {% include mathjax.html %}
</header>

# One-Shot Image Recognition Using Siamese Networks

Conventionally, deep neural networks have been really good at learning from high dimensional data like images, audio, and video provided that there are huge amounts of labeled examples to learn from. Comparatively, humans are capable of what is called "one-shot learning". If you show a picture of a ball to a human who has never seen a football (soccer ball for Americans) before, they will probably be able to distinguish footballs from rugby balls, baseballs, basketballs, and so on with high accuracy.

![one shot learning](https://raw.githubusercontent.com/sharwinbobde/siamese-nn-oneshot-reproduction/gh-pages/images/one_shot.png)
> Figure depicting a one-shot image recognition task
> _Taken from: Siamese Neural Networks for One-Shot Image Recognition._

Yet another one of the things humans can do that seemed trivial until we tried to make an algorithm do it.

This ability to learn accurately from little data is desirable for a machine learning system to have because collecting and labeling data is expensive.

The one-shot learning problem has been studied extensively before, but in this post, we will focus on one approach by Gregory Koch et al. which uses an interesting Deep Neural Network called a Siamese Network.[^one-shot-paper]

In this post, we will:

1. [Introduce and formulate the problem of one-shot image recognition](#1.-One-Shot-Image-Recognition)

2. [Describe the Siamese Network and the author's approach to One-Shot Image Recognition](#2.-How/Why-Siamese-Networks-Work)

3. [Understand the dataset and the experimental setup for the image recognition task](#3.-Experimental-Setup)

4. [Try to reproduce the results](#4.-Results)

5. [See some interesting insights](#5.-Insights)

For a more in depth look at the code, we refer you to our [Jupyter Notebook](https://github.com/sharwinbobde/siamese-nn-oneshot-reproduction/blob/master/notebooks/test-bench/siamese-colab.ipynb)


## 1. One-Shot Image Recognition

Unlike normal classification tasks, which consists of classifying each image into their correct class after training on lots of images from the same class, one-shot classifications learns to discriminate samples with access to just 1 image per class. How is that possible? One-shot classification uses a trick up its sleeve: instead of mapping images to many classes, it maps pairs of images to just two classes: same or different.

In this case, instead of classifying an image to one of the many classes, we input two images at the same time, and the network outputs whether these images represent the same object or not. The network must have the same weights when transforming each pair of images. This property can be applied to more realistic scenarios like face recognition. 

![traditional_classification](https://raw.githubusercontent.com/sharwinbobde/siamese-nn-oneshot-reproduction/gh-pages/images/traditional.png)

> Traditional way of training a CNN to distinguish the class of an image
> _Taken from [Oneshot Explanation](https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d)_

Imagine you have a database with photos of university employees and you want to implement a face unlock functionality in some classrooms. Without one-shot classification you would need to gather hundreds of photos of each employee in order to train a classifier with as many classes as employees. With one-shot learning however, you just need one photo per employee, and two classes: same and different.

![traditional_classification](https://raw.githubusercontent.com/sharwinbobde/siamese-nn-oneshot-reproduction/gh-pages/images/siamese.png)

> In a siamese network we input two images at the same time and output a similarity score between 0 and 1
> _Taken from [Oneshot Explanation](https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d)_

A straightforward approach would involve creating pairs of images from all the photos of the employees and label those depending on whether they portray the same person. After training, the outcome is a Siamese network capable of distinguishing similar from dissimilar objects, at a much lower cost of data gathering.

## 2. Siamese Networks
![si and am](https://raw.githubusercontent.com/sharwinbobde/siamese-nn-oneshot-reproduction/gh-pages/images/si_and_am.png)

### 2.1 Why Siamese Networks?
Koch et al.'s approach to one-shot classification is to give a network two images and train it to learn whether they belong to the same category. Then when doing a one-shot classification task described above, the network can compare the test image to each image in the support set, and picks which one it thinks is most likely to be of the same category.

> So, intuitively we want a neural net that takes two images as input and outputs the probability they share the same class.

Say $X_1$ and $X_2$ are two images in our dataset. If we define $X_1 \bigodot X_2$ to mean  "$X_1$ and $X_2$ are images with the same class", $X_1 \bigodot X_2$ is the same as $X_2 \bigodot X_1$. We therefore need a neural net whose output should be the same even if we reverse the order of the inputs

$$
p(X_1 \bigodot X_2) = p(X_2 \bigodot X_2)
$$

Siamese Nets are designed to have this symmetry property. Symmetry is important because it’s required for learning a distance metric: Distance from $X_1$ to $X_2$ = Distance from $X_2$ to $X_1$.

Concatenation of the two images may result in the images being convolved with a different set of weights which would break symmetry. Technically it is possible to learn the same weights for both images but there are no guarantees. It would be much easier to learn a single set of  weights applied to both inputs. Then we can propagate both inputs  through identical neural nets with shared parameters and use a distance metric as an input to a linear classifier. This is in essence a Siamese net. Two identical twins, joined at the  head, hence the name.

### 2.2 Architecture 
Unfortunately, properly explaining how and why a CNN would make this post twice as long. If you want to understand CNNs, we suggest checking out cs231n[^cs231n] and then colah[^colah].

Koch et al uses a convolutional siamese network (CNN) to classify pairs of images, each of the networks in the pair are CNNs. The twins have the following architecture:


```
64@10x10 CONV -> MAXPOOL -> 128@7x7 CONV -> MAXPOOL -> 128@4x4 CONV -> MAXPOOL -> 256@4x4 CONV
```

![image](https://raw.githubusercontent.com/sharwinbobde/siamese-nn-oneshot-reproduction/gh-pages/images/siamese_net.png)
> Deptiction of a Convolutional Siamese Net
> _Taken from https://sorenbouma.github.io/blog/oneshot/_

Like in any modern basic CNN, the twin networks reduce their inputs down to smaller and smaller outputs. Final part of each twin is a fully  connected layer with 4096 units. The absolute difference (L1 distance) between the two vectors is used as input to a linear classifier.

The output is squashed into $[0,1]$ with a sigmoid activation function to make it a probability. We use the target $t=1$ when the images belong to the same class and $t=0$ when they belong to different classes. The loss function is a binary cross-entropy between the  predictions and targets. There is also a L2 weight decay term to improve generalization:
$$
L(X_1, X_2, t) = t\log(p(X_1\bigodot X_2)) + (1-t)\log(1-p(X_1\bigodot X_2)) + \lambda^T |w|^2
$$
In the one-shot task, the network simply classifies the test image as whatever image in the support set it thinks is most similar to the test image. We describe this below in [section 4.2](#4.2-One-Shot-Learning)

## 3. Experimental Setup
In this section, we describe the setup required to reproduce the paper. You can find the code on our GitHub page[^github].

### 3.1 Omniglot Dataset
For training the Siamese network to discern equal from different sets of images, we make use of the Omniglot dataset[^omniglot]. This dataset consists of drawings of characters from 40 different alphabets, some of them real like Bengali, and some of them made up like the Futurama alphabet.

![oneshot_characters](https://raw.githubusercontent.com/sharwinbobde/siamese-nn-oneshot-reproduction/gh-pages/images/characters.png)

> Image showing an example of 8 alphabets of the dataset
> _Taken from: "Siamese Neural Networks for One-shot Image Recognition"_



Each alphabet can have a different number of letters in it, and each letter is drawn by 20 different "drawers" or artists, which interpret the original characters and try to produce a hand-written version of them.

The authors of the paper and us in accordance divide the dataset into three different sets:

- **Training set**: Consisting of 20 alphabets and 12 out of the 20 drawers or alternative representations of each character
- **Evaluation set**: Consisting of 10 of the remaining alphabets and 4 out of the remaining drawers
- **Test set**: Consisting of the remaining 10 alphabets and 4 drawers

This way, we know that when testing the performance of our network, it has not seen either the alphabet from which a certain image originates, not the artist who drew it, along with its drawing style or calligraphy, making matters worse.

### 3.2 Training Details

#### Hyperparameters
We tried to implemented the authors' setup by setting the following
1. An SGD (Stochastic Gradient Descend) optimizer.
2. A Binary Cross Entropy Loss
3. Exponential Learning rate with $\gamma = 0.99$ and with a fixed momentum of $0.5$ at the start.
4. minibatch size 128
5. Weight and bias initializations as:
    1. For convolution layers weights from $\mathcal{N}(\mu=0, \sigma=10^{-2})$ and biases from $\mathcal{N}(\mu=0.5, \sigma=10^{-2})$
    2. For fully connected layers layers weights from $\mathcal{N}(\mu=0, \sigma=0.2)$ and biases from $\mathcal{N}(\mu=0.5, \sigma=10^{-2})$
6. Maximum of 200 epochs, aditionally stopping training when validation error didnt improve for 20 epochs.

We modified the above to possibly get better results
1. An Adam optimizer
2. Constant learning rate of $3e-4$
3. L2 penalty(`weight_decay`) of $2e-4$ for the whole network, except the prediction layer where it is $1e-3$.

#### Training details 
We trained using 2 resources
1. A machine with *RTX 2070 Max-Q 8GB DDR6 VRAM and 16GB RAM* for smaller datasets and cyclic learning rate schedulers experiments.
2. Google's Colabratory[^colab] for the larger datasets.

Training time in Google Colaboratory was measured per epoch when training on the 30k sample dataset, taking 31 seconds per epoch, and scaling accordingly with the size of the dataset for larger sample sizes.
The results on the validation set were on average achieved with networks that were trained for 10-20 epochs, which is a great success in terms of efficiency in training time.



#### Problems Faced
The size of the datasets with affine transformations was a huge limitation. We could easily train with the 30k and 90k datasets on the local setup, but running with affine transforms was not possible RAM constraints thus we chose to use Colabratory.


## 4. Results
In this section, we tabulate the results and give the verdict on whether we were able to reproduce the results of the paper.

### 4.1 Reproduction of Classification Results

We first try to reproduce the results achieved by the original authors in the verification task. This task consists of generating datasets of different sizes by randomly sampling pairs from the original dataset, and training our siamese network to tell apart same and different inputs.

Similar to the authors, we choose **three different dataset sizes: 30k, 90k and 150k** pairs of images. In each of the datasets the ratio of same to different pairs is equal to 50%.

We also perform **dataset augmentation** by making the images of each dataset undergo a set of affine transformations including rotations, translations, scaling and shearing. We apply these random transformations 8 times to each image, with transformations applied to the images of a same pair being different.

This renders datasets with 9x the size of the originals, since we also use the unmodified images as part of the set. We train our network on these datasets and validate every epoch in order to save the best model yet. If after 5-10 epochs the network shows no improvement in the validation task, we keep that model as the best.

For the validation dataset, we create a dataset of 10k randomly sampled pairs from the validation alphabets and drawers, both of which the network did not see in the training process.

It is important to remark the converging speed in all situations, with the network being able to almost interpolate the training set in just over 10 epochs. The results obtained with the different training set sizes are shown below, as well as the original results achieved by the authors. It is easy to see that adding the affine transformations immediately renders better results than adding extra samples without transformations.



|       Dataset       | Our Accuracy (%) | Author Accuracy (%) |
|:-------------------:| ---------------- | ------------------- |
|         30k         | 90.96            | 90.61               |
|  270k (30k+affine)  | 93.24            | 93.15               |
|         90k         | 91.37            | 91.54               |
|  810k (90k+affine)  | -           | 93.15               |
|        150k         | 91.62            | 91.63               |
| 1.35M (150k+affine) |  -                | 93.42               |

We were not able to run the 810k and 1.35M pair datasets due to lack of RAM in order to perform all of the affine transformations and load all the data at the same time. Our 26GB RAM Colab Notebook was not able to fully load the dataset without crashing. Nevertheless, the accuracy progression with the number of samples suggests that we would see similar numbers to what the authors' reported.

### 4.2 One-Shot Learning

After testing our network on the basic task of the validation set, we want to check its true similarity prediction capabilities by making it undergo a one-shot classification task. For this purpose, we need to again sample random sets of images from the validation set.

The process we follow is different to the random sampling we did for the creation of the normal datasets:

1. We first choose a random character from the training set, and a particular drawer for that character
2. We randomly sample the whole dataset in search for images different to the one we sampled in step 1
3. We sample the same character as in step 1 but drawn by anoter drawer
4. We create pairs of the image sampled in step 1 with all the others sampled in steps 2 and 3

![oneshot_batch_example](https://raw.githubusercontent.com/sharwinbobde/siamese-nn-oneshot-reproduction/gh-pages/images/oneshot_test.png)

> Example of the oneshot task. We compare a reference image to a subset of images of which just one represents the same object but with some changes
> _Taken from: "Siamese Neural Networks for One-shot Image Recognition"_

This way, we create a batch of N pairs of images, in which only 1 of those is made up of the same character, and the other N-1 are not. We then input these batches into the network and calculate from these which pair was the network more confident about, with this being the chosen class of the classification.

![oneshot_batch](https://raw.githubusercontent.com/sharwinbobde/siamese-nn-oneshot-reproduction/gh-pages/images/20-way-oneshot.png)

> We perform 20 way oneshot classification just like in the paper, here is an example of a batch accompanying a certain image. 
> We have to check that the `argmax` of the output vector corresponds to the `argmax` of the targets
> _Taken from: "Siamese Neural Networks for One-shot Image Recognition"_

The accuracy is then 
$$
\frac{1}{N}\sum_{i = 0}^N[argmax(y_i) = argmax(\hat{y_i})], \hat{y_i} = \hat{f}(x_i)
$$
where $x_i$ and $y_i$ are a batch of B pairs of images and a vector of B components respectively as opposed to before during the verification task, and B is the size of the one-shot set size, in our case 20.

Below we show our results in the OneShot task, we used the best network trained on the 270k pairs as it is the one that showed the best results during the verification task. For some extra information we also add the results as reported by the authors with all the extra comparisons they provided. As can be seen, our network performs decently across the board, improving many of the baselines in one-shot recognition.


| Method                                  | Test     |
| --------------------------------------- | -------- |
| Humans                                  | 95.5     |
| Hierarchical Bayesian Program Learning  | 95.2     |
| **Convolutional Siamese Net (Authors)** | **92.0**     |
| Affine model                            | 81.8     |
| **Convolutional Siamese Net (Ours)**    | **75.5** |
| Hierarchical Deep                       | 65.2     |
| Deep Boltzmann Machine                  | 62.0     |
| Siamese Neural Net                      | 58.3     |
| Simple Stroke                           | 35.2     |
| 1-Nearest Neighbor                      | 21.7     |

As can be seen, our network does however lack some accuracy when compared to the authors' results, despite our network and theirs having similar performance in the verification task. There are several possible causes for this difference in performance:

1. The way the random sets of oneshot batches was sampled from the validation dataset might be different to the authors'. They state that they follow a procedure of "within alphabet oneshot classification" that is not very clearly explained in the paper. 
We decided to choose random characters from different alphabets since to our judgement that is more of a real-world oneshot task, but limiting it to the alphabet scope might increase performance
2. In the 20-way oneshot task, only 1/20 or 5% of results have the label "same" compared to the 50% used during training. This change of label distribution could affect the network negatively and prevent it from performing as well in the task
3. The quality of the drawings. As can be seen in the images below, some of the drawers are quite sloppy with the way the draw some characters. By choosing a couple of drawers that do a good job of faithfully reproducing letters we could avoid some cases as the one below and get better accuracy

Nevertheless in many of the cases the network classifies all of the pairs correctly, and most mistakes involve a single image that is falsely classified as same with more confidence than the true image.

![fail](https://raw.githubusercontent.com/sharwinbobde/siamese-nn-oneshot-reproduction/gh-pages/images/fail.png)

![true](https://raw.githubusercontent.com/sharwinbobde/siamese-nn-oneshot-reproduction/gh-pages/images/true.png)

> An example of the network getting confused between multiple characters. The remaining 18 characters of the batch all ended up with confidences lower than 0.15, so there was just 1 false positive out of the 20 images in the batch
> Here the network might focus more on the 2 closed circles of the character that the one at the top has in contrast to the sloppy drawing below, hence giving it a slight 0.01 higher confidence and resulting in a failed oneshot task

## 5. Insights
This section has a bunch of extra experiments that we conducted and interesting insights.

### 5.1 Comparison of Exponential LR vs Other LR Schedulers
We decided to observe the effect of cyclical learning rates on the training in contrast to the author's method. In general, cyclic learning rate schedulers provide better accuracy in fewer iterations without need for extensive hyperparameter tuning for the learning rates [^cyclical-learning-rates]. Cyclical learning rates have documented benefits when encountering the following for training neural networks for image tasks:
1. Saddle points: saddle points have small gradients that slow the learning process. However, increasing the learning rate allows more rapid traversal of saddle point plateaus [^adaptive-learning-rates].
2. Noise: generalization benefit of CLR for training on small batches (SGD) comes from the varying noise level, rather than just from cycling the learning rate[^minima-sgd].

We experimented with two cyclic learning rate schedulers against the author's: an exponential decay with $\gamma = 0.99$, decayed every epoch. 

```python
optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)
```

Firstly, we used a basic triangular cycle with no amplitude scaling.

```python
optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=3e-4, cycle_momentum=False, mode="triangular2", step_size_up=500)
```

Secondly, we used a triangular cycle which scales the maximum learning rate by half every cycle.

```python
optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=3e-4, cycle_momentum=False, mode="triangular", step_size_up=500)
```

We conducted the experiment on the 30k dataset for 25 epochs. We only recorded for 25 epochs because we only wanted to see the training loss and validation accuracy approach towards convergence, we knew from previous experiments that we got good performance at 20 epochs. It should be noted that the `ExponentialLR` is decayed every epoch, whereas `triangular` and `triangular2` are changed after processing every batch. The Learning rates for the experiments have been plotted below along with the training loss.
![](https://raw.githubusercontent.com/sharwinbobde/siamese-nn-oneshot-reproduction/gh-pages/images/lr-schedulars-lr.png "Learning Rates")

![image alt](https://raw.githubusercontent.com/sharwinbobde/siamese-nn-oneshot-reproduction/gh-pages/images/lr-schedulars-train_loss.png "Tarining Loss")
We can clearly see that cyclic learning rates result in a lower training loss faster as compared to the exponential decay.

![image alt](https://raw.githubusercontent.com/sharwinbobde/siamese-nn-oneshot-reproduction/gh-pages/images/lr-schedulars-val_acc.png )
We observed that the cyclic learning rates helped the model achieve convergence faster (from the 12th epoch) as compared to the exponential learning rate (18th epoch).
We can also see that `triangular2` does comparatively poorly on the validation accuracy which might be because it decayed to too small a value quicker. As the `triangular` loss does not reduce the maximum amplitude it provides good training here without having to adjust any other parameters. We also see that `triangular` performs better than `ExponentialLR` for the validation set for the lower values of learning rate. This cyclic nature of the validation/test accuracy can also be seen in [^cyclical-learning-rates]. The test accuracy for the model trained on the three schedules are as follows:

| LR Schedule                         | Test Loss |
| ----------------------------------- | --------- |
| Exponential decay [^one-shot-paper] | 90.43     |
| Triangular                          | **91.06** |
| Triangular2                         | 87.90     |


### 5.2 Feature Maps

We were inspired by [this excellent Distil Pub article](https://distill.pub/2017/feature-visualization/) by Chris Olah and colleagues to visualize the feature maps for our convolutional network to really **see** what it was learning. Code for this visualization and one in the next section is available [here](https://github.com/sharwinbobde/siamese-nn-oneshot-reproduction/tree/feature-maps/notebooks/experiments/activation_maps.ipynb)

![feature maps 270k](https://raw.githubusercontent.com/sharwinbobde/siamese-nn-oneshot-reproduction/gh-pages/images/feature_maps.png)

In above figure, we extract all 64 filters from the first convolutional layer of our top performing network (trained on the 270k dataset). We can see that the model has learned edges at various angles and curves, dots and other complex shapes. These act as feature extractors and help the network distinguish between alphabets.

Below, we see the filters, again from the first layer but from the model trained on 30k images (no affine transforms). While there is some co-adaptation between filters, it is easy to see that the filters learned by the top performing model are more well defined, complex. Many of the filters we see below have similar shapes while most filters from above are unique. This could be the reason for the difference in classification performance.

![feature maps 30k](https://raw.githubusercontent.com/sharwinbobde/siamese-nn-oneshot-reproduction/gh-pages/images/feature_map_30k.png)


### 5.3 Activation Maps

To visualize the increasing receptive field and the ability of CNNs to learn information hierarchically, we input this image:

![activation input](https://raw.githubusercontent.com/sharwinbobde/siamese-nn-oneshot-reproduction/gh-pages/images/activation_inp.png)

and then we observe the activations in the model:

![activation conv1](https://raw.githubusercontent.com/sharwinbobde/siamese-nn-oneshot-reproduction/gh-pages/images/activation_conv1.png)
![activation conv2](https://raw.githubusercontent.com/sharwinbobde/siamese-nn-oneshot-reproduction/gh-pages/images/activation_conv2.png)
![activation conv3](https://raw.githubusercontent.com/sharwinbobde/siamese-nn-oneshot-reproduction/gh-pages/images/activation_conv3.png)
![activation conv4](https://raw.githubusercontent.com/sharwinbobde/siamese-nn-oneshot-reproduction/gh-pages/images/activation_conv4.png)
> Top to Bottom: Visualized Activations of 1st to 4th Convolutional Layers
---

The first layer clearly focuses more on the edges of the alphabet, some focus on the horizontal edges, some vertical.

The second layer focuses on strokes since the full letter is activated.

The third layer activates in specific complex parts of the letter such as loops, dots and joints.

It is quite hard to interpret what the fourth layer is trying to do, but we can see that it has much denser information which has been encoded in an almost bitmap fashion.

We can also see a gradual decrease in size of the images from 1st to 4th layer. This is due to increasing receptive field. This is what allows the network to learn more complex ideas.

## References
[^one-shot-paper]: Siamese Neural Networks for One-Shot Image Recognition, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
[^cyclical-learning-rates]: Cyclical Learning Rates for Training Neural Networks, available at:https://arxiv.org/abs/1506.01186.
[^adaptive-learning-rates]: Equilibrated adaptive learning rates for non-convex optimization, available at: https://arxiv.org/abs/1502.04390
[^minima-sgd]: Three Factors Influencing Minima in SGD, available at: https://arxiv.org/abs/1711.04623
[^cs231n]: CS231n Convolutional Neural Networks for Visual Recognition, https://cs231n.github.io/convolutional-networks
[^colah]: Convolutional Neural Networks, http://colah.github.io/posts/tags/convolutional_neural_networks.html
[^github]: siamese-nn-oneshot-reproduction, https://github.com/sharwinbobde/siamese-nn-oneshot-reproduction
[^omniglot]: Omniglot Dataset, https://cims.nyu.edu/~brenden/lab-website/resources.html
[^colab]: Google Colaboratory colab.research.google.com/
