`Material Came from the University of Texas at Austin - Prof. Philipp Krähenbühl`

# ADV-DL
Advanced Deep Learning


## [0] Basic
### [0-1] Vector Operation

* Inner product: return `scalar value`
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/160245695-022d7690-653c-4b2d-9a15-a6351fa4611e.jpg">

* Outer product: return vector
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/160245699-d04922d8-7b15-4efc-8c35-d75c5308ae69.jpg">

* Frobenius norm: the summation of square of all matrix elements

  * `Why Frobenius norm is important in deep learning?`

    * can measure average magnitude of weights (parameters) of neural network.

    > <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/160245683-91e9b3bf-c737-4690-a724-314e52ffc3dd.jpg">


* Gradient of scalar function
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/160245692-4091c644-6ffe-4b05-af62-fc2c37483679.jpg">

* Jacobian: Derivative of `vector g(x)` by `vector x`
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/160245702-e664351d-44bc-4be4-8dc7-5fe154cf7a68.jpg">

      lay gradient vector horizontally
      
      
### [0-2] Probability and likelihood

> likelihood refers to the past
> 
> probability refers to the future


### [0-3] Tensor

> Tensor is used in all the parts in deep learning.
> 
> (1) Image: H X W X 3 tensor
> 
> (2) N-char documents: N tensor
> 
> (3) t second of audio: T X M tensor


## `Regression and Classification`

### `Difference between linear regression and classificaiton`

----
```diff
+ Key of Regression: Regression predicts real value(smooth range)
+ Key of Classification: Classification predicts integer/class(specific kinds)

- Be careful: 
```
----

### `Linear regression / logistic regression / multi-class classification`

> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/160286720-7b4eea2c-58b2-4c82-a052-bfc3b5bb06c3.jpg">

> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/160286721-33e26992-3334-492c-b604-c7b21cd170d1.jpg">



## `Backpropagation`

----
```diff
+ Purpose: to update the parameters with the optimal values by analyzing and differentiating the loss function.

+ Benefits 1: Different from numerical differentiation, backpropagation saves the differentiation result matrix` from the the highest layer.
+ Benefits 2: Then goes down `with saved matrix`. Thus, it is robust in memory utility and the amount of operation.

! Simple Algorithm 1: Take the derivative (gradient) of the loss with respect to each parameter
! Simple Algorithm 2: Shift parameters in order to minimize loss

```

> <img width="950" alt="IMG" src="https://user-images.githubusercontent.com/73331241/160856967-d7de2f36-f8bb-491b-b9f3-4b3c76186494.jpeg">

----

## `Neural Network Modeling`

### `[NN] Classifier`

----
```diff
+ Non-linearities: ReLU(x) = max(x,0)

- Problem of linear classifier: we cannot solve the problem which `requires curve` to classify two groups

- Do layer-ed linear layers help?: No. That is just combination of linear layers still linear.
```
----

<!--

### [2-2] Positive regression

> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166146143-4d7fb0d0-9d4b-4300-a373-dde7fdfce2b0.png">
> 
> > option1(ReLU)
> > > just want to `predict a value` that is positive, and you `don't care about to train network` we can use ReLU function.
>
> > option2(Soft ReLU)
> > > if we `want to train` a network with positive prediction, we should always use softrelu(softplus) function

-->

#### `[Classifier] Binary Classification`

----
```diff
- Option 1: Step function

! Prob. of Step: Impossible to train because output activation function shape. Thus, the derivative in everywhere is 0.

+ Option 2: Sigmoid

! Prob. of Sigmoid: Differentiable, but it has a zero gradient in big range.
```
----

#### `[Classifier] General Classification`

----
```diff
- Option 1: argmax

! Prob. of argmax: hard to train because we `cannot compute` gradient of argmax function.

+ Option 2: softmax (Possible to train and actively used in multi-classification)
```
----

<!--
> `Tip in output representations`
> 
>  > `do not` `add` other `output transformation` into model's output. Always output raw values
>  > 
>  > reason: many output transformation makes `harder to differentiate`. If bigger output transformation, numericably unstable.

-->

### `[NN] Loss`

> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166390923-c1c541ce-8c65-4c3b-a8f8-8b18b289ecee.png">

#### `[Loss] Loss in Regression`

----

L1 Loss function  	      |      L2 Loss function
:---------------: | :-------------:
Least Absolute Deviations.  | Least Square Errors


```diff
! Difference L1 vs L2: They handle outliers differently. L2 is much more sensitive to outliers because the differences are squared, whilst L1 is the absolute difference and is therefore not as sensitive
```

* `L1 vs L2 loss, which is better?`

  * The choice between L1 and L2 comes down to `how much you want to punish outliers` in your predictions. If minimising large outliers is important for your model then L2 is best as this will highlight them more due to the squaring, however if occasional `large outliers are not an issue` then L1 may be best.

> <img width="190" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166210192-74bdc977-674a-4549-aedf-33fd3c9ae40d.png">

----

<!--
> Additional information: in deep learning, it came out that L1 loss and L2 loss doesn't make a huge difference. (`We need to perceive that L2 loss is different from L2 norm.`)
-->

#### `[Loss] Loss in Classification`
 
----
 
 Negative Log-likelihood  	      |      Cross-Entropy Loss
:---------------: | :-------------:
A loss function that represents how much the predicted probabilities deviate from the true ones. It is used in binary cases.  | A generalized form of the log loss, which is used for multi-class classification problems.

```diff
! Same situation: In one-hot encoding, cross entropy and -log likelihood is exactly the same.
```

* PyTorch - Negative Log-likelihood: BCEWithLogitsLoss (log + sigmoid)
* PyTorch - Cross-Entropy: CrossEntropyLoss (log + softmax)
----

<!--

> `[Issue]`
> > * From the classification process, because of sigmoid and softmax functions attributes, we can get `0 from the activation function`. Thus, once we put this value into log, there is no value is defined.
>
> `[Solution]`
> > * To solve it, modern deep learning package `combine` log and sigmoid or `combine` log and softmax

-->



### `[NN] Gradient Descent`

#### [GD-Stochastic Gradient Descent]

<img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166613395-fad93739-6997-4582-9baf-0e0b625ca367.png">

----

```diff
+ Attribute: highly oscillate and has large variance.
! Same situation: In one-hot encoding, cross entropy and -log likelihood is exactly the same.
```
Why SGD works etter than Standard Gradient Descent in `non-convex function`
:---------------: 
Reason1: oscillation of SGD helps to explore the function  
Reason2: faster. We don't have to wait to loop over entire training set to compute gradient  

<!--
> Compute the gradient as we move, so always we take a step and compute the gradient and take a step ..
-->

----

#### [SGD] Variance of SGD
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166393020-04314dc3-0935-4eef-a871-80c87f9402b2.png">

* Measure the agreement between different data samples about gradient, we can compute the variance of gradient estimates.

* difference between the gradient of one single data point and the gradient of entire objects' loss
 
`Reason of introduction(biggest issue): "not" all training set give a "good direction" to update`

### [NN] Mini-batches

> Not just get one data element(x, y), it `takes bunch`(so-called batch or mini-batch) of data elements.

----
```diff
+ Key: Mini-batches can estimate thr gradient robustly by `reducing the variance` of SGD.

! Param Update: we compute the `average gradient` over the mini-batch and `update the parameters` rather than considering single data element.

- Be careful: We should use MB as much as we can. Set the batch size as big as possible GPU can.
```
----
<!--

> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166613762-e105009e-af17-4cec-8dd0-31a8909ed68c.png">

> ##### `Difference in variance`
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166613961-a3ecc639-17b5-44ee-9f4c-ad83b8181703.png">

-->

### [NN] Momentum

----
```diff
+ Key: Momentum(v) is that `functionalized gradient` by adding previous Momentum(v).

+ Benefit 1: consider past training situation
+ Benefit 2: Variance reduction of Momentum (since update term(`blue`) is getting smaller.)

- Be careful: Use mini-batched and Momentum every time as default.
```
----

> <img width="190" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166614563-f4a678d3-7cc3-4534-937c-d713c02809c9.png">

> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166614982-b3a36031-062b-4487-bbae-9e821a27481e.png">

<!--

### [2-8] Training code

`optimizer.zero_grad()` is mandatory before we backward the loss.
When we use Fully connected model which only has linear layer, it can be overfitted massively.

```python
# Training code
# TODO: initialize some variables / hyper-parameters

# TODO: Create the tensorboard logger

# TODO: Create network

# TODO: Create optimizer (optimizer decides how to optimize the model using the deriavtive values of loss function.)

# TODO: Create loss function

# TODO: Start training
    
    # TODO Shuffle training data
    
        # TODO: Construct batch
        
        # TODO: Compute output
        
        # TODO: Compute the loss
        
        # TODO: Logging
        
        # TODO: Compute the gradient
        
        # TODO: Step
        
    # TODO: Log training accuracy
    
    # TODO: Evaluate on validation set
    
    # TODO: Log validation accuracy
```
-->

### [NN] Layers

> Largest computational unit that remains unchanged throughout different architectures

----
```diff
+ How to count layer: Count from the first hidden layer to the output layer. (Below example: 4 layers)

- Be careful: Normal output of activation function: `activation` / Activation function goes in to output layer: `feature`
```

> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166881431-3d53270c-10d2-45f3-9396-114fe5c1b6b3.png">

----





### [NN] Non-linearities (activation function)

> > Allow a deep network to model arbitrary differentiable functions
> 
> ##### `Zoo of activation functions`
> > <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166887675-166bdc7b-9050-44ea-8b0d-5331febd274c.png">
> 
> ##### `Traditional activation functions-Sigmoid`
> > <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166888042-c40c82ba-4c7d-423e-9c85-940196cef24a.png">
> >
> > ###### `Issue` of sigmoid: derivative is quite flat as we can see the above orange line
> > 
> > If we want to use sigmoid, we should initialize the value closed to zero.
> > 
> * `tanh` is just the scaled activation function of sigmoid
> 
> ##### `Traditional activation functions-ReLU`
> > <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166888623-046afc6e-66ab-4476-b4c7-51100fdb2193.png">
> > 
> > All positive value have derivative one, and all negative value have derivative zero.
> 
> > ###### `Issue` of ReLU: Dead ReLU
> >  > If all the input values are negative, we can never get derivative to update the model.
> > 
> > ###### `How to Prevent dead ReLUs?`
> >  > <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166889718-b801f5ac-b3af-4a73-bde2-30880acdbb71.png">
> >  > 
> >  > (1) Initialize Network carefully (input data can be distributed near zero or bigger than that.)
> >  > 
> >  > (2) Decrease the learning rate
> >  > 
> >  > (3) Leaky ReLU
> >  > > <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166890041-8b7fc686-c2ff-4aa5-92c2-685daf1e2e4c.png">
> >  > > 
> >  > > PReLU: Parameterized ReLU
> >  > 
> >  > (4) ELU
> >  > > <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166892425-2fe1dae2-f2be-4671-920b-8823e3eee97d.png">
> >  > > 
> >  > > `Attribute`: Sometime, ELU is slower than ReLU or Leaky ReLU. Because of gradient calculation.
>
> ##### `Activation functions-Maxout`
> > <img width="250" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166892834-5b956c7a-f1b8-4d33-8f5f-2abe6ac7969f.png">
> 
> ##### `Which activation to choose?`
> > <img width="250" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166893493-c263b50b-547f-4e68-b781-ae0c66ba7c0d.png">
> >
> > Try to use `ReLU first`. If you have a problem with Dead ReLU, we can try `Leaky ReLU`, and if Leaky ReLU works, try to use `PReLU`.
> > 
> > sigmoid and tanh are usually used in Natural language process or RNN.

### [2-11] Hyper-parameters
> Any parameters set by hand and not learned by SGD.

### [2-12] `Loss function selection`
> <img width="250" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166894368-990121d0-6d63-4f55-a4bc-b67ca5316b5e.png">

* MSELoss is also referred to as L2Loss
 
* Momentum is 0.9 in most practice.

----

## [3] Convolutional Neural Network

### [CNN - The problem of FCL for image processing]
> <img width="250" alt="IMG" src="https://user-images.githubusercontent.com/73331241/167284388-c8f105de-0053-4ac4-8e7f-f48431b631b4.png">
> 
> > <img width="250" alt="IMG" src="https://user-images.githubusercontent.com/73331241/167284471-70588bfe-1cf6-4b58-a757-a318c1750525.png">
> > 
> > (Bad reason 1) Fully connected networks do not preserve any pattern in the image. Even though image shifts or rotates little bit, the input to the deep learning changes drastically.
> > 
> > (Bad reason 2) Lots of parameters are required in training process.

### [CNN - Convolutions]
> `Sliding` linear transformation
> > <img width="250" alt="IMG" src="https://user-images.githubusercontent.com/73331241/167284715-803aa68c-becf-47fe-addb-41b365145bd4.png">
> 
> Why `1X1 convolution kernel` is necessary?
> > To reduce the size of feature map to do model compression. (maintaining W and H of the input size.)


### [CNN - Convolution-parameters: Padding, Stride, and Grouping]
* Padding: (1) to keep the image resolution(avoid kernel size is getting too smaller), (2) to do more convolution operation from the edge elements

> How to avoid the size of feature map is shrinking by adjusting padding size?
> 
> `padding = (kernel_size-1)//2`

* Stride: make convolution operation faster.
> Unclear output size with striding: can be floating number. We round down it(We don't consider the operation that is done from the outside of the input data).
> 
> > <img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/167324735-d7d8914a-fa71-445a-97a1-a68edd358354.png">


* Grouping: group certain number of input channel and the same number of output channel together. We split input channels into g groups. Then, we only connect input channels with output channels of the same group. (We do not have connection that cross groups.)

> It allows us reducing parameters and computation by factor g.
>
> <img width="250" alt="IMG" src="https://user-images.githubusercontent.com/73331241/167325279-e376fefe-e0ff-4bd7-b692-b45592e4448c.png">

* Depthwise convolution: the number of groups are equal to number of channel

> Often and only used from small and efficient network
>
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/167325700-6e4ef6b7-b749-4d21-9708-96dcd4115119.png">

### [CNN - Pooling]
> presereve the channel

##### (1) Average Pooling: `Linear layer`
> In each channel, it computes the mean of all possible x, y location inside of the kernel.
>
> Reason of using: To reduce the dimensionality of feature map
>
> `Older networks`: Put the average pooling layer inside a network. Usually, Conv Layer has just 1 stride value. Thus, Pooling layer was used to reduce the dimensionality.
>
> If we do Avg Pool before we do Conv Layer, this's combining `two linear operations(Pooling and Conv Operation)`. => Inefficient
> 
> `Why we rarely put` the average pooling layer inside of the network?
> > Conv Layer can do exactly the same job as Avg Pooling layer more efficiently.
>
> Usage of Avg Pooling in Modern networks
> > Use Global average pooling layer at the very end of the network.
> 
> `Global average pooling`: produce `one single spatial output` that has same number of channels as the input channels of average pooling.
> 
> <img width="250" alt="IMG" src="https://user-images.githubusercontent.com/73331241/167761686-1454ad15-3194-4c17-b40a-036804c93d27.png">

> > Then we can use Global average pooling output from Linear layer to classify
> > Average Pooling layer is not used at the middle position of the network.

##### (2) Max Pooling: `Non-Linear layer`
> <img width="250" alt="IMG" src="https://user-images.githubusercontent.com/73331241/167761841-78be228d-9efc-4974-887d-bbdcc5577290.png">

> `Purpose` of max pooling: Down-sampling in modern neural network. (We can use Max Pooling `inside` of the network.)
> 
> [Modern modeling]
> > Conv layer stride with 1 unit, then network reduces the sample size from Max Pooling Layer.

### [CNN - Modeling]

```python
class ConvNet2(torch.nn.Module):
    def __init__(self, layers=[], n_input_channels=3, kernel_size=3, stride=2):
        super().__init__()
        L = []
        c = n_input_channels
        for l in layers:
            L.append(torch.nn.Conv2d(c, l, kernel_size, padding=(kernel_size-1)//2))
            L.append(torch.nn.ReLU())
            L.append(torch.nn.MaxPool2d(3, padding=1, stride=stride))
            # L.append(torch.nn.MaxPool2d(2*stride-1, padding=stride-1, stride=stride))
            # "2*stride-1" kernel size guarantees whatever stride we have, we always pool over all input in our activation map.
            c = l
        L.append(torch.nn.Conv2d(c, 1, kernel_size=1))
        self.layers = torch.nn.Sequential(*L)
    
    def forward(self, x):
        return self.layers(x).mean([1,2,3])

net3 = ConvNet2([32,64,128])
```

### [CNN - Receptive fields]
> The receptive field is defined as the region in the `input space` that a `particular CNN’s feature is looking at` (i.e. be affected by).
> 
> However, not all pixels in a receptive field is equally important to its corresponding CNN’s feature.
> Within a receptive field, the closer a pixel to the `center of the field`, the more it contributes to the calculation of the output feature.
> 
The pixel at the center can influence more than the pixel at the edge.
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/167976356-b9a49047-1348-47c6-b601-5948bb2575a7.png">

> Therefore, pixels at the center of receptive field have much higher weight on the output decision than the side one.


### [CNN - `Model Design Principles`]
##### [1] Use striding(downsample activation map), increase channel. It helps balance some of the computation that held inside of deep neural network.
* Striding by 2 and increase the number of channel by the factor of 2, this will make sure all the convolusions have `roughly the same number of operation` throughout the network.
>
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/167977239-582a241d-5507-4a85-9ae4-86994142a04c.png">

##### [2] Keep kernels small
  * 3 X 3 Kernels almost everywhere
   
  * Reason: (1) Small kernel size with deep network has a benefit in the number of parameters., (2) We can put more non-linearity between 3X3 Kernel
   
  * `Exception`: First layer up to 7X7 (Reason of exception: First layers with 7X7 kernel size allows us to `capture more interesting patterns` from the high resolution input image itself.)

> <img width="150" alt="IMG" src="https://user-images.githubusercontent.com/73331241/167978054-fd417e28-3083-492c-8d11-5b5f38557c9f.png">

##### [3] Use repeat patterns
 * First layer or two are special. So we can set paramters sophisticatedly.
 
 * However, all other layers usually follow a fixed pattern, because setting all the hyperparameters for all layers are so overwhelming.

##### [4] Keep all network convolutional] 
> Use Avg Pooling layer at the end.

[Benefits]
 * Fewer parameters
 
 * Better training signal
 
 * Ensemble voting effect for testing
 
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/167979391-708665f5-cfd0-4e4c-b182-ebf4c1d6243a.png">


### [CNN - `What do networks learn?`]
(1) `The filter at the First layer`:
local features: edges, texture patterns, color, 

(2) `The filter at the Middle layer`:
combined local features

(3) `The filter at the Final Layer`:
color features of eyes or nose or tongue or head.



### [CNN - `Segmentation`]
> Label all the pixel as `Dog(1)` or `Not Dog(0)`

> ##### [1] `Issue`
> > we need to produce an output that `has high resolution` as high as input data


> ##### [2] `Solution` to increase receptive field
> > (1) Only use Convolutional Layer
> > 
> > (2) Use ReLU
> >
> > (3-1) Non-striding and use big kernel size / (3-2)Striding by the factor of 2

> ##### `Solution (3-1): Non-striding and use big kernel size`
> > <img width="190" alt="IMG" src="https://user-images.githubusercontent.com/73331241/168461393-1fd9859b-cf72-4e76-9cfe-20cd96f9d715.png">
> > 
> > Attribute: `Inefficient`. It can only increase the receptive field linearly by enlarging the kernel size.

> ##### `Solution 2: Striding by the factor of 2`
> > <img width="190" alt="IMG" src="https://user-images.githubusercontent.com/73331241/168454395-1fa9c6c0-aefa-4248-b822-78dcb5942449.png">
> > 
> > Efficient. We can see the receptive field get much larger.
> > ##### But, the `main issue` is `striding does lose resolution`.

### [CNN - Segmentation - Dilation]
> Way to virtually `inflate a kernel` `without` increasing number of parameters and computations.

> ##### `Dilation process`
> > Take a kernel and then `spread out` the original kernel. Then, `put the zero at the empty` elements in the enlarged kernel.
> > 
> > <img width="190" alt="IMG" src="https://user-images.githubusercontent.com/73331241/168461516-de307060-b4dc-4a66-a54b-e712d1d8e186.png">

> ##### `Dilation benefits`
> > Dilation `can compute every single element` without losing resolution.
> > 
> <img width="190" alt="IMG" src="https://user-images.githubusercontent.com/73331241/168461597-36312201-f8f2-42e7-8334-7e78e7c43a21.png">
> 
> ##### [Process]
> > First CNN layer is not dilated. Second CNN layer is dilated by two. Third CNN layer is dilated by four.
> 
> ##### [Benefit]
> > It can have `larger receptive field`, but `compute every single element`. Then, can use CNN to produce high resolution output with a larger receptive field with an input image.


### [CNN - Segmentation - Up-convolution]
> We `dilate input data` and run `regular convolution` to the input data
> 
> <img width="190" alt="IMG" src="https://user-images.githubusercontent.com/73331241/168461912-db59eec7-dea6-473c-9bda-133c4fe78580.png">
>
> ##### `Advice in up-convolution`
> > When we use up-convolution inside of our network, we have to `be careful` how to `set up the parameters and stride` value because `strided convolution round down`.
>
> ##### `How to fix for rounding down problems of strided convolution?`
> > (1-not robust) avoid round operation through mathematical calculation of parameter.(Not good because it is impossible )
> >
> > (2-robust) Much better way detect whatever rounding happen, and then you can add a parameter call output_padding to our up-convolution.
>
> <img width="190" alt="IMG" src="https://user-images.githubusercontent.com/73331241/168462449-860e007e-585d-46fd-97b7-7e5afb896782.png">

> ##### `Where can we use Up-convolution in general?`
> > Up-convlutions are `after used convolutions`.
>
> <img width="190" alt="IMG" src="https://user-images.githubusercontent.com/73331241/168462681-4252336e-9e65-41fa-844c-d1cfba851e77.png">
> 
> <img width="190" alt="IMG" src="https://user-images.githubusercontent.com/73331241/168463025-37ed4829-343d-4445-9fa1-cacee0286054.png">

> ##### `Skip connections`: Provides lower-level high-resolution features to output
> > <img width="190" alt="IMG" src="https://user-images.githubusercontent.com/73331241/168464957-b0482578-0756-4b9d-b1f0-a62996982eec.png">


The many names of up-convolution (1) Transpose convolution: When we implement up-convolution we multiply with the transpose of the matrix, (2) Fractionally strided convolutions

----

## [4] Make a Network Better
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/168528496-b87a45bd-2440-46ab-a6e9-c485980bf818.png">
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/168537087-a5a13c12-fd60-47b5-b2f9-75fb0283158a.png">

> `Look at your data`: set what task `we need to solve`. Recognize `what is the hard part` of the task and what is the `easy part` of the task. How much `high level information` we need? How much `low level information` we need? Can we `borrow some representation` from somewhere else? Or should we need to design the `network from the scratch`?

`[Cycle]`
`Evaluate your model on unseen data`: at this stage we are about 80 percent of `coding`. In terms of making our model works, just 10 percent done.
From this process, we tune the model over and over again to perform better.

(1) [Looking at the data] look at `how well does this model` perform on our data?(Where does this make mistakes? Where does it work?)

(2) [Design and train your model] Modify property of `learning algorithm, structure of our module, loss function, labels` and so on. (No need to change code that much, just modify the model to improve.)

(3) Evaluate


Once we are happy with the performance of our model, we go out and test our model.

### [Make Better - (1) Looking at the data]
(1) look at random image
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/168530977-f73a7b0b-b08e-44b3-82ac-b9341d8862a8.png">

(2) Smallest / largest file size
Smallest file size - best way to check the availability of the image.
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/168531201-dba9f555-7a53-42b4-9775-e29c9d3237d0.png">

Largest file size- best way to find the weird but special image
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/168531172-140af33f-09fc-46db-9bb0-4d637c840bb1.png">


(3) Try to solve the `task manually`
> Record `our response` with label, it would help us to recognize the `error in labeling or the difficulty of the data` to recognize.


### [Make Better - (1) Looking at the data - dataset]
> (1) Training set (60-80% of data) - Learn model prameters
> 
> (2) Validation set (10-20% of data) - Learn hyper-parameters
> 
> (3) Test set (10-20% of data) - Measure generalization performance. Used exactly `once`.



### [Make Better - (2) Network weight initialization]
What do we set the initial parameters(weight) to?

[Idea 1. All zero]: Never use
0 weights make all the output of activation as zero. Additionally, the gradients with respect to all weights in our network will be zero.

[Idea 2. Constant]: Never use
(1) all of the column in the first weight matrix will have the same value
(2) does not break symmetries

[Idea 3. Random initialization]: the best
> <img width="95" alt="IMG" src="https://user-images.githubusercontent.com/73331241/168709391-600922c6-ed2c-4f51-a693-2c09f10a9dc6.png">


### `IMPORTANT BACKGROUND`

* `Initialize weights` through: `Normal distribution` or `Uniform distrubution`

* How to `set mean and standard deviation of Normal distribution or Uniform distrubution`: `Xavier and Kaiming initialization`

* `For simplicity mean and bias are set by 0.` So we need to care about `standard deviation`

> > Since `magnitude of standard deviation` is `directly related` to the `magnitude of weights`, it is important to select the proper value. (It heavily affect on gradient decent training process, so single weight can impact on other layers.)

### [Make Better - Random weight initialization - `Xavier and Kaiming initialization`]

* Strategy to set `standard variance` of `Normal or Uniform initialization`
  * Set all activations are of similar scale and set all gradients are of similar scale

* How do we compute the scale of activation without knowing what is the inputs without seeing the data just by initializing the weights?
> Random matrix multiplication

* Random matrix multiplication
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/168965741-994c860b-c122-4fde-a799-34731c73bc40.png">
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/168966251-52f2bf69-9661-4f47-bd0e-d3edebeeae9a.png">
> 
> `vector a` is drawn from normal distribution with a certain mean and certain standard deviation.

> We can multiply vector a with an arbitrary vector(input vector), then the output will follow the normal distribution because of vector a. (The only thing we need is that the magnitude of input not the actual distribution.)

* Random matrix multiplication in Linear Layer (z_i)
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/168967425-fb863838-ab67-404c-96b4-5ce631993067.png">

> when we initialize the linear layer with a random normal distribution, we can guarantee that the output follows random normal distribution


* Random matrix multiplication in Activation Layer (z_i+1 )
> Since half of the output from ReLU is zero, we can see 1/2 from the expection.

> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/168968986-8c64088d-2569-4f94-b723-0026f232e30f.png">

> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/168971915-2d5fcb4e-7ad3-48f7-a7a2-2eab475a3dd7.png">

By using it, we can know how it is large certain activation will be at the any linear layer
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/168972285-469be201-6a76-4332-aeb5-3faf54439f04.png">

The above formula is almost same from the backpropagation as well. (just the propagation's direction is different)
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/168972503-5f970e5c-306a-4d10-b251-53f6101465ae.png">

> Try to keep the magnitude of activation the magnitude of gradient as constant as possible to learn with same rate.


* Xavier initialization - default in PyTorch
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/168974920-55e3ea94-63ba-491b-b243-5f1f928c67b3.png">

* Kaiming initialization
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/168975233-c574735b-3657-49c5-8013-d8ba4c9011cd.png">


* Initialization in practice
  * Xavier (default) is often good enough
  
  * Initialize last layer to zero.


### [Make Better - `Optimization`]

`Goals`

> (1) Convergence speed, (2) Training accuracy, (3) Generalization performance (getting a good accuracy on our validation set)


### [Make Better - `Input data normalization`]
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/169199374-0315e168-4e91-4017-8fb8-5322b0e4633a.png">

> Reason of Input data normalization
> > Depends on the input data, the training performance at the first layer

> `How to do` Input data normalization
> > Compute the mean and standard deviation value for entire `green channel`, `red channel`, and `blue channel`


> Possible problem with inappropriate input
> > Big scale of input: Big scale input can update hugely, but small scale will not affect that much. It would be followed by unstable training.
> > 
> > Postiive value of input: Update direction has a limitation. Could be updated in one direction(+ or -).


Exploding gradients
> Weight of `one layer` grows (Causes bad feedback loop)
> 
> > `Gradient of all` other layers grow
> > 
> > `Weights of other` layers grow

> Not a bit issue for most networks

> An issue for recurrent networks, and networks that share parameters across layers
(because we apply same operation over and over again)
>
> The sign of exploding gradients
> > <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/169633181-ce6040f2-8c89-4b7b-bba7-3e071dae8f2f.png">


Vanishing gradients
> Weight of `one layer` shrink (Causes no progress or very very slow convergence)
> 
> > `Gradient of all` other layers shrink
> > 
> > `Weights of other` layers stay constant

> Big issue for larger networks

> Issue for recurrent networks and weights tied across layers


`How to detect` vanishing gradients
> Plot loss
> 
> Plot weight and gradient magitude per layer
> 
> > <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/169432701-0d4d858e-da24-43ca-a9fa-7b555e8bb0a5.png">


If any layer's magintude of gradient is smaller than the magintude of weight, then we can notice the vanishing gradient at that layer


### [Make Better - `Normalization`]

How to prevent vanishing (or exploding) gradients?

inserting layers either before or after the Conv layers that scaled up(for vanishing grad) or scaled down(for exploding grad) the activation

[Additional handler: 1. Activation function, 2. Weight initialization, 3. Network architecture]

> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/169929179-3822e75a-2a7d-474e-996c-d058b6768561.png">


#### [Normalization1 - `Batch Normalization`]

> Make activations zero mean and unit variance using entire batch of data

> <img width="150" alt="IMG" src="https://user-images.githubusercontent.com/73331241/169929919-0c606a46-cc5b-4f21-9ca5-b8f8011118e7.png">

How to do `Batch Normalization`: focus on the same channel of `all batch`
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/169930656-051a45c0-2834-4469-b413-8e9888dac604.png">

> (1) Calculate `channel-wise` mean and standard deviation
> 
> > <img width="150" alt="IMG" src="https://user-images.githubusercontent.com/73331241/169930426-2b64bc0e-fc1f-420d-b88f-8c0192873aa6.png">
>
> (2) Normalize activations using the value from the above
> 
> > <img width="150" alt="IMG" src="https://user-images.githubusercontent.com/73331241/169930441-0b1d6891-d41d-48dc-a70b-eca64c0f692f.png">


What does batch normalization do?
* The good:
  * Regularizes the network
  * Handles badly scaled weights

* The bad:
  * Mixes gradient information between samples

How to avoid Mixes gradient information?
> `Use large batch sizes`. Because it would have `more stable` mean and standard deviation estimates.


How to use batch normalization at `test time`?

> Once we are done training, we can compute mean and standard deviation on training set using running average with all training set.

> Most deep learning framework updates running mean and running standard deviation as we trained.


#### [Normalization2 - `Layer Normalization`]

> (1) Different from BN, it does `not use summary statistics` after training because it focuses on one image to normalize. 
> 
> (2) Works well for `sequence models`

What LN does?
> Make activations zero mean and unit variance without collecting statistics across batches

> <img width="100" alt="IMG" src="https://user-images.githubusercontent.com/73331241/173475903-42f96351-a71c-4b60-bdde-3d6261212a58.png"><img width="250" alt="IMG" src="https://user-images.githubusercontent.com/73331241/173475909-3a1da3dd-f616-4af4-9755-8d5d66e38f61.png">

#### [Normalization2 - `Instance Normalization`]
> (1) Compute `same operation with BN`, but not thoughout the entire batch, just about the `individual image`.
> 
> (2) Works well for `graphics applications`, `Not used` much in `recognition` because of small sample size, we can have unstable statistics.
>
> Thus, we can use Instance Normalization in the matter less than recognition

> <img width="300" alt="IMG" src="https://user-images.githubusercontent.com/73331241/173476800-ae3f27ad-f5f9-441a-a2a6-211e95291842.png">


#### [Normalization2 - `Group Normalization`]
> <img width="300" alt="IMG" src="https://user-images.githubusercontent.com/73331241/173826286-32df1a30-0015-4dd0-8e1a-8f47c1ae0fd2.png">

> Group Normalization = Instace Normalization + Layer Normalization
>
> [Property 1: `Better than Instance Normalization`]
> > Since it has `more samples`, it can have `more stable statistics` than `Instance Norm`

> [Property 2: `Better than Layer Normalization`]
> > Normalizing different groups of channel differently. It means `not all channels tied` as in `Layer Norm`


#### [Normalization - Where do we need to add Normalization?]
> <img width="300" alt="IMG" src="https://user-images.githubusercontent.com/73331241/173827876-1b8d0016-b952-4921-908b-c31a5c66a6a4.png">


#### [BN - * Option A: After Convolution] => `More popular`

> <img width="300" alt="IMG" src="https://user-images.githubusercontent.com/73331241/174414647-3830bf8c-90da-44e0-81b5-1c16a9a80f03.png">

> Since any bias, which is added from Conv Layer, will be subtracted from the BN process. Thus, we can set `bias = False`

> `Possible problem`: Half of activations zeroed out in ReLU.
> 
> `Solution for zeroed out problem`: Learn a `scale variable s` and `bias b` after normalization



#### [BN - * Option B: After ReLU (Non-Linearity)]

> <img width="300" alt="IMG" src="https://user-images.githubusercontent.com/73331241/174414750-1e1b502b-8a05-426e-8dc9-a3546cc7795a.png">

> Benefit: Scale s and bias b is optional. (`Conv unchanged`)



#### [BN - Where `not to add` batch normalization]
> We don't have to put Any type of BN `after fully connected layers`. Because output of fully-connected layer `does not have spatial dimension`.

> Because of it, mean and startd deviation estimates too unstable.


#### [BN - Why does normalization work]
> `Key1`: Can handle `badly scaled weights`
> 
> `Key2`: We have a scale parameter which can scale entire channel easily. It can replace lots of parameters for the same tasks.(Good for training efficiency)


#### [BN - Coding mode]
> We have to remember set the mode of the model (train or eval). Because we just compute the statistics from training process not the testing.

```python
net = ConvNet()
net.train()
print(net.training)
# True

net.eval()
print(net.training)
# False
```

### [Relation between the Deepness of network and Normalization]

(1) `Without normalization`: Max reasonable trainable depth 10-12 consecutive layers

(2) `With normalization`: Max reasonable trainable depth 20-30 consecutive layers

> Parallel layer can be inserted inside of the consecutive layers, but in terms of consecutive layers, we need to keep that in mind.

> <img width="300" alt="IMG" src="https://user-images.githubusercontent.com/73331241/174417556-49241158-223e-495c-a7e5-0e3de9573629.png">

> > Unsuitable out-of-the-capacity layer works poorer than lower-layer network.

### [Make Better - `Residual Connection`]
> <img width="300" alt="IMG" src="https://user-images.githubusercontent.com/73331241/174958930-418193bf-7f88-4ed3-b451-606a5ccef358.png">

(1) One of the most fundamental building block of modern DNN.

(2) Backward propagation is symmetric with forward propagation

(3) Through Residual connections, 1000 layers model also can be trained not that terribly.


####  `Why do residual connection work? - Practical question and answer`

(i) Backward: Gradient travels further `without major modifications (vanishing)`

(ii) Forward: Reuse of patterns
> (ii-1) Since patterns are reused. Thus, it only needs to capture what is changing and only needs to update activation map. It doesn't need to re-compute everything.
> 
> (ii-2) Dropping some layers does not even hurt performance
> 
> (ii-3) Even though some layer has `Weights = 0`, network would not be destroyed.


####  `Thing to be careful in Residual Network Constuction`

* `Possible Problem`: Identity x and the output of the network can have different size because of stride parameter or different output channel size.

* `Solution`: Add `downsample` small network (Do `matching the number of channels` through n_intput, n_output arguements, and also do `matching shape` of it through stride argument.)

```python
class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
              torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride, bias=False),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU(),
              torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1, bias=False),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU()
            )
            self.downsample = None
            if stride != 1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output, 1, stride=stride), # prevent different numbers of channels and shape
                                                      torch.nn.BatchNorm2d(n_output))
        
        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)
            return self.net(x) + identity
```

----

## Optimization Algorithm

### `Types of Optimization Algorithm`

----
* Key: For the `large` models and data, `SGD` with Momentum is better. For the `small` models and data: `Adam` is better.

* Be careful: 
----

#### `[Types 1] Stochastic Gradient Descent with Momentum`

----
* Key: Works well in most cases

* Be careful: Tuning learning rate is necessary
----

#### `[Types 2] RMSProp`

> Slight modification of SGD

----
* Key: Working well on some RL problem

* Be careful: We don't use it that much from Computer Vision and Image Processing
----

#### `[Types 3] ADAM`

----
* Key: 

* Be careful: 
----

### `Hyper Paramter`

Learning Rate  	                      |      Momentum    | Batch Size
:---------------:                     | :-------------:  | :-------------:
We should care the most regularly among Hyper Parameter  | We can use the default setting | Always try to use large size that GPU allows
Use close to the largest LR that trains  | e.g.) SGD-rho = 0.9, ADAM-beta1 = 0.9, beta2 = 0.999 | 



#### `[HP] Learning Rate - Learning Rate vs Batch Size`

----
* Key: Linear Scaling Rule: When the minibatch size is multiplied by k, multiply the learning rate by k.
----

#### `[HP] Learning Rate - Learning Rate and Training Result Relation`

> Situation1: Val_loss is not going down or remain constant: Learning rate is too small

> Situation2: Val_loss is going up or oscillate wildly: Learning rate is too big


#### `[HP] Learning Rate Scheduling- PyTorch`

----
* Key: Using `torch.optim.lr_scheduler.ReduceLROnPlateau` allow us to tune Learning Rate automatically as soon as our model stop making progress.

* Be careful: Put the proper criteria, corresponding to `lr_scheduler argument`, into `scheduler.step(np.mean(accuracies)`)
----

<!--

I skipped Network distillation

Lec: %.making_it_work

from to 3:21 ~ 3:30

## How do we train a small network?

> Idea 1: Randomly intialize and train network.

> Idea 2: Train a larger network and make it small.

`Idea 2 will lead a better network`

### `[] Network distillation`

----
* Key: Take

* Be careful: 
----

#### `[] KEY WORD 1`

>

ELEMENT1  	      |      ELEMENT2
:---------------: | :-------------:
EXPLANATION1  | EXPLANATION2
EXPLANATION3  | EXPLANATION4
EXPLANATION5  | EXPLANATION6

-->


## Overfitting

### `[Overfitting] Basic Understanding`

----
* Definition: Fit model to training set, and the model work poorly on unseen data (e.g. validation set)

* Key: We can detect overfitting by plotting training and validation accuracy.

* Reason of overfit: Our training dataset and actual real-life application(test) can have slightly different data distribution.

* Be careful: Even though we have adequate data, if data distribution between training and test data are still different, we will have overfitting issue too.
----

### `[Overfitting] How to prevent overfitting`

Many ways to prevent overfitting 
:---------------:
(1) Collect more data  
(2) Make the model simpler through regluarization  
(3) Transfer learning  


Training considering overfitting
:---------------:
(1) Early stopping in practice 
` `
Don't have a stop button. Just save our model periodically. Then, we choose the model in person after considering train and validation graph. 



### `[Overfitting] Data Augmentation`
----
* Key: `capture invariances` in data. Always use data augmentation if possible.

Type of Image Augmentations 
:---------------:
(1) Geometric augmentations
(2) Color augmentations

> <img width="300" alt="IMG" src="https://user-images.githubusercontent.com/73331241/175846812-b2587aca-ba26-40b4-a407-b7ea5caf150e.png">

* be careful: some augmentations require augmentation of labels (such as in segmentation, we have to crop the same location in label as well.)
----

Data augmentation in training
:---------------:
(1) (Randomly) augment every single iteration
` `
`Network never sees exact same data twice`


### `[Overfitting] Overfitting in deep networks`

----
* Key: Exploit patterns from training data make some activation node bigger. However, that patterns can be invalid from val / test data.

* Be careful: Deeper layers get the overfitting impact more since it relys on overfit activations from previous layers over and over again.


----


#### `[Prevent Overfitting] Dropout`

* Key: Randomly remove some activation nodes.

* Be careful: Do not use dropout before general convolutions.

Dropout during Training  	      |      Dropout during Evaluation
:---------------: | :-------------:
With probability `a` set activation a_l(i) to `zero`  | Use all activations, but scale by multiplying  `1/(1-a)`

> <img width="250" alt="IMG" src="https://user-images.githubusercontent.com/73331241/176069901-24d77714-c9a4-4534-8444-111311d0259d.png">


Where to add dropout   	      
:---------------: 
Before any large fully connected layer
Before some 1X1 convolutions
`Not before general convolutions`


Smaller model

Pro  	      |      Con
:---------------: | :-------------:
Overfits less  | Worse generalization


Big model with regularization

Using `Weight decay`: Not only optimize our loss, but also add weight term in optimization process to maintain weight small.

> Optimizer charges of `Weight decay` (Parameter call `weight_decay` inside of the torch.optim.SGD or torch.optim.Adam)

Advice: Use the default value of `weight_decay`.

Weight Decay Pro                  |     Weight Decay  Con
:---------------:                 | :-------------:
Keep weights small                | Worse generalization
Helps handle exploding gradients  |   


Ensembles
Key: Train multiple models and use average predictions of multiple models

Can consider different random initializations / data augmentation

> <img width="300" alt="IMG" src="https://user-images.githubusercontent.com/73331241/176078701-69b54c40-6cb8-45f9-a3e7-16b8f27c5b16.png">

Why do ensembles work?

Ensemble Pros                  |     Ensemble Cons
:---------------:                 | :-------------:
Fewer parameters                | Longer training
Each model overfits in its own way  |   ` `
Boost 1-3 % accuracy on most tasks  |   ` `

When to use ensembles?

* If you have the compute power

* If you really need the last bit of accuracy

Argumentation Code in PyTorch

```python
if __name__ == "__main__":
    import torch
    # Parse all input arguments
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('logdir')
    parser.add_arguement('-n', '--n_epochs', type=int, default=10)
    parser.add_arguement('-o', '--optimizer', default='optim.Adam(parameter)')
    parser.add_arguement('-b', '--batch_size', type=int, default=128)
    parser.add_arguement('--no_normalization', action='store_true')
    args = parser.parse_args()
    
    ...
    
    # Parse the optimizer
    optimizer = eval(args.optimizer, {'parameters': net.parameters(), 'optim': torch.optim})
    
    # Train
    train(net, args.logdir, device=device, resize(128, 128), n_epochs=args.n_epochs, optimizer=optimizer)
```

## Practical Guide in deep learning

### `[Practical] How to training small datasets`

----

```diff
+ Solution: Utilize pre-trained model and do fine-tuning

+ Key: Use transfer learning whenever possible if large pre-trained model exists.
```

### `[Practical] Generalization in deep learning`

```diff
- Wrong intuition: Bigger/wider models overfit more.

+ Key: Bigger model can generalize better than small one. Hence overfit less.

@@ Tip: (1) Models overfitting is dependent on learning algorithms. @@
@@ Therefore, SGD and data augmentation is robust for overfitting. However, Adam makes overfitting worse. @@

- Be careful: Measuring overfitting is still wide opened problem.
```

> <img width="300" alt="IMG" src="https://user-images.githubusercontent.com/73331241/176103088-784eca7f-b988-4950-9c87-ee0bda8e2d8a.png"><img width="300" alt="IMG" src="https://user-images.githubusercontent.com/73331241/176103255-30542d5b-a4e2-4cf4-b202-5ca24aa5610f.png">


### `[Practical] Graduate Student Descent`

> <img width="300" alt="IMG" src="https://user-images.githubusercontent.com/73331241/176121668-9b312b17-1fce-4505-9fe6-f268a4135cee.png">

```diff
+ Possible manual modification 1: Learning rate
+ Possible manual modification 2: Architecture
+ Possible manual modification 3: Initialization
+ Possible manual modification 4: Normalization
+ Possible manual modification 5: Data augmentation
+ Possible manual modification 6: Loss function

- Vanishing or Exploding gradient(can notice through plotted loss function): Fix initialization and learning rate

- Slow training: Add normalization or Use residual connections.

- Overfitting: Add data augmentation or Early stopping or Try a pre-trained network or Collect more data

@@ If network fits training and validation data well, evaluate on test set.  @@

- If test set performance is poor: Include test into validation set and then collect new test set

```


## Recurrent Neural Network

### `[RNN] Inside of the Recurrent Neural Network`

----
```diff
+ Key: RNN maintains its internal state h(t) - which is updated at each step.

- Be careful: 
```
----


#### [RNN] Intuition - Sequence Modeling Applications

> <img width="500" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177021067-d6fc81ae-b3b4-48f3-a44e-ba773d704630.png"><img width="500" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177021837-d9007556-a91c-4f5a-bd85-efe73fbf09a9.png">


#### `[RNN] Neurons with Recurrence`

> <img width="400" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177021232-b6e0dbec-9534-4ff7-9bfe-673e391b8387.png">

----
```diff
+ h_t: maintain internal memory or state of prior tasks and pass it to forward.

! Training process 1: h_t is updated at each time step as a sequence is processed.

! Training process 2: the same weight function w is the same across all time steps.

+ Key: The output of the node consider the `input` and `past memory`

- Be careful: 
```

##### `RNN internal state computation`

> <img width="300" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177021388-6e5fe846-16c1-4de6-b276-77835794d390.png">

> <img width="300" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177021541-7558918a-0e96-4ebb-b135-e4182f7409b4.png">

----

#### `[RNN] Code Intutition`

`Task: predict the next word that's going to come at the end of the sentence.`

##### `Step 1: RNN Tensorflow model from Scratch`

> <img width="500" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177021781-39effac4-1ac0-4cf7-9f40-078d8e25efe0.png">

##### `Step 2: Use the functions`

> <img width="500" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177021456-b6d8f601-e76a-49d3-b009-bbf0cde6da33.png">

##### `Tensorflow application`

`tf.keras.layers.SimpleRNN(run_units)`


#### `[RNN] Sequence Model Design Criteria`

----
```diff
 + 1. Handle `variable-length` sequences (able to short or long sentence)
 + 2. Track `long-term` dependencies
 + 3. Maintain information about order
 + 4. Share parameters across the sequence to keep track of the dependency
```
----

#### [RNN] Graph

> <img width="500" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177021611-fe2b9df1-8602-493d-a357-1ac0dc01e6d9.png"> <img width="500" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177022257-4e7d92e8-bf49-4305-a203-ec4ea49ae305.png">
Key: Re-use the same weight matrices at every time step

#### `[RNN] Encoding Language for a Neural Network`

----
```diff
+ Key: Neural networks cannot interpret words, so they require numerical inputs
```

> <img width="500" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177022135-dceaa549-5b85-4cc6-9585-e140844a4783.png">
----

#### `[RNN] Overcome Vanishing Gradient`

----
```diff
+ Key: Neural networks cannot interpret words, so they require numerical inputs
- Be careful: Vanishing gradients problem can be more crucial in longer sentences.
```
----


#### `[Overcome Vanishing Gradient] Trick #1: Activation Functions`

> <img width="500" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177022443-560aa324-309a-4a20-8d2d-06f476be0615.png">


#### `[Overcome Vanishing Gradient] Trick #2: Parameter Initialization`

> Initialize weights to identity matrix and biases to zero


#### `[Overcome Vanishing Gradient] Trick #3: Gated Cells`

> `Most robust solution: LSTM`


#### [RNN] Long Short Term Memory (LSTMs)

----
```diff
+ Key: use gates to `selectively` add or remove information within each recurrent unit with gates (optionally let information through the cell).

! Key conept of LSTMs 1: networks rely on a gated cell to track information throughout many time steps.

! Key conept of LSTMs 2: Use `gates` to control the `flow of information`
!    (1) `Forget` gate gets rid of irrelevant information
!    (2) `Store` relevant information from current input
!    (3) Selectively `update` cell state
!    (4) `Output` gate returns a filtered version of the cell state

! Key conept of LSTMs 3: Backpropagation through time with partially `uninterrupted gradient flow`.

- Limitations of RNN 1: Slow to train. (We cannot use parallelization because of sequence dependency)

- Limitations of RNN 2: Long sequences lead to vanishing / exploding gradients
```

> <img width="250" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177022483-8e9a2638-b418-45c6-8a66-aae80403fffb.png">

> <img width="500" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177022534-394e18e4-6878-43fe-8bab-f5c93e020db1.png">
----


<!--

Transformer

> There is no concept of time step for the input. Put the sentence (e.g. The red dog) simultaneously.

![20220703_134524](https://user-images.githubusercontent.com/73331241/177025047-8283d57e-78e1-4a57-942d-aa783cb99a1d.png)

[Transformer Components]
Embedding Space: Similar words or words with similar meaning have physically close location with each other.
(We can use pre-trained Embedding Space)

Attention: What part of the input should we focus?


(First step) Input Embedding: Put the word as an input into the Embedding Space, we can get corresponding vector.

Positional Encoders: Vector that gives context based on potition of word in sentence.
(The meaning of word can be different according to the position in sentence.)

![20220703_135002](https://user-images.githubusercontent.com/73331241/177025200-0d2a04d5-526b-448f-a4af-7fbce3999f72.png)


Encoder Block

![20220703_135301](https://user-images.githubusercontent.com/73331241/177025278-390c9298-3162-426d-941c-5fa67ed71f69.png)

(Second step) Multi-Head Attention layer
> Get the attention vectors

![20220703_135208](https://user-images.githubusercontent.com/73331241/177025255-fdc5dadd-6556-4e30-99a2-d9f49c44d50a.png)

(Second step) Feed Forward layer
> Feed Forward layer is applyed to every attention vectors

> In practice, to transform attention vector into the form of digestable form to used in encoder block or decoder block.

![20220703_135744](https://user-images.githubusercontent.com/73331241/177025408-87b9f885-40bf-4623-bd0b-9847cd9b0d7b.png)

Decoder

-->

<!--
Limitations of Recurrent Models

(1) Encoding bottleneck (We can lose information from encoding process)
(2) Slow to train (no parallelization)
(3) Not long memory


### Transformer

Intuition Behind Self-Attention: Attending to the most important parts of an input.

1. Identify which parts to attend to. (Similar to a search problem!)
2. Extract the features with high attention


Understanding Attention with Search through YouTube Example
[1] Compute attention mask: How similar is each key to the desired query?

![20220703_132120](https://user-images.githubusercontent.com/73331241/177024445-5ed31ce2-055c-446c-b644-e3e3f6c181b0.png)

[2] Extract values based on attention: Return the values hightest attention


Understanding Attention with Search through YouTube Example

-->




## Reinforcement Learning

### `[RL] `

----
```diff
+ Key: 


- Be careful: 

! Why Reinforcement Learning has a loop?
(1) To differentiate considering path information.
(2) To be able to recognize long-range dependencies.
```
----

> <img width="500" alt="IMG" src="https://user-images.githubusercontent.com/73331241/181663187-8261d646-aa18-42c3-9d30-92901c330fd2.png">


Simple way to think: Deep Network plays video game

Environment: Gives the certain state (or observation) and reward(how well agent's doing) to agent. So it tells the agent what it can see.

Agent: Take the information(e.g. image) from environment and takes the action. So it tells the environment what it would like next.


How to train agent: Environment tells agent whether they do well or not.


Markov decision process (MDP) - Formal definition












----

# `Lec 5: 3:54:00 : 2022-06-27`

--------
?torch.nn.BachNorm2d

<!--

## [] PyTorch
#### [ -1] Basic tensor
```python
t = torch.zeros(10) # tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

# Control the dimension of tensor
t[None, :]          # torch.Size([1, 10])

torch.ones([4,5]) == torch.ones(4,5)
#tensor([[1., 1., 1., 1., 1.],
#        [1., 1., 1., 1., 1.],
#        [1., 1., 1., 1., 1.],
#        [1., 1., 1., 1., 1.]])

v = torch.arange(100)
# tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, ..., 99])

# "view" is very useful in resizing
v.view(10, -1) # we can put -1 to define another size automatically
# tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
#        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
#        [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
#        [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
#        [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
#        [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
#        [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
#        [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
#        [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]])
```

#### [ -2] Load image
```python
from PIL import Image
I = Image.open('cat.jpg')

# 1. covert image to numpy / size: (220, 220, 3)
np.array(I)
# array([[[213, 210, 175],
#        [217, 214, 179],
#        [222, 219, 188],
#        ...,
#        [234, 219, 216],
#        [236, 218, 214],
#        [236, 218, 214]]])

# 2. covert image to tensor: torchvision module is necessary / torch.Size([3, 220, 220])
from torchvision import transforms
# define the class for transformation
image_to_tensor = transforms.ToTensor()
image_tensor = image_to_tensor(I)

# 3. convert tensor to image
tensor_to_image = transforms.ToPILImage()
tensor_to_image(image_tensor)
```

#### [ -3] Linear Neural Network
```python
class Network1(torch.nn.Module):
    def __init__(self, n_hidden=100):
        super().__init__()                                          # inherit parent's __init__ function
        self.linear1 = torch.nn.Linear(input_size, n_hidden)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(n_hidden, 1)
    
    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x.view(x.size(0), -1))))

# genertate the Class
net1 = Network1(100)
# check the result
print()
```

```python
class Network2(torch.nn.Module):
    def __init__(self, *hidden_size): # *hidden_size receives multiple lists.
        super().__init__()
        layers = []
        # Add the hidden layers
        n_in = input_size
        for n_out in hidden_size:
            layers.append(torch.nn.Linear(n_in, n_out))
            layers.append(torch.nn.ReLU())
            n_in = n_out
        
        # Add the output classifier
        layers.append(torch.nn.Linear(n_out, 1))
        
        # Define Neural Network
        self.network = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x.view(x.size(0), -1))
```
#### [ -4] Training code

```python
# Training code
# TODO: initialize some variables / hyper-parameters
n_epochs = 100
batch_size = 128

# TODO: Create the logger
import torch.utils.tensorboard as tb
train_logger = tb.SummaryWriter(log_dir+'/deepnet1/train', flush_secs=1)
valid_logger = tb.SummaryWriter(log_dir+'/deepnet1/valid', flush_secs=1)

# TODO: Create network
# Take the model and upload it to GPU
net2 = Network2(100,50,50).to(device)

# TODO: Create optimizer
optimizer = torch.optim.SGD(net2.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# TODO: Create loss function
loss = torch.nn.BCEWithLogitsLoss()

# TODO: Start training
global_step = 0
for epoch in range(n_epochs):
    # TODO: Shuffle the data
    permutation = torch.randperm(train_data.size(0)) # make the permutation from 0 to the number of training dataset -1
    
    # TODO: Iterate
    train_accuracy = []
    for it in range(0, len(permutation)-batch_size+1, batch_size):
        # TODO: Construct batch
        batch_samples = permutation[it:it+batch_size]
        batch_data, batch_label = train_data[batch_samples], train_label[batch_samples]
        
        # TODO: Compute output
        o = net2(batch_data)
        
        # TODO: Compute the loss
        loss_val = loss(o, batch_label.float())
        
        # TODO: Logging
        train_logger.add_scalar('train/loss', loss_val, global_step=global_step)
        
        # TODO: Compute the accuracy
        train_accuracy.extend(((o > 0).long() == batch_label).cpu().detach().numpy()) # transfer all of te label data to cpu and detach it from computational graph and convert it to numpy
        
        # TODO: Compute the gradient from loss function
        optimizer.zero_grad() # through zero_grad(), we don't accumulate the gradient over multiple iteration.
        loss_val.backward()
        
        # TODO: Update model weights
        optimizer.step()
        
        # TODO: Increase the global step
        global_step += 1
    
    # TODO: Evaluate the model
    valid_pred = net2(valid_data) > 0
    valid_accuracy = float((valid_pred.long() == valid_label).float().mean())
    # TODO: Log training accuracy
    train_logger.add_scalar('train/accuracy', np.mean(train_accuracy), global_step=global_step)
    # TODO: Log validation accuracy
    valid_logger.add_scalar('valid/accuracy', valid_accuracy, global_step=global_step)
```

#### [ -5] Convolutional Neural Network

```python
class ConvNet2(torch.nn.Module):
    def __init__(self, layers=[], n_input_channels=3, kernel_size=3, stride=2):
        super().__init__()
        L = []
        c = n_input_channels
        for l in layers:
            L.append(torch.nn.Conv2d(c, l, kernel_size, padding=(kernel_size-1)//2))
            L.append(torch.nn.ReLU())
            L.append(torch.nn.MaxPool2d(3, padding=1, stride=stride))
            # L.append(torch.nn.MaxPool2d(2*stride-1, padding=stride-1, stride=stride))
            # "2*stride-1" kernel size guarantees whatever stride we have, we always pool over all input in our activation map.
            c = l
        L.append(torch.nn.Conv2d(c, 1, kernel_size=1))
        self.layers = torch.nn.Sequential(*L)
    
    def forward(self, x):
        return self.layers(x).mean([1,2,3])

net3 = ConvNet2([32,64,128])
```

```python
class ConvNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
              torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
              torch.nn.ReLU(),
              torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
              torch.nn.ReLU()
            )
        
        def forward(self, x):
            return self.net(x)
        
    def __init__(self, layers=[32,64,128], n_input_channels=3): # layers have the number of channels
        super().__init__()
        L = [torch.nn.Conv2d(n_input_channels, 32, kernel_size=7, padding=3, stride=2),
             torch.nn.ReLU(),
             torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        c = 32
        for l in layers:
            L.append(self.Block(c, l, stride=2))
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, 1)
    
    def forward(self, x):
        # Compute the features
        z = self.network(x)       # torch.Size([10, 3, 128, 128])
        # Global average pooling
        z = z.mean(dim=[2,3])     # torch.Size([10, 128])
        # Classify
        return self.classifier(z)[:,0] # [:,0] removes one dimension

net = ConvNet()
```

#### [ - ] Tensorboard

<!--
Course Information: http://www.philkr.net/cs342/
-->

-->
