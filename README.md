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


## [1] Regression and classification

### [1-1] Difference between linear regression and classificaiton

> * Regression predicts real value(smooth range)
> 
> * Classification predicts integer/class(specific kinds)

## [1-2] Linear regression / logistic regression / multi-class classification

> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/160286720-7b4eea2c-58b2-4c82-a052-bfc3b5bb06c3.jpg">

> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/160286721-33e26992-3334-492c-b604-c7b21cd170d1.jpg">

## [1-3] Backpropagation
> * [1] Purpose: to update the parameters with the optimal values by analyzing and differentiating the loss function
>  
> * [2] Benefit: Different from numerical differentiation, Backpropagation `saves the differentiation result matrix` from the the highest layer, then goes down `with saved matrix`. Thus, it is beneficial when it comes to memory utility and the amount of operation.

> <img width="950" alt="IMG" src="https://user-images.githubusercontent.com/73331241/160856967-d7de2f36-f8bb-491b-b9f3-4b3c76186494.jpeg">


## [2] Neural Network Modeling

```python
# Type the below to notice what is necessary arguments
?torch.nn.Linear
?torch.nn.Conv2d
?torch.nn.MaxPool2d
```

### [2-1] Classifier

> #### (1) The `limitation of linear classifier`: we cannot solve the problem which `requires curve` to classify two groups
> 
> * #### Does adding more linear layers help?
>      > No. Combination of linear layers still linear.

> #### (2) Non-linearities: ReLU(x) = max(x,0)
> 
> Non-linear and differentiable almost everywhere


### [2-2] Positive regression

> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166146143-4d7fb0d0-9d4b-4300-a373-dde7fdfce2b0.png">
> 
> > option1(ReLU)
> > > just want to `predict a value` that is positive, and you `don't care about to train network` we can use ReLU function.
>
> > option2(Soft ReLU)
> > > if we `want to train` a network with positive prediction, we should always use softrelu(softplus) function


### [2-3] Binary Classification

option1 Thresholding: step function

> hard to train because output activation function shape is `step function`. Thus, the derivative in everywhere is 0.

`option1 Logistic Regression`: sigmoid


### [2-3] General Classification

> option1: argmax
> 
>  > hard to train because we `cannot compute` gradient of argmax function.

> `option2: softmax`
> 
>  > optimize it using `cross entropy`

> `Tip in output representations`
> 
>  > `do not` `add` other `output transformation` into model's output. Always output raw values
>  > 
>  > reason: many output transformation makes `harder to differentiate`. If bigger output transformation, numericably unstable.

### [2-4] Loss

> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166390923-c1c541ce-8c65-4c3b-a8f8-8b18b289ecee.png">

> #### (1) Loss - Regression
> 
> > <img width="190" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166210192-74bdc977-674a-4549-aedf-33fd3c9ae40d.png">
>
> `difference` between L1 loss and L2 loss: when we predict wrong, L1 loss increases linearly, but L2 loss increases dramatically(Huge gradient, huge loss).
> 
> Additional information: in deep learning, it came out that L1 loss and L2 loss doesn't make a huge difference.

> #### (2) Loss - Classification
> 
> > (i) -Log likelihood
> > - log(p(y))
> 
> > y: ground truth label
> > 
> > p = activation function's output (e.g. softmax(o))
> > 
> > p(y): take the probability of the ground truth label through our output activation function
>
> > (ii) Cross entropy: sum over all labels y
> > 
> > `In one-hot encoding, cross entropy and -log likelihood is exactly the same.`
>
> `[Issue]`
> > * From the classification process, because of sigmoid and softmax functions attributes, we can get 0 from the activation function. Thus, once we put this value into log, there is no value is defined.
>
> `[Solution]`
> > * To solve it, modern deep learning package combine log and sigmoid or combine log and softmax
> > 
> > ```python
> > # In PyTorch: log + sigmoid
> > BCEWithLogitsLoss
> > ```
> > 
> > ```python
> > # In PyTorch: log + softmax
> > CrossEntropyLoss
> > ```

### [2-5] Gradient Descent

#### [GD-Stochastic Gradient Descent]

<img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166613395-fad93739-6997-4582-9baf-0e0b625ca367.png">

`highly oscillate and has large variance.`

> Compute the gradient as we move, so always we take a step and compute the gradient and take a step ...

> SGD can `work better` than standard gradient descent in `non convex function`.
> 
> reason1: oscillation of SGD helps to explore the function
> 
> reason2: faster. We don't have to wait to loop over entire training set to compute gradient



> #### (1) Variance of SGD
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166393020-04314dc3-0935-4eef-a871-80c87f9402b2.png">
> 
> measure the agreement between different data samples about gradient, we can compute the variance of gradient estimates.
> 
> difference between one single data point and the gradient of entire objects' loss
> 
> Reason of introduction(biggest issue): `not` all training set give a `good direction` to update

### [2-6] Mini-batches
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166613762-e105009e-af17-4cec-8dd0-31a8909ed68c.png">
> 
> Way to `reduce the variance` of SGD. Thus we should use mini-batch as much as we can.
> 
> ##### `What Mini-batches do?`
> * Not just get one data element(x, y), it `takes bunch`(so-called batch or mini-batch) of data elements.
>
> ##### `How mini-batches update parameters?`
> * for each mini-batch, we compute the `average gradient` over the mini-batch and `update the parameters` instead of updating with single data element.
> 
> ##### `What is the benefits of mini-batches?`
> * Estimate the gradient robustly, reduce the variance of SGD massively
>
> ##### `Difference in variance`
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166613961-a3ecc639-17b5-44ee-9f4c-ad83b8181703.png">

### [2-7] Momentum
> <img width="190" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166614563-f4a678d3-7cc3-4534-937c-d713c02809c9.png">
> 
> Momentum(v) is that `functionalized gradient` by adding previous Momentum(v). It helps to take into account the past. Rho is the parameter for past considering.
> 
> Additional Benefit of Momentum
> 
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166614982-b3a36031-062b-4487-bbae-9e821a27481e.png">
> 
> Variance reduction of Momentum (since update term(`blue`) is getting smaller.)

* Use mini-batched and Momentum every time.

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

### [2-9] Layers
> ##### `What is a layer?`
> 
> > Largest computational unit that remains unchanged throughout different architectures
> 
> ##### `Layer naming`
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166881431-3d53270c-10d2-45f3-9396-114fe5c1b6b3.png">
> 
> > The number of layer: 4
> > 
> > Normal output of activation function: `activation`
> > 
> > The output of activation function goes in to output layer: `feature`


### [2-10] Non-linearities (activatino function)
> <img width="190" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166887445-318949cb-63e2-4c5b-b421-657c4f09e20d.png">
> 
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
> > Try to use `ReLU first`. If you have a problem with Dead ReLU, we can try `Leaky ReLU`, and if Leadky ReLU works, try to use `PReLU`.
> > 
> > sigmoid and tanh are usually used in Natural language process or RNN.

### [2-11] Hyper-parameters
> Any parameters set by hand and not learned by SGD.

### [2-12] `Loss function selection`
> <img width="250" alt="IMG" src="https://user-images.githubusercontent.com/73331241/166894368-990121d0-6d63-4f55-a4bc-b67ca5316b5e.png">

* MSELoss is also referred to as L2Loss
* 
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
> > > <img width="250" alt="IMG" src="https://user-images.githubusercontent.com/73331241/167284715-803aa68c-becf-47fe-addb-41b365145bd4.png">
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
> In each channel, it computes the mean of all possible x, y location in side of the kernel.
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
> `Global average pooling`: produce one single spatial output that has same number of channels as the input channels of average pooling.
> 
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/167761686-1454ad15-3194-4c17-b40a-036804c93d27.png">

> > Then we can use Global average pooling output from Linear layer to classify
> > Average Pooling layer is not used at the middle position of the network.

##### (2) Max Pooling: `Non-Linear layer`
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/167761841-78be228d-9efc-4974-887d-bbdcc5577290.png">

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
> The receptive field is defined as the region in the `input space` that a `particular CNN’s feature(kernel) is looking at` (i.e. be affected by).
> 
> However, not all pixels in a receptive field is equally important to its corresponding CNN’s feature.
> Within a receptive field, the closer a pixel to the `center of the field`, the more it contributes to the calculation of the output feature.
> 
The pixel at the center can influence more than the pixel at the edge.
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/167976356-b9a49047-1348-47c6-b601-5948bb2575a7.png">

> Therefore, pixels at the center of receptive field have much higher weight on the output decision than the side one.


### [CNN - `Model Design Principles`]
##### [1] Use striding(downsample activation map), increase channel. It helps balance some of the computation that held inside of deep neural network.
* Striding by 2 and increase the number of channel by the factor of 2, this will make sure all the convolusions have roughly the same number of operation throughout the network.
>
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/167977239-582a241d-5507-4a85-9ae4-86994142a04c.png">

##### [2] Keep kernels small
  * 3 X 3 Kernels almost everywhere
  * 
  * Reason: (1) Small kernel size with deep network has a benefit in the number of parameters., (2) We can put more non-linearity between 3X3 Kernel
  * 
  * `Exception`: First layer up to 7X7 (Reason of exception: First layers with 7X7 kernel size allows us to capture more interesting patterns in the input image itself.)

> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/167978054-fd417e28-3083-492c-8d11-5b5f38557c9f.png">

##### [3] Use repeat patterns
 * First layer or two are special. So we can set paramters sophisticatedly.
 * 
 * However, all other layers usually follow a fixed pattern, because setting all the hyperparameters for all layers are so overwhelming.

##### [4] Keep all network convolutional] 
> Use Avg Pooling layer at the end.
[Benefits]
 * Fewer parameters
 * 
 * Better training signal
 * 
 * Ensemble voting effect for testing
 
> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/167979391-708665f5-cfd0-4e4c-b182-ebf4c1d6243a.png">


----

# `Lec 5: 1:48:40 : 2022-05-12`

--------

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

#### [ - ] Tensorboard

<!--
Course Information: http://www.philkr.net/cs342/
-->
