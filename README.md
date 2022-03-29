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
      
      
### [0-2] Probability and likehood

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

### [1-1] Diference between linear regression and classificaiton

> * Regression predicts real value(smooth range)
> 
> * Classification preditcs integer/class(specific kinds)

## [1-2] Linear regression / logistic regression / multi-class classification

> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/160286720-7b4eea2c-58b2-4c82-a052-bfc3b5bb06c3.jpg">

> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/160286721-33e26992-3334-492c-b604-c7b21cd170d1.jpg">

## [1-3] Backpropagation
> * [1] Purpose: to update the parameters with the optimal values by analyzing and differentiating the loss function
>  
> * [2] Benefit: Different from numerical differentiation, Backpropagation `saves the differentiation result matrix` from the the highest layer, then goes down `using saved matrix`. Thus, it is beneficial when it comes to memory utility and the amount of operation.

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


#### [ - ] Tensorboard
