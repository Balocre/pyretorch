![PyreTorch Logo](https://github.com/Balocre/pyretorch/blob/master/baniere.png)

---

# Py(re)Torch

Like PyTorch, but worst...

---

## A propos

This project started as an exercise I had to do for a Machine Learning course
for my Masters. It is a reimplementation of (a very small part of) PyTorch,
using Python built-in modules and NumPy.
Ultimately I'd like to make a pedagogic tool for anyone who wants to understand
the basics of Machine Learning / Deep Learning from a practical point of view.

## What's inside

As of now, this project structure follows the one of the PyTorch project.
I also tried to match the PyTorch calling conventions.

The main difference from PyTorch (apart from the immense performance gap, and 
the lack of functionnalities) with PyTorch is that this project does not 
implement the "autograd mechanism", so, backpropagation must be done manually 
when implementing a custom model.

Here is a list of functionalities that are as of yet partially or fully 
implemented :


### modules :

 - activation : `Sigmoid`, `Tanh`, `SoftMax`, `ReLU`
 - conv : `Conv1d`, `Conv2d`
 - dropout : `Dropout`
 - flatten : `Flatten`
 - linear : `Linear`
 - loss : `MSELoss`, `CrossEntropyLoss`, `BCELoss`, `BCELossWithLogits`
 - module : `Module`
 - pooling : `MaxPooling1d`, `MaxPooling2d`
 - sequential : `Sequential`

### optim :
 - sgd : `SGD`
 - lr_scheduler : `StepLR`

 There are also basic saving/loading functions in the `serialization` file and
 a custom training script `train`.
