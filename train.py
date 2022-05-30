import sys
import logging

import numpy as np

from typing import List
from numpy.typing import ArrayLike

from retorch.modules.loss import Loss
from retorch.modules.module import Module
from retorch.optim.sgd import SGD
from retorch.optim.lr_scheduler import StepLR

TRAIN_MESSAGE  = "++=[{:03}/{:03}]=(TRAINING)===>>> [batch : {:04}/{:04}, lr : {:.2E}]"
VALID_MESSAGE  = "++=[{:03}/{:03}]=(VALIDATION)=>>> [batch : {:04}/{:04}]"
FINISH_MESSAGE = "||=[{:03}/{:03}]=(FINISHED)===>>> [loss train : {:.6f}, loss val : {:.6f}, lr : {:.2E}]"

def train(
    net: Module,
    optimizer: SGD,
    examples: ArrayLike, 
    targets: ArrayLike, 
    batch_size: ArrayLike, 
    epochs: int, 
    loss: Loss,
    val_split: float = 0.2,
    scheduler: StepLR = None,
    quiet = False
) -> List[int] :
    """
    Training script with support for validation
    """

    if val_split < 0 or val_split >= 1:
        raise ValueError('Split value must be in [0, 1[')

    split_idx = int(val_split*len(examples))

    examples_valid, examples_train = np.array_split(examples, [split_idx], axis=0)
    targets_valid, targets_train = np.array_split(targets, [split_idx], axis=0)

    # XXX: le dernier batch incomplet est abandoné
    if (c := (len(examples_train) % batch_size)) != 0:
        examples_train, targets_train = examples_train[:-c], targets_train[:-c]
    if (c := (len(examples_valid) % batch_size)) != 0:
        examples_valid, targets_valid = examples_valid[:-c], targets_valid[:-c]


    if len(examples_train) > batch_size:
        examples_train = np.split(examples_train, len(examples_train)//batch_size, axis=0)
        targets_train = np.split(targets_train, len(targets_train)//batch_size, axis=0)
    else:
        raise RuntimeError('Training dataset size can\'t be smaller than batch size')

    if len(examples_valid) > batch_size:
        examples_valid = np.split(examples_valid, len(examples_valid)//batch_size, axis=0)
        targets_valid = np.split(targets_valid, len(targets_valid)//batch_size, axis=0)


    loss_train_at_epoch = []
    loss_val_at_epoch = []

    for epoch in range(1, epochs+1):

        # training
        l = 0
        i = 0
        
        net.train()
        for examples_batch, targets_batch in zip(examples_train, targets_train):
            if quiet == False and (int(len(examples_train)*0.1) != 0 and i % int(len(examples_train)*0.1) == 0): # print roughly every 10%
                print(TRAIN_MESSAGE.format(epoch, epochs, i, len(examples_train), optimizer.lr), end='\r')
            i = i+1

            optimizer.zero_grad()
            prediction = net.forward(examples_batch)

            l += loss.forward(prediction, targets_batch)

            delta_loss = loss.backward(prediction, targets_batch)
            net.backward(examples_batch, delta_loss)

            optimizer.step()            

        loss_train_at_epoch.append(l/len(examples_train*batch_size))

        if val_split == 0: # ignore validation if no validation set
            if quiet == False:
                print(FINISH_MESSAGE.format(epoch, epochs, loss_train_at_epoch[-1], np.nan, optimizer.lr))  
            continue

        # validation
        l = 0
        i = 0

        net.eval()
        for examples_batch, targets_batch in zip(examples_valid, targets_valid):
            if quiet == False and (int(len(examples_valid)*0.1) != 0 and i % int(len(examples_valid)*0.1) == 0): # print roughly every 10%
                print(VALID_MESSAGE.format(epoch, epochs, i, len(examples_valid)), end='\r')
            i = i+1

            optimizer.zero_grad()
            prediction = net.forward(examples_batch)

            l += loss.forward(prediction, targets_batch)

            optimizer.step()

        loss_val_at_epoch.append(l/len(examples_valid*batch_size))

        if quiet == False:
            print(FINISH_MESSAGE.format(epoch, epochs, loss_train_at_epoch[-1], loss_val_at_epoch[-1], optimizer.lr))

        if scheduler != None: # one step of scheduler
            scheduler.step()     

    return loss_train_at_epoch, loss_val_at_epoch