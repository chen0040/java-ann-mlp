package com.github.chen0040.mlp.enums;


/**
 * Created by xschen on 6/6/2017.
 *
 * The weight update mode specifies how often the weights should be updated for a number of training cases:
 * The "online" version essentially update the weights after each training case.
 * The "mini-batch" version update the weights over a batch of training cases.
 * The "full" version update the weights over all the training cases.
 *
 * The "mini-batch" is considered to be the most appropriate for most of the deep net training.
 *
 * For the "mini-batch" to work well on a multi-class classification problem, the batches should be "balanced" for classes
 * (i.e., containing roughly the same number of different classes for each batch)
 *
 * For a small datasets (e.g., 10,000 cases) or bigger datasets without much redundancy, consider using
 * 1. full-batch steepest descend
 * 2. adaptive learning rates, resilient back-propagation
 *
 * For big, redundant datasets, consider using mini-batches:
 * 1. try gradient descent with momentum
 * 2. try rmsprop without momentum
 */
public enum WeightUpdateMode {
   OnlineStochasticGradientDescend, // stochastic gradient descend, online version which updates weights after each case
   SteepestGradientDescend, // steepest gradient descend, full version which update weights after all training cases.
   MiniBatchGradientDescend // stochastic gradient descent, mini-batch version which update weights after a batch of training cases.
}
