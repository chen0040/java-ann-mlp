package com.github.chen0040.mlp.enums;


/**
 * Created by xschen on 7/6/2017.
 *
 * For a small datasets (e.g., 10,000 cases) or bigger datasets without much redundancy, consider using
 * 1. full-batch steepest descend
 * 2. adaptive learning rates, resilient back-propagation
 *
 */
public enum LearningMethod {
   BackPropagation,
   ResilientBackPropagation
}
