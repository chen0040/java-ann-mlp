package com.github.chen0040.mlp.functions;


import com.github.chen0040.mlp.ann.MLPLayer;


/**
 * Created by xschen on 5/9/15.
 */
public abstract class AbstractTransferFunction implements TransferFunction {
    public abstract double calculate(MLPLayer layer, int j);
}
