package com.github.chen0040.mlp.functions;


import com.github.chen0040.mlp.ann.MLPLayer;
import com.github.chen0040.mlp.ann.MLPNeuron;


/**
 * Created by xschen on 21/8/15.
 */
public interface TransferFunction {
    double calculate(MLPLayer layer, int j);

    double gradient(MLPLayer layer, int j);
}
