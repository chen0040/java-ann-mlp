package com.github.chen0040.mlp.functions;

/**
 * Created by xschen on 21/8/15.
 */
public interface TransferFunction extends Cloneable {
    double calculate(double x);

    double gradient(double hx, double y);
}
