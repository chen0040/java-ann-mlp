package com.github.chen0040.mlp.functions;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;


/**
 * Created by memeanalytics on 5/9/15.
 */
public abstract class AbstractTransferFunction implements TransferFunction {
    public abstract double calculate(double x);

    @Override
    public Object clone(){
        throw new NotImplementedException();
    }
}
