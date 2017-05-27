package com.github.chen0040.mlp.ann.regression;


import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.mlp.ann.MLP;


/**
 * Created by xschen on 5/9/15.
 */
public class MLPWithNumericOutput extends MLP {

    @Override
    protected boolean isValidTrainingSample(DataRow tuple){
        return !tuple.getTargetColumnNames().isEmpty();
    }

    @Override
    public double[] getTarget(DataRow tuple) {
        double[] target = new double[1];
        target[0] = tuple.target();
        return target;
    }

    @Override
    public Object clone() throws CloneNotSupportedException {
        MLPWithNumericOutput clone = (MLPWithNumericOutput)super.clone();
        clone.copy(this);

        return clone;
    }
}
