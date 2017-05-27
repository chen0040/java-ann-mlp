package com.github.chen0040.mlp.ann.classifiers;


import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.mlp.ann.MLP;
import com.github.chen0040.mlp.ann.MLPNet;

import java.util.List;
import java.util.function.Supplier;


/**
 * Created by memeanalytics on 5/9/15.
 */
public class MLPWithLabelOutput extends MLP {
    public Supplier<List<String>> classLabelsModel;

    @Override
    public boolean isValidTrainingSample(DataRow tuple){
        return !tuple.getCategoricalColumnNames().isEmpty();
    }

    @Override
    public double[] getTarget(DataRow tuple) {
        List<String> labels = classLabelsModel.get();
        double[] target = new double[labels.size()];
        for (int i = 0; i < labels.size(); ++i) {
            target[i] = labels.get(i).equals(tuple.categoricalTarget()) ? 1 : 0;
        }
        return target;
    }

    @Override
    public Object clone() throws CloneNotSupportedException {
        MLPWithLabelOutput clone = (MLPWithLabelOutput)super.clone();
        clone.copy(this);
        return clone;
    }

    @Override
    public void copy(MLPNet rhs) throws CloneNotSupportedException {
        super.copy(rhs);

        MLPWithLabelOutput rhs2 = (MLPWithLabelOutput)rhs;
        classLabelsModel = rhs2.classLabelsModel;
    }
}
