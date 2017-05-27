package com.github.chen0040.mlp.ann;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.data.utils.transforms.Standardization;

import java.util.ArrayList;
import java.util.List;


/**
 * Created by memeanalytics on 21/8/15.
 */
public abstract class MLP extends MLPNet {
    private Standardization inputNormalization;
    private Standardization outputNormalization;

    private boolean normalizeOutputs;


    public void copy(MLPNet rhs) throws CloneNotSupportedException {
        super.copy(rhs);

        MLP rhs2 = (MLP)rhs;
        inputNormalization = rhs2.inputNormalization == null ? null : (Standardization)rhs2.inputNormalization.clone();
        outputNormalization = rhs2.outputNormalization == null ? null : (Standardization)rhs2.outputNormalization.clone();
        normalizeOutputs = rhs2.normalizeOutputs;
    }

    public MLP(){
        super();
        normalizeOutputs = false;
    }

    protected abstract boolean isValidTrainingSample(DataRow tuple);

    public void setNormalizeOutputs(boolean normalize){
        this.normalizeOutputs = normalize;
    }

    public abstract double[] getTarget(DataRow tuple);


    public void train(DataFrame batch, int training_epoches)
    {
        inputNormalization = new Standardization(batch);


        if(normalizeOutputs) {
            List<double[]> targets = new ArrayList<double[]>();
            for(int i = 0; i < batch.rowCount(); ++i){
                DataRow tuple = batch.row(i);
                if(isValidTrainingSample(tuple)) {
                    double[] target = getTarget(tuple);
                    targets.add(target);
                }
            }
            outputNormalization = new Standardization(targets);
        }

        for(int count=0; count<training_epoches; ++count)
        {
            for(int i = 0; i<batch.rowCount(); i++)
            {
                DataRow row = batch.row(i);
                if(isValidTrainingSample(row)) {
                    double[] x = row.toArray();
                    x = inputNormalization.standardize(x);

                    double[] target = getTarget(row);

                    if (outputNormalization != null) {
                        target = outputNormalization.standardize(target);
                    }

                    train(x, target);
                }
            }
        }
    }

    public double[] predict(DataFrame context, DataRow tuple){

        double[] x = tuple.toArray();
        x = inputNormalization.standardize(x);

        double[] target = predict(x);

        if(outputNormalization != null){
            target = outputNormalization.revert(target);
        }

        return target;
    }
}
