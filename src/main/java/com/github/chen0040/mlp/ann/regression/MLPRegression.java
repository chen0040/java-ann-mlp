package com.github.chen0040.mlp.ann.regression;

import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.mlp.functions.LogSig;
import com.github.chen0040.mlp.functions.TransferFunction;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;


/**
 * Created by xschen on 21/8/15.
 */
public class MLPRegression {
    private MLPWithNumericOutput mlp;

    @Getter
    @Setter
    private int epoches = 1000;
    private List<Integer> hiddenLayers;

    @Getter
    @Setter
    private double learningRate = 0.2;

    public void copy(MLPRegression that) throws CloneNotSupportedException {
        mlp = that.mlp == null ? null : (MLPWithNumericOutput)that.mlp.clone();
        epoches = that.epoches;
        hiddenLayers.clear();
        for(int i=0; i < that.hiddenLayers.size(); ++i){
            hiddenLayers.add(that.hiddenLayers.get(i));
        }
        learningRate = that.learningRate;
    }

    @Override
    public Object clone() throws CloneNotSupportedException {
        MLPRegression clone = (MLPRegression)super.clone();
        clone.copy(this);
        return clone;
    }

    public MLPRegression(){
        epoches = 1000;

        learningRate = 0.2;
        hiddenLayers = new ArrayList<>();
        hiddenLayers.add(6);
    }

    public List<Integer> getHiddenLayers() {
        return hiddenLayers;
    }

    public void setHiddenLayers(int... hiddenLayers) {
        this.hiddenLayers = new ArrayList<>();
        for(int hiddenLayerNeuronCount : hiddenLayers){
            this.hiddenLayers.add(hiddenLayerNeuronCount);
        }
    }

    public double transform(DataRow tuple) {
        double[] target = mlp.transform(tuple);
        return target[0];
    }

    public void fit(DataFrame batch) {

        mlp = new MLPWithNumericOutput();
        mlp.setNormalizeOutputs(true);

        TransferFunction transferFunction = new LogSig();


        int dimension = batch.row(0).toArray().length;

        mlp.setLearningRate(learningRate);
        mlp.createInputLayer(dimension);
        for (int hiddenLayerNeuronCount : hiddenLayers){
            mlp.addHiddenLayer(hiddenLayerNeuronCount, transferFunction);
        }
        mlp.createOutputLayer(1);
        mlp.outputLayer.setTransfer(transferFunction);

        mlp.train(batch, epoches);
    }
}
