package com.github.chen0040.mlp.ann.regression;

import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;

import java.util.ArrayList;
import java.util.List;


/**
 * Created by memeanalytics on 21/8/15.
 */
public class MLPRegression {
    private MLPWithNumericOutput mlp;
    private int epoches;
    private List<Integer> hiddenLayers;
    private double learningRate;

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
        this.hiddenLayers = new ArrayList<Integer>();
        for(int hiddenLayerNeuronCount : hiddenLayers){
            this.hiddenLayers.add(hiddenLayerNeuronCount);
        }
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public int getEpoches() {
        return epoches;
    }

    public void setEpoches(int epoches) {
        this.epoches = epoches;
    }

    public double transform(DataRow tuple) {
        double[] target = mlp.predict(tuple.toArray());
        return target[0];
    }

    public void fit(DataFrame batch) {

        mlp = new MLPWithNumericOutput();
        mlp.setNormalizeOutputs(true);

        int dimension = batch.row(0).toArray().length;

        mlp.setLearningRate(learningRate);
        mlp.createInputLayer(dimension);
        for (int hiddenLayerNeuronCount : hiddenLayers){
            mlp.addHiddenLayer(hiddenLayerNeuronCount);
        }
        mlp.createOutputLayer(1);

        mlp.train(batch, epoches);
    }
}
