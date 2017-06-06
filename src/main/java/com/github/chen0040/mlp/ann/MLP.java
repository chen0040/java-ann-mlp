package com.github.chen0040.mlp.ann;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.data.utils.transforms.Standardization;
import com.github.chen0040.mlp.enums.WeightUpdateMode;
import com.github.chen0040.mlp.functions.RangeScaler;

import java.util.ArrayList;
import java.util.List;


/**
 * Created by xschen on 21/8/15.
 */
public abstract class MLP extends MLPNet {
    private Standardization inputNormalization;
    private RangeScaler outputNormalization;

    private boolean normalizeOutputs;

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
            List<double[]> targets = new ArrayList<>();
            for(int i = 0; i < batch.rowCount(); ++i){
                DataRow tuple = batch.row(i);
                if(isValidTrainingSample(tuple)) {
                    double[] target = getTarget(tuple);
                    targets.add(target);
                }
            }
            outputNormalization = new RangeScaler(targets);
        }


        for(int epoch=0; epoch < training_epoches; ++epoch)
        {
            if(weightUpdateMode == WeightUpdateMode.StochasticGradientDescend) {
                for (int i = 0; i < batch.rowCount(); i++) {
                    DataRow row = batch.row(i);
                    if (isValidTrainingSample(row)) {
                        double[] x = row.toArray();
                        x = inputNormalization.standardize(x);

                        double[] target = getTarget(row);

                        if (outputNormalization != null) {
                            target = outputNormalization.standardize(target);
                        }

                        stochasticGradientDescend(x, target);
                    }
                }
            } else {
                List<MLPLayer> backLayers = new ArrayList<>();
                backLayers.add(outputLayer);
                for(int index = hiddenLayers.size()-1; index >= 0; --index){
                    backLayers.add(hiddenLayers.get(index));
                }

                double[][][] dE_dwji = new double[backLayers.size()][][];
                double[][] dE_dwj0 = new double[backLayers.size()][];
                for(int index =0; index < backLayers.size(); ++index) {
                    MLPLayer layer = backLayers.get(index);
                    dE_dwji[index] = new double[layer.neurons.size()][];
                    dE_dwj0[index] = new double[layer.neurons.size()];
                    for(int j = 0; j < layer.neurons.size(); ++j) {
                        dE_dwji[index][j] = new double[layer.neurons.get(0).dimension()];
                    }
                }

                for (int rowIndex = 0; rowIndex < batch.rowCount(); rowIndex++) {
                    DataRow row = batch.row(rowIndex);
                    if (isValidTrainingSample(row)) {
                        double[] x = row.toArray();
                        x = inputNormalization.standardize(x);

                        double[] target = getTarget(row);

                        if (outputNormalization != null) {
                            target = outputNormalization.standardize(target);
                        }

                        double[] propagated_output = inputLayer.setOutput(x);
                        for(int i=0; i < hiddenLayers.size(); ++i) {
                            propagated_output = hiddenLayers.get(i).forward_propagate(propagated_output);
                        }
                        propagated_output = outputLayer.forward_propagate(propagated_output);

                        double[] dE_dyj = minus(target, propagated_output);
                        for(int index = 0; index < backLayers.size(); ++index){

                            MLPLayer layer = backLayers.get(index);

                            double[] dE_dzj = new double[dE_dyj.length];
                            for(int j = 0; j < dE_dzj.length; ++j) {
                                MLPNeuron neuron_j = layer.neurons.get(j);
                                double zj = neuron_j.getValue(neuron_j.values);
                                dE_dzj[j] = layer.getTransfer().gradient(zj) * dE_dyj[j];
                            }
                            int dimension = layer.neurons.get(0).dimension();
                            double[] dE_dyi = new double[dimension];
                            for(int i = 0; i < dimension; ++i) {
                                double sum = 0;
                                for(int j=0; j < dE_dzj.length; ++j) {
                                    double w_ij = layer.neurons.get(j).getWeight(i);
                                    sum += w_ij * dE_dzj[j];
                                }
                                dE_dyi[i] = sum;
                            }

                            for(int j = 0; j < dE_dzj.length; ++j) {
                                for(int i=0; i < dimension; ++i) {
                                    double yi = layer.neurons.get(j).values[i];
                                    dE_dwji[index][j][i] += yi * dE_dzj[j];
                                }
                                dE_dwj0[index][j] += dE_dzj[j];
                            }



                            dE_dyj = dE_dyi;
                        }
                    }
                }

                for(int index = 0; index < backLayers.size(); ++index){
                    MLPLayer layer = backLayers.get(index);
                    List<MLPNeuron> neurons = layer.neurons;
                    for(int j = 0; j < layer.neurons.size(); ++j) {
                        int dimension = neurons.get(0).dimension();

                        for(int i=0; i < dimension; ++i){
                            double wij = layer.neurons.get(j).getWeight(i);

                            double dwij = getLearningRate() * dE_dwji[index][j][i] / batch.rowCount();
                            layer.neurons.get(j).setWeight(i, wij + dwij);
                        }
                    }

                    for(int j=0; j < neurons.size(); j++)
                    {
                        MLPNeuron neuron = neurons.get(j);
                        double sink_w0 = neuron.bias_weight;
                        sink_w0 += learningRate * dE_dwj0[index][j] / batch.rowCount();
                        neuron.bias_weight = sink_w0;
                    }
                }
            }

        }
    }

    public double[] transform(DataRow tuple){

        double[] x = tuple.toArray();
        x = inputNormalization.standardize(x);

        double[] target = transform(x);

        if(outputNormalization != null){
            target = outputNormalization.revert(target);
        }

        return target;
    }
}
