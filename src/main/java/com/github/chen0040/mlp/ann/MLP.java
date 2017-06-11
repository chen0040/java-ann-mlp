package com.github.chen0040.mlp.ann;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.data.utils.transforms.Standardization;
import com.github.chen0040.mlp.enums.LearningMethod;
import com.github.chen0040.mlp.enums.WeightUpdateMode;
import com.github.chen0040.mlp.functions.RangeScaler;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;


/**
 * Created by xschen on 21/8/15.
 */
@Getter
@Setter
public abstract class MLP extends MLPNet {
    private Standardization inputNormalization;
    private Standardization outputNormalization = new Standardization();

    private boolean adaptiveLearningRateEnabled = false;

    private boolean normalizeOutputs;

    public MLP(){
        super();
        normalizeOutputs = false;
    }

    public void enabledAdaptiveLearningRate(boolean enabled){
        adaptiveLearningRateEnabled = enabled;
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
            outputNormalization.fit(targets);
        }

        double[][][] dE_dwji_prev = null;
        double[][] dE_dwj0_prev = null;


        for(int epoch=0; epoch < training_epoches; ++epoch)
        {
            if(weightUpdateMode == WeightUpdateMode.OnlineStochasticGradientDescend) {
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



                int batchSize = batch.rowCount();
                if(weightUpdateMode == WeightUpdateMode.MiniBatchGradientDescend){
                    batchSize = miniBatchSize;
                }

                for (int batchStart = 0; batchStart < batch.rowCount(); batchStart+=batchSize) {
                    int actualBatchSize = 0;
                    double[][][] dE_dwji = new double[backLayers.size()][][];
                    double[][] dE_dwj0 = new double[backLayers.size()][];
                    for(int index =0; index < backLayers.size(); ++index) {
                        MLPLayer layer = backLayers.get(index);
                        dE_dwji[index] = new double[layer.size()][];
                        dE_dwj0[index] = new double[layer.size()];
                        for(int j = 0; j < layer.size(); ++j) {
                            dE_dwji[index][j] = new double[layer.inputDimension()];
                        }
                    }

                    actualBatchSize = batchSize;
                    for(int bIndex = 0; bIndex < batchSize; ++bIndex) {
                        int rowIndex = batchStart + bIndex;

                        if(rowIndex >= batch.rowCount()) {
                            actualBatchSize = bIndex;
                            break;
                        }

                        DataRow row = batch.row(rowIndex);
                        if (!isValidTrainingSample(row)) {
                            throw new RuntimeException("training input does not have target defined!");
                        }

                        double[] x = row.toArray();
                        x = inputNormalization.standardize(x);

                        double[] target = getTarget(row);

                        if (outputNormalization != null) {
                            target = outputNormalization.standardize(target);
                        }

                        double[] propagated_output = inputLayer.setOutput(x);
                        for (int layerIndex = 0; layerIndex < hiddenLayers.size(); ++layerIndex) {
                            propagated_output = hiddenLayers.get(layerIndex).forward_propagate(propagated_output);
                        }
                        propagated_output = outputLayer.forward_propagate(propagated_output);

                        double[] dE_dyj = minus(propagated_output, target);

                        for (int layerIndex = 0; layerIndex < backLayers.size(); ++layerIndex) {

                            MLPLayer layer = backLayers.get(layerIndex);
                            int dimension = layer.inputDimension();

                            double[] dE_dzj = new double[dE_dyj.length];
                            for (int j = 0; j < dE_dzj.length; ++j) {

                                dE_dzj[j] = layer.getTransfer().gradient(layer, j) * dE_dyj[j];
                            }

                            double[] dE_dyi = new double[dimension];
                            for (int i = 0; i < dimension; ++i) {
                                double sum = 0;
                                for (int j = 0; j < dE_dzj.length; ++j) {
                                    double w_ij = layer.get(j).getWeight(i);
                                    sum += w_ij * dE_dzj[j];
                                }
                                dE_dyi[i] = sum;
                            }

                            for (int j = 0; j < dE_dzj.length; ++j) {
                                for (int i = 0; i < dimension; ++i) {
                                    double yi = layer.get(j).inputs[i];
                                    dE_dwji[layerIndex][j][i] += yi * dE_dzj[j];
                                }
                                dE_dwj0[layerIndex][j] += dE_dzj[j];
                            }

                            dE_dyj = dE_dyi;
                        }
                    }

                    for(int index = 0; index < backLayers.size(); ++index){
                        MLPLayer layer = backLayers.get(index);
                        int dimension = layer.inputDimension();
                        for(int j = 0; j < layer.size(); ++j) {
                            for(int i=0; i < dimension; ++i){
                                double wji = layer.get(j).getWeight(i);
                                double gji = layer.get(j).getLearningRateGain(i);



                                double dwij = 0;

                                if(learningMethod == LearningMethod.BackPropagation){
                                    // adaptive learning rate
                                    if(adaptiveLearningRateEnabled && dE_dwji_prev != null){
                                        double grad_prod = dE_dwji_prev[index][j][i] * dE_dwji[index][j][i];
                                        if(grad_prod > 0) {
                                            gji = gji + 0.05;
                                        } else if(grad_prod < 0) {
                                            gji = gji * 0.95;
                                        }

                                        gji = Math.min(maxLearningRate / learningRate, gji);
                                        layer.get(j).setLearningRateGain(i, gji);
                                    }

                                    dwij = gji * dE_dwji[index][j][i] / actualBatchSize;
                                } else if(learningMethod == LearningMethod.ResilientBackPropagation){
                                    if(dE_dwji_prev != null) {
                                        double grad_prod = dE_dwji_prev[index][j][i] * dE_dwji[index][j][i];
                                        if (grad_prod > 0) {
                                            gji = gji * 1.2;
                                        }
                                        else if (grad_prod < 0) {
                                            gji = gji * 0.5;
                                        }

                                        gji = Math.min(50 / learningRate, gji);
                                        gji = Math.max(0.0000001 / learningRate, gji);

                                        layer.get(j).setLearningRateGain(-1, gji);
                                    }

                                    dwij = gji / 1000;
                                }

                                if(L2Penalty > 0){
                                    dwij += L2Penalty * wji;
                                }

                                // gradient descend
                                wji = wji - learningRate * dwij;

                                layer.get(j).setWeight(i, wji);
                            }
                        }

                        for(int j=0; j < layer.size(); j++)
                        {
                            MLPNeuron neuron = layer.get(j);
                            double wj0 = neuron.bias_weight;
                            double gji = layer.get(j).getLearningRateGain(-1);



                            double dwj0 = 0;

                            if(learningMethod == LearningMethod.BackPropagation){
                                // adaptive learning rate
                                if(adaptiveLearningRateEnabled && dE_dwj0_prev != null){
                                    double grad_prod = dE_dwj0_prev[index][j] * dE_dwj0[index][j];
                                    if(grad_prod > 0) {
                                        gji = gji + 0.05;
                                    } else  if(grad_prod < 0) {
                                        gji = gji * 0.95;
                                    }

                                    gji = Math.min(maxLearningRate / learningRate, gji);


                                    layer.get(j).setLearningRateGain(-1, gji);
                                }

                                dwj0 = gji * dE_dwj0[index][j] / actualBatchSize;
                            } else if(learningMethod == LearningMethod.ResilientBackPropagation){
                                if(dE_dwji_prev != null) {
                                    double grad_prod = dE_dwj0_prev[index][j] * dE_dwj0[index][j];
                                    if (grad_prod > 0) {
                                        gji = gji * 1.2;
                                    }
                                    else if (grad_prod < 0) {
                                        gji = gji * 0.5;
                                    }

                                    gji = Math.min(50 / learningRate, gji);
                                    gji = Math.max(0.0000001 / learningRate, gji);

                                    layer.get(j).setLearningRateGain(-1, gji);
                                }

                                dwj0 = gji / 1000;
                            }

                            if(L2Penalty > 0){
                                dwj0 += L2Penalty * wj0;
                            }

                            // gradient descend
                            wj0 = wj0 - learningRate * dwj0;

                            neuron.bias_weight = wj0;
                        }

                        if(weightConstraint > 0) {
                            layer.applyWeightConstraint(weightConstraint);
                        }
                    }



                    dE_dwj0_prev = dE_dwj0;
                    dE_dwji_prev = dE_dwji;
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
