package com.github.chen0040.mlp.ann;


import com.github.chen0040.mlp.functions.AbstractTransferFunction;
import com.github.chen0040.mlp.functions.LogSig;
import com.github.chen0040.mlp.functions.TransferFunction;

import java.util.ArrayList;
import java.util.Random;


//default network assumes input and output are in the range of [0, 1]
public class MLPLayer implements Cloneable {
	private static Random rand = new Random();
	private TransferFunction transfer = new LogSig();
    private ArrayList<MLPNeuron> neurons;

    public void copy(MLPLayer rhs){
        transfer = rhs.transfer == null ? null : (TransferFunction) ((AbstractTransferFunction)rhs.transfer).clone();
        neurons.clear();
        for(int i=0; i < rhs.neurons.size(); ++i){
            neurons.add((MLPNeuron)rhs.neurons.get(i).clone());
        }
    }

    @Override
    public Object clone(){
        MLPLayer clone = new MLPLayer();
        clone.copy(this);

        return clone;
    }

    public MLPLayer(){
        neurons = new ArrayList<MLPNeuron>();
    }

    public MLPLayer(int neuron_count)
	{
        neurons = new ArrayList<>();
		for(int i=0; i<neuron_count; i++)
		{
			neurons.add(new MLPNeuron());
		}
	}

    public double[] output(){
        double[] output = new double[neurons.size()];
        for(int i=0; i < output.length; ++i){
            output[i] = neurons.get(i).output;
        }
        return output;
    }

    public double[] setOutput(double[] output){
        for(int i=0; i< neurons.size(); i++)
        {
            neurons.get(i).output = output[i];
            output[i] = output[i];
        }
        return output.clone();
    }

    public TransferFunction getTransfer() {
        return transfer;
    }

    public void setTransfer(TransferFunction transfer) {
        this.transfer = transfer;
    }

    public ArrayList<MLPNeuron> getNeurons() {
        return neurons;
    }

    public void setNeurons(ArrayList<MLPNeuron> neurons) {
        this.neurons = neurons;
    }
	
	public double[] forward_propagate(double[] input)
	{
        double[] output = new double[neurons.size()];
        for(int i=0; i< neurons.size(); i++)
        {
            MLPNeuron neuron= neurons.get(i);
            output[i] = transfer.calculate(neuron.getValue(input));
            neuron.output = output[i];
        }

        return output;
	}
	
	protected void adjust_weights(double[] input, double learningRate, double momentum)
	{
        for(int j=0; j< neurons.size(); j++)
        {
            MLPNeuron neuron = neurons.get(j);
            int dimension = neuron.dimension();
            for(int i=0; i < dimension; ++i) {

                double sink_error = neuron.error;
                double dWeight = neuron.getWeightDelta(i);
                double weight = neuron.getWeight(i);

                double dw = learningRate * sink_error * input[i];
                weight += (dw + momentum * dWeight);
                dWeight = dw;
                neuron.setWeightDelta(i, dWeight);
                neuron.setWeight(i, weight);
            }
        }

        for(int j=0; j < neurons.size(); j++)
        {
            MLPNeuron neuron = neurons.get(j);
            double sink_w0 = neuron.bias_weight;
            double sink_error = neuron.error;
            sink_w0 += learningRate * sink_error;
            neuron.bias_weight = sink_w0;
        }
	}

    private int dimension(){
        return neurons.get(0).dimension();
    }
	
	public double[] back_propagate(double[] error)
	{
        for(int i=0; i< neurons.size(); i++)
        {
            MLPNeuron neuron= neurons.get(i);
            double y = neuron.output;
            neuron.error = y * (1-y) * error[i];
        }

        int k = dimension();
        double[] propagated_error = new double[k];
        for(int i = 0; i < k; ++i) {
            double error_sum = 0;

            for (int j = 0; j < neurons.size(); j++) {
                MLPNeuron neuron = neurons.get(j);
                double weight = neuron.getWeight(i);
                double sink_error = neuron.error;
                error_sum += (weight * sink_error);
            }
            propagated_error[i] = error_sum;
        }

        return propagated_error;
	}
}
