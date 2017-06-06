package com.github.chen0040.mlp.ann;


import com.github.chen0040.mlp.functions.AbstractTransferFunction;
import com.github.chen0040.mlp.functions.Sigmoid;
import com.github.chen0040.mlp.functions.TransferFunction;

import java.util.ArrayList;
import java.util.List;


//default network assumes input and output are in the range of [0, 1]
public class MLPLayer {
	private TransferFunction transfer = new Sigmoid();
    final List<MLPNeuron> neurons = new ArrayList<>();

    public MLPLayer(int neuron_count, int dimension)
	{
		for(int i=0; i < neuron_count; i++)
		{
			neurons.add(new MLPNeuron(dimension));
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

    public void setTransfer(TransferFunction transfer) {
        this.transfer = transfer;
    }

    public TransferFunction getTransfer(){
        return this.transfer;
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
	
	protected void adjust_weights(double learningRate)
	{
        for(int j=0; j< neurons.size(); j++)
        {
            MLPNeuron neuron = neurons.get(j);
            int dimension = neuron.dimension();
            for(int i=0; i < dimension; ++i) {

                double sink_error = neuron.error;

                double weight = neuron.getWeight(i);

                double dw = learningRate * sink_error * neuron.values[i];
                weight += dw;
                neuron.setWeightDelta(i, dw);
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
            double[] values = neuron.values;
            double hx = neuron.getValue(values);

            neuron.error = transfer.gradient(hx) * error[i];
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
