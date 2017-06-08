package com.github.chen0040.mlp.ann;


import com.github.chen0040.mlp.functions.Sigmoid;
import com.github.chen0040.mlp.functions.TransferFunction;

import java.util.ArrayList;
import java.util.List;


//default network assumes input and output are in the range of [0, 1]
public class MLPLayer {
	private TransferFunction transfer = new Sigmoid();
    private final List<MLPNeuron> neurons = new ArrayList<>();

    public MLPNeuron get(int j){
        return neurons.get(j);
    }

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

	public double[] forward_propagate(double[] inputs)
	{
        double[] output = new double[neurons.size()];
        for(int i=0; i< neurons.size(); i++) {
            MLPNeuron neuron = neurons.get(i);
            neuron.setInputs(inputs);
        }

        for(int i=0; i < neurons.size(); ++i) {
            MLPNeuron neuron = neurons.get(i);
            output[i] = transfer.calculate(this, i);
            neuron.output = output[i];
        }

        return output;
	}
	
	protected void adjust_weights(double learningRate, double lambda, double weightConstraint)
	{
        for(int j=0; j< neurons.size(); j++)
        {
            MLPNeuron neuron_j = neurons.get(j);
            int dimension = neuron_j.inputDimension();
            for(int i=0; i < dimension; ++i) {

                double dE_dzj = neuron_j.dE_dzj;

                double w_ji = neuron_j.getWeight(i);

                double yi = neuron_j.inputs[i];
                double dw = dE_dzj * yi;

                if(lambda > 0){
                    dw += lambda * w_ji;
                }

                // gradient descend
                w_ji = w_ji - learningRate * dw;

                neuron_j.setWeightDelta(i, - learningRate * dw);
                neuron_j.setWeight(i, w_ji);
            }
        }

        for(int j=0; j < neurons.size(); j++)
        {
            MLPNeuron neuron = neurons.get(j);
            double w_j0 = neuron.bias_weight;
            double dE_dzj = neuron.dE_dzj;
            double yi = 1;

            double d_w0 = yi * dE_dzj;

            if(lambda > 0){
                d_w0 += lambda * w_j0;
            }

            // gradient descend
            w_j0 = w_j0 - learningRate * d_w0;

            neuron.bias_weight = w_j0;
        }

        if(weightConstraint > 0) {
            this.applyWeightConstraint(weightConstraint);
        }
	}

	public void applyWeightConstraint(double weightConstraint){
        for(int j=0; j < neurons.size(); ++j){
            neurons.get(j).applyWeightConstraint(weightConstraint);
        }
    }

    private int dimension(){
        return neurons.get(0).inputDimension();
    }
	
	public double[] back_propagate(double[] dE_dyj)
	{
        for(int j=0; j< neurons.size(); j++)
        {
            MLPNeuron neuron_j = neurons.get(j);
            neuron_j.dE_dzj = transfer.gradient(this, j) * dE_dyj[j];
        }

        int dimension = dimension();
        double[] dE_dyi = new double[dimension];
        for(int i = 0; i < dimension; ++i) {
            double error_sum = 0;

            for (int j = 0; j < neurons.size(); j++) {
                MLPNeuron neuron = neurons.get(j);
                double w_ji = neuron.getWeight(i);
                double dE_dzj = neuron.dE_dzj;
                error_sum += (w_ji * dE_dzj);
            }
            dE_dyi[i] = error_sum;
        }

        return dE_dyi;
	}


    public int size() {
        return neurons.size();
    }

    public int inputDimension(){
        return neurons.get(0).inputDimension();
    }
}
