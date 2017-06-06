package com.github.chen0040.mlp.functions;


import com.github.chen0040.mlp.ann.MLPLayer;
import com.github.chen0040.mlp.ann.MLPNeuron;


public class Sigmoid extends AbstractTransferFunction
{
	@Override
	public double calculate(MLPLayer layer, int j)
	{
		MLPNeuron neuron = layer.get(j);
		double z = neuron.aggregate();
		return 1/(Math.exp(-z)+1);
	}


	@Override public double gradient(MLPLayer layer, int j) {
		double y = calculate(layer, j);
		return y * (1-y);
	}
}
