package com.github.chen0040.mlp.ann;

import com.github.chen0040.mlp.functions.TransferFunction;
import lombok.Getter;
import lombok.Setter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;


//default network assumes input and output are in the range of [0, 1]
public class MLPNet {
	protected MLPLayer inputLayer =null;
	public MLPLayer outputLayer =null;

	protected List<MLPLayer> hiddenLayers;

	private static final Logger logger = LoggerFactory.getLogger(MLPNet.class);

	@Getter
	@Setter
	protected double learningRate =0.25; //learning rate



	public MLPLayer createInputLayer(int dimension){
		inputLayer = new MLPLayer(dimension);
		return inputLayer;
	}

	public MLPLayer createOutputLayer(int dimension){
		outputLayer = new MLPLayer(dimension);
		return outputLayer;
	}



	
	public MLPNet()
	{
		hiddenLayers = new ArrayList<>();
	}



	public void addHiddenLayer(int neuron_count)
	{
		MLPLayer layer=new MLPLayer(neuron_count);
		hiddenLayers.add(layer);
	}
	
	public void addHiddenLayer(int neuron_count, TransferFunction transfer_function)
	{
		MLPLayer layer=new MLPLayer(neuron_count);
		layer.setTransfer(transfer_function);
		hiddenLayers.add(layer);
	}
	
	public double train(double[] input, double[] target)
	{
		//forward propagate
		double[] propagated_output = inputLayer.setOutput(input);
		for(int i=0; i < hiddenLayers.size(); ++i) {
			propagated_output = hiddenLayers.get(i).forward_propagate(propagated_output);
		}
		propagated_output = outputLayer.forward_propagate(propagated_output);


		double error = get_target_error(target);

		
		//backward propagate
		double[] propagated_error = outputLayer.back_propagate(minus(target, propagated_output));
		for(int i = hiddenLayers.size()-1; i >= 0; --i){
			propagated_error = hiddenLayers.get(i).back_propagate(propagated_error);
		}

		//adjust weights
		double[] input2 = inputLayer.output();
		for(int i = 0; i < hiddenLayers.size(); ++i){
			hiddenLayers.get(i).adjust_weights(input2, getLearningRate());
			input2 = hiddenLayers.get(i).output();
		}
		outputLayer.adjust_weights(input2, getLearningRate());

		
		return error; 
	}

	public double[] minus(double[] a, double[] b){
		double[] c = new double[a.length];
		for(int i=0; i < a.length; ++i){
			c[i] = a[i] - b[i];
		}
		return c;
	}

	
	protected double get_target_error(double[] target)
	{
		double t_error=0;
		double error=0;
		double[] output = outputLayer.output();
		for(int i=0; i< output.length; i++)
		{
			error = target[i] - output[i];
			t_error+=(0.5 * error * error);
		}
		
		return t_error;
	}
	
	public double[] transform(double[] input)
	{
		double[] propagated_output = inputLayer.setOutput(input);
		for(int i=0; i < hiddenLayers.size(); ++i) {
			propagated_output = hiddenLayers.get(i).forward_propagate(propagated_output);
		}
		propagated_output = outputLayer.forward_propagate(propagated_output);

		return propagated_output;
	}
	
}
