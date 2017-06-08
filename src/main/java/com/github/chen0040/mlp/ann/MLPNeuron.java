package com.github.chen0040.mlp.ann;


import java.util.HashMap;
import java.util.Map;
import java.util.Random;


public class MLPNeuron {
	double bias_weight = 0;
	private double bias = 0;

	double output = 0;
	double dE_dzj = 0;

	double[] inputs = null;

	private static Random rand = new Random();

    private Map<Integer, Double> weights;
    private Map<Integer, Double> learningRateGains; // local gain on the learning rate;

    private int inputDimension;

    public void setInputs(double[] inputs){
       this.inputs = inputs;
    }

    public int inputDimension(){
        return inputDimension;
    }

    public void copy(MLPNeuron rhs){
        bias = rhs.bias;
        bias_weight = rhs.bias_weight;
        output = rhs.output;
        dE_dzj = rhs.dE_dzj;
        weights.clear();
        learningRateGains.clear();

        this.inputDimension = rhs.inputDimension;

        for(Integer i : rhs.weights.keySet()){
            weights.put(i, rhs.weights.get(i));
        }
        for(Integer i : rhs.learningRateGains.keySet()){
            learningRateGains.put(i, rhs.learningRateGains.get(i));
        }
    }

    @Override
    public Object clone(){
        MLPNeuron clone = new MLPNeuron(inputDimension);
        clone.copy(this);
        return clone;
    }

    public double getWeight(int index){
        if(weights.containsKey(index)){
            return weights.get(index);
        }else{
            double weight = (rand.nextDouble() - 0.5) / 10;
            weights.put(index, weight);
            return weight;
        }
    }

    public void setWeightDelta(int index, double val){
        learningRateGains.put(index, val);
    }

    public void setWeight(int index, double val){
        weights.put(index, val);
    }

    public void setLearningRateGain(int index, double value) {
        learningRateGains.put(index, value);
    }

    public double getLearningRateGain(int index) {
        return learningRateGains.getOrDefault(index, 1.0);
    }


	
	public MLPNeuron(int inputDimension)
	{
	    this.inputDimension = inputDimension;
		bias_weight =rand.nextDouble()-0.5;
		bias =-1;

        weights = new HashMap<>();
        learningRateGains = new HashMap<>();
	}

	public double aggregate() {
       double sum=0;

       for(int i=0; i < inputs.length; i++)
       {
          sum+=(inputs[i] * getWeight(i));

       }
       sum+=(bias_weight * bias);
       return sum;
    }


    public void applyWeightConstraint(double weightConstraint) {
        double squared_length = 0;
        for(int i=0; i < inputDimension; ++i){
            squared_length += getWeight(i) * getWeight(i);
        }
        if(squared_length > weightConstraint){
            double ratio = Math.sqrt(weightConstraint / squared_length);
            for(int i=0; i < inputDimension; ++i) {
                setWeight(i, getWeight(i) * ratio);
            }
        }
    }
}
