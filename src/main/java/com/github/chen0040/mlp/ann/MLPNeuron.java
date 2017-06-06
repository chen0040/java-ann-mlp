package com.github.chen0040.mlp.ann;


import java.util.HashMap;
import java.util.Map;
import java.util.Random;


public class MLPNeuron {
	double bias_weight = 0;
	private double bias = 0;

	double output = 0;
	double error = 0;

	double[] values = null;

	private static Random rand = new Random();

    private Map<Integer, Double> weights;
    private Map<Integer, Double> weightDeltas;

    private int dimension;

    public int dimension(){
        return dimension;
    }

    public void copy(MLPNeuron rhs){
        bias = rhs.bias;
        bias_weight = rhs.bias_weight;
        output = rhs.output;
        error = rhs.error;
        weights.clear();
        weightDeltas.clear();

        this.dimension = rhs.dimension;

        for(Integer i : rhs.weights.keySet()){
            weights.put(i, rhs.weights.get(i));
        }
        for(Integer i : rhs.weightDeltas.keySet()){
            weightDeltas.put(i, rhs.weightDeltas.get(i));
        }
    }

    @Override
    public Object clone(){
        MLPNeuron clone = new MLPNeuron(dimension);
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
        weightDeltas.put(index, val);
    }

    public void setWeight(int index, double val){
        weights.put(index, val);
    }
	
	public MLPNeuron(int dimension)
	{
	    this.dimension = dimension;
		bias_weight =rand.nextDouble()-0.5;
		bias =-1;

        weights = new HashMap<>();
        weightDeltas = new HashMap<>();
	}
	
	double getValue(double[] x)
	{
	   values = x;
		double sum=0;

		for(int i=0; i < x.length; i++)
		{
			sum+=(x[i] * getWeight(i));
			
		}
		sum+=(bias_weight * bias);
		return sum;
	}
}
