package com.github.chen0040.mlp.ann;

public class MLPTrainParam {
	public MLPTrainParam(int max_epoches, int training_epoches, int max_stagnation_epoches, double min_test_error)
	{
		this.min_test_error=min_test_error;
		this.max_epoches=max_epoches;
		this.max_stagnation_epoches=max_stagnation_epoches;
		this.training_epoches=training_epoches;
	}
	
	public MLPTrainParam()
	{
		
	}
	
	public double min_test_error=0.01;
	public int max_epoches=2000;
	public int max_stagnation_epoches=2000;
	public int training_epoches=1;
}
