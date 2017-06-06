package com.github.chen0040.mlp.functions;



public class Sigmoid extends AbstractTransferFunction
{
	@Override
	public double calculate(double z)
	{
		return 1/(Math.exp(-z)+1);
	}


	@Override public double gradient(double z) {
		double y = calculate(z);
		return y * (1-y);
	}
}
