package com.github.chen0040.mlp.functions;

public class BiPolarLogSig extends AbstractTransferFunction {
	@Override
	public double calculate(double x)
	{
		return 2/(1+Math.exp(-x)) - 1;
	}

	@Override
	public Object clone(){
		return new BiPolarLogSig();
	}
}
