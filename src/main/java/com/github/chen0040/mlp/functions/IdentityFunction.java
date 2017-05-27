package com.github.chen0040.mlp.functions;


public class IdentityFunction extends AbstractTransferFunction
{
	@Override
	public double calculate(double x)
	{
		return x;
	}

	@Override
	public Object clone(){
		return new IdentityFunction();
	}
}
