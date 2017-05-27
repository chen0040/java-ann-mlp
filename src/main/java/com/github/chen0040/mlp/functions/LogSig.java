package com.github.chen0040.mlp.functions;



public class LogSig extends AbstractTransferFunction
{
	@Override
	public double calculate(double x)
	{
		return 1/(Math.exp(-x)+1);
	}

	@Override
	public Object clone(){
		return new LogSig();
	}
}
