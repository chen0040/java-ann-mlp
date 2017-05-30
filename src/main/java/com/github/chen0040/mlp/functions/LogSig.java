package com.github.chen0040.mlp.functions;



public class LogSig extends AbstractTransferFunction
{
	@Override
	public double calculate(double x)
	{
		return 1/(Math.exp(-x)+1);
	}


	@Override public double gradient(double hx, double y) {
		y = calculate(hx);
		return y * (1-y);
	}


	@Override
	public Object clone(){
		return new LogSig();
	}
}
