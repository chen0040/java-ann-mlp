package com.github.chen0040.mlp.functions;

public class LogSig2 extends LogSig{
	@Override
	public double calculate(double x)
	{
		if(super.calculate(x) < 0.5)
		{
			return 0;
		}
		return 1;
	}

	@Override
	public Object clone(){
		return new LogSig2();
	}
}
