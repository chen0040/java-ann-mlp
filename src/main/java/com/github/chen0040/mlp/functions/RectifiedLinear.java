package com.github.chen0040.mlp.functions;


/**
 * Created by xschen on 31/5/2017.
 */
public class RectifiedLinear extends AbstractTransferFunction {
   @Override public double gradient(double hx, double y) {
      if(hx > 0) return 1;
      return 0;
   }


   @Override public double calculate(double x) {
      if(x > 0) return x;
      return 0;
   }
}
