package com.github.chen0040.mlp.functions;


import com.github.chen0040.mlp.ann.MLPLayer;


/**
 * Created by xschen on 7/6/2017.
 */
public class Identity extends AbstractTransferFunction {
   @Override public double calculate(MLPLayer layer, int j) {
      return layer.get(j).aggregate();
   }


   @Override public double gradient(MLPLayer layer, int j) {
      return 1;
   }
}
