package com.github.chen0040.mlp.functions;


import com.github.chen0040.mlp.ann.MLPLayer;
import com.github.chen0040.mlp.ann.MLPNeuron;


/**
 * Created by xschen on 6/6/2017.
 */
public class SoftMax extends AbstractTransferFunction {
   @Override public double calculate(MLPLayer layer, int J) {

      double sum = 0;
      double exp_J = 0;
      for(int j = 0; j < layer.size(); ++j){
         MLPNeuron neuron_j = layer.get(j);
         double exp_j = Math.exp(neuron_j.aggregate());
         sum += exp_j;
         if(j == J){
            exp_J = exp_j;
         }
      }
      return exp_J /sum;
   }


   @Override public double gradient(MLPLayer layer, int j) {
      double yj = calculate(layer, j);
      return yj * (1 - yj);
   }
}
