package com.github.chen0040.mlp.functions;


import com.github.chen0040.mlp.ann.MLPLayer;
import com.github.chen0040.mlp.ann.MLPNeuron;


/**
 * Created by xschen on 6/6/2017.
 * The soft-max activation function can be used as the output activation function
 * for the multi-class classification as it is a distribution of probability
 * across different output classes, which sums up to 1.
 * The corresponding cost function for the neural network that is suitable for the
 * soft-max output activation function is the cross-entropy cost function, which has
 * the following form:
 *
 * C = - sum_j (t_j * log(y_j))
 * where j is the index of an output class label
 *       t_j is 1 if j is the target class label; 0 otherwise
 *       y_j is the predicted value which the the value returns from from SoftMax.calculate() method for output index j
 *
 * C has the following nice properties
 * C has a very large big gradient when the target value is t_j is 1 and the predicted output y_j is almost zero
 * In other words, C has a very steep gradient if the predicted output is very wrong
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
