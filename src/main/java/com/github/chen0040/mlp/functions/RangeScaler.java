package com.github.chen0040.mlp.functions;


import com.github.chen0040.data.utils.transforms.Standardization;
import lombok.Getter;
import lombok.Setter;

import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * Created by xschen on 30/5/2017.
 */
@Getter
@Setter
public class RangeScaler extends Standardization {

   private final Map<Integer, Double> minValue = new HashMap<>();
   private final Map<Integer, Double> maxValue = new HashMap<>();


   @Override
   public void fit(List<double[]> targets) {
      for(int i = 0; i < targets.size(); ++i){
         double[] values = targets.get(i);
         for(int j=0; j < values.length; ++j) {
            minValue.put(j, Math.min(minValue.getOrDefault(j, Double.MAX_VALUE), values[j]));
            maxValue.put(j, Math.max(maxValue.getOrDefault(j, Double.NEGATIVE_INFINITY), values[j]));
         }
      }
   }




   @Override
   public Object clone() throws CloneNotSupportedException {
      RangeScaler obj = (RangeScaler)super.clone();
      obj.getMaxValue().putAll(maxValue);
      obj.getMinValue().putAll(minValue);
      return obj;
   }


   @Override
   public double[] standardize(double[] target) {
      double[] result = new double[target.length];
      for(int i=0; i < result.length; ++i){
         result[i] = (target[i] - minValue.get(i)) / (maxValue.get(i) - minValue.get(i));
      }
      return result;
   }


   @Override
   public double[] revert(double[] target) {
      double[] result = new double[target.length];
      for(int i=0; i < result.length; ++i){
         result[i] = minValue.get(i) + target[i] * (maxValue.get(i) - minValue.get(i));
      }
      return result;
   }
}
