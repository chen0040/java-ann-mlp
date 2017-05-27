package com.github.chen0040.mlp.ann.regression;


import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataQuery;
import com.github.chen0040.data.frame.Sampler;
import com.github.chen0040.data.utils.TupleTwo;
import com.github.chen0040.mlp.utils.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

import java.io.InputStream;
import java.util.Random;

import static org.testng.Assert.*;


/**
 * Created by xschen on 27/5/2017.
 */
public class MLPRegressionUnitTest {
   private static final Logger logger = LoggerFactory.getLogger(MLPRegressionUnitTest.class);

   private static Random random = new Random();

   public static double rand(){
      return random.nextDouble();
   }


   public static double randn(){
      double u1 = rand();
      double u2 = rand();
      double r = Math.sqrt(-2.0 * Math.log(u1));
      double theta = 2.0 * Math.PI * u2;
      return r * Math.sin(theta);
   }

   @Test
   public void testSimple() {
      InputStream inputStream = FileUtils.getResource("heart_scale");

      DataFrame dataFrame = DataQuery.libsvm().from(inputStream).build();

      System.out.println(dataFrame.head(10));

      TupleTwo<DataFrame, DataFrame> miniFrames = dataFrame.shuffle().split(0.9);

      DataFrame trainingData = miniFrames._1();
      DataFrame crossValidationData = miniFrames._2();

      MLPRegression regression = new MLPRegression();
      regression.setHiddenLayers(8);
      regression.setEpoches(1000);
      regression.fit(trainingData);

      for(int i = 0; i < crossValidationData.rowCount(); ++i){
         double predicted = regression.transform(crossValidationData.row(i));
         double actual = crossValidationData.row(i).target();
         logger.info("predicted: {}\texpected: {}", predicted, actual);
      }


   }
}
