package com.github.chen0040.mlp.ann.regression;


import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataQuery;
import com.github.chen0040.data.frame.Sampler;
import com.github.chen0040.data.utils.TupleTwo;
import com.github.chen0040.mlp.enums.WeightUpdateMode;
import com.github.chen0040.mlp.functions.RectifiedLinear;
import com.github.chen0040.mlp.utils.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

import java.io.InputStream;
import java.util.Random;


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

   @Test
   public void test_simple_regression() {
      DataQuery.DataFrameQueryBuilder schema = DataQuery.blank()
              .newInput("x1")
              .newInput("x2")
              .newOutput("y")
              .end();

      // y = 4 + 0.5 * x1 + 0.2 * x2
      Sampler.DataSampleBuilder sampler = new Sampler()
              .forColumn("x1").generate((name, index) -> randn() * 0.3 + index / 100.0)
              .forColumn("x2").generate((name, index) -> randn() * 0.3 + index * index / 10000.0)
              .forColumn("y").generate((name, index) -> 4 + 0.5 * index / 100.0 + 0.2 * index * index / 10000.0 + randn() * 0.3)
              .end();

      DataFrame data = schema.build();

      data = sampler.sample(data, 200);

      TupleTwo<DataFrame, DataFrame> frames = data.shuffle().split(0.9);

      DataFrame trainingData = frames._1();
      System.out.println(trainingData.head(10));

      DataFrame crossValidationData = frames._2();

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

   @Test
   public void test_simple_regression_steepest_gradient_descend() {
      DataQuery.DataFrameQueryBuilder schema = DataQuery.blank()
              .newInput("x1")
              .newInput("x2")
              .newOutput("y")
              .end();

      // y = 4 + 0.5 * x1 + 0.2 * x2
      Sampler.DataSampleBuilder sampler = new Sampler()
              .forColumn("x1").generate((name, index) -> randn() * 0.3 + index / 100.0)
              .forColumn("x2").generate((name, index) -> randn() * 0.3 + index * index / 10000.0)
              .forColumn("y").generate((name, index) -> 4 + 0.5 * index / 100.0 + 0.2 * index * index / 10000.0 + randn() * 0.3)
              .end();

      DataFrame data = schema.build();

      data = sampler.sample(data, 200);

      TupleTwo<DataFrame, DataFrame> frames = data.shuffle().split(0.9);

      DataFrame trainingData = frames._1();
      System.out.println(trainingData.head(10));

      DataFrame crossValidationData = frames._2();

      MLPRegression regression = new MLPRegression();
      regression.setWeightUpdateMode(WeightUpdateMode.SteepestGradientDescend);
      regression.setHiddenLayers(8);
      regression.setEpoches(1000);
      regression.fit(trainingData);

      for(int i = 0; i < crossValidationData.rowCount(); ++i){
         double predicted = regression.transform(crossValidationData.row(i));
         double actual = crossValidationData.row(i).target();
         logger.info("predicted: {}\texpected: {}", predicted, actual);
      }


   }

   @Test
   public void test_simple_regression_linear_rectifier_transfer() {
      DataQuery.DataFrameQueryBuilder schema = DataQuery.blank()
              .newInput("x1")
              .newInput("x2")
              .newOutput("y")
              .end();

      // y = 4 + 0.5 * x1 + 0.2 * x2
      Sampler.DataSampleBuilder sampler = new Sampler()
              .forColumn("x1").generate((name, index) -> randn() * 0.3 + index / 100.0)
              .forColumn("x2").generate((name, index) -> randn() * 0.3 + index * index / 10000.0)
              .forColumn("y").generate((name, index) -> 4 + 0.5 * index / 100.0 + 0.2 * index * index / 10000.0 + randn() * 0.3)
              .end();

      DataFrame data = schema.build();

      data = sampler.sample(data, 200);

      TupleTwo<DataFrame, DataFrame> frames = data.shuffle().split(0.9);

      DataFrame trainingData = frames._1();
      System.out.println(trainingData.head(10));

      DataFrame crossValidationData = frames._2();

      MLPRegression regression = new MLPRegression();
      regression.setTransferFunction(new RectifiedLinear());
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
