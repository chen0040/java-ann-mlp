package com.github.chen0040.mlp.ann.regression;

import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataQuery;
import com.github.chen0040.data.frame.Sampler;
import com.github.chen0040.data.utils.TupleTwo;
import com.github.chen0040.mlp.enums.WeightUpdateMode;
import com.github.chen0040.mlp.functions.RangeScaler;
import com.github.chen0040.mlp.utils.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

import java.io.InputStream;

public class TransparentConductorRegressionUnitTest {

    private static final Logger logger = LoggerFactory.getLogger(TransparentConductorRegressionUnitTest.class);

    @Test
    public void test_simple_regression_mini_batch_gradient_descend() {
        InputStream inputStream = FileUtils.getResource("train.csv");

        DataFrame dataFrame = DataQuery.csv(",").from(inputStream).skipRows(1)
                .selectColumn(1)
                .asCategory()
                .asInput("spacegroup")
                .selectColumn(2)
                .asNumeric()
                .asInput("number_of_total_atoms")
                .selectColumn(12)
                .asNumeric()
                .asOutput("formation_energy_ev_natom")
                .build();

        System.out.println(dataFrame.head(10));

        TupleTwo<DataFrame, DataFrame> frames = dataFrame.shuffle().split(0.9);

        DataFrame trainingData = frames._1();
        System.out.println(trainingData.head(10));

        DataFrame crossValidationData = frames._2();

        MLPRegression regression = new MLPRegression();
        regression.setWeightUpdateMode(WeightUpdateMode.MiniBatchGradientDescend);
        regression.setMiniBatchSize(20);
        regression.setOutputNormalization(new RangeScaler());
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
