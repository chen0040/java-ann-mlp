package com.github.chen0040.mlp.ann.classifiers;

import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.mlp.enums.WeightUpdateMode;
import com.github.chen0040.mlp.functions.Sigmoid;
import com.github.chen0040.mlp.functions.SoftMax;
import com.github.chen0040.mlp.functions.TransferFunction;
import lombok.Getter;
import lombok.Setter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;


/**
 * Created by xschen on 21/8/15.
 */
public class MLPClassifier {
    private MLPWithLabelOutput mlp;

    private final Logger logger = LoggerFactory.getLogger(MLPClassifier.class);


    public static final String HIDDEN_LAYER1 = "hiddenLayer1";
    public static final String HIDDEN_LAYER2 = "hiddenLayer2";
    public static final String HIDDEN_LAYER3 = "hiddenLayer3";
    public static final String HIDDEN_LAYER4 = "hiddenLayer4";
    public static final String HIDDEN_LAYER5 = "hiddenLayer5";
    public static final String HIDDEN_LAYER6 = "hiddenLayer6";
    public static final String HIDDEN_LAYER7 = "hiddenLayer7";

    @Setter
    protected WeightUpdateMode weightUpdateMode = WeightUpdateMode.StochasticGradientDescend;

    private boolean adaptiveLearningRateEnabled = false;

    public void enabledAdaptiveLearningRate(boolean enabled){
        adaptiveLearningRateEnabled = enabled;
    }

    private List<String> classLabels = new ArrayList<>();

    @Setter
    private int miniBatchSize = 50;

    @Getter
    @Setter
    private int epoches = 1000;

    @Getter
    @Setter
    private double learningRate = 0.2;

    @Getter
    @Setter
    private TransferFunction hiddenLayerTransfer = new Sigmoid();

    // soft-max is suitable for multi-class classification
    @Getter
    @Setter
    private TransferFunction outputLayerTransfer = new SoftMax();

    private Map<String, Integer> hiddenLayer = new HashMap<>();

    public List<String> getClassLabels(){
        return classLabels;
    }

    public MLPClassifier(){
        setHiddenLayers(6);
    }

    public List<Integer> getHiddenLayers() {
        return parseHiddenLayers();
    }

    private String hiddenLayerName(int i){
        String hiddenLayerName = HIDDEN_LAYER7;
        switch(i){
            case 0:
                hiddenLayerName = HIDDEN_LAYER1;
                break;
            case 1:
                hiddenLayerName = HIDDEN_LAYER2;
                break;
            case 2:
                hiddenLayerName = HIDDEN_LAYER3;
                break;
            case 3:
                hiddenLayerName = HIDDEN_LAYER4;
                break;
            case 4:
                hiddenLayerName = HIDDEN_LAYER5;
                break;
            case 5:
                hiddenLayerName = HIDDEN_LAYER6;
                break;
            case 6:
                hiddenLayerName = HIDDEN_LAYER7;
                break;
        }
        return hiddenLayerName;
    }

    public void setHiddenLayers(int... hiddenLayers) {
        for(int i = 0; i < hiddenLayers.length; ++i){
            hiddenLayer.put(hiddenLayerName(i), hiddenLayers[i]);
        }
    }

    public String classify(DataRow tuple) {
        double[] target = mlp.transform(tuple.toArray());

        int selected_index = -1;
        double maxValue = Double.NEGATIVE_INFINITY;
        for(int i=0; i < target.length; ++i){
            double value = target[i];
            if(value > maxValue){
                maxValue = value;
                selected_index = i;
            }
        }

        if(selected_index==-1){
            logger.error("transform failed due to label not found");
        }

        return getClassLabels().get(selected_index);
    }

    private void scan4ClassLabels(DataFrame batch){
        int m = batch.rowCount();
        Set<String> set = new HashSet<>();
        for(int i=0; i < m; ++i){
            DataRow tuple = batch.row(i);
            if(!tuple.getCategoricalTargetColumnNames().isEmpty()) {
                set.add(tuple.categoricalTarget());
            }
        }

        List<String> labels = new ArrayList<>();
        for(String label : set){
            labels.add(label);
        }

        classLabels.clear();
        classLabels.addAll(labels);
    }

    private List<Integer> parseHiddenLayers(){
        List<Integer> hiddenLayers = new ArrayList<>();
        for(int i=0; i < 7; ++i){
            int neuronCount = getAttribute(hiddenLayerName(i));
            if(neuronCount > 0){
                hiddenLayers.add(neuronCount);
            }
        }
        return hiddenLayers;
    }

    private int getAttribute(String layerName) {
        return hiddenLayer.getOrDefault(layerName, 0);
    }

    public void fit(DataFrame batch) {

        if (getClassLabels().isEmpty()) {
            scan4ClassLabels(batch);
        }

        logger.info("class labels: {}", classLabels.size());

        mlp = new MLPWithLabelOutput();
        mlp.setWeightUpdateMode(weightUpdateMode);
        mlp.setMiniBatchSize(miniBatchSize);
        mlp.enabledAdaptiveLearningRate(adaptiveLearningRateEnabled);
        mlp.classLabelsModel = () -> getClassLabels();

        int dimension = batch.row(0).toArray().length;

        List<Integer> hiddenLayers = parseHiddenLayers();

        mlp.setLearningRate(learningRate);
        mlp.createInputLayer(dimension);
        for (int hiddenLayerNeuronCount : hiddenLayers){
            mlp.addHiddenLayer(hiddenLayerNeuronCount, hiddenLayerTransfer);
        }
        mlp.createOutputLayer(getClassLabels().size()).setTransfer(outputLayerTransfer);

        mlp.train(batch, epoches);
    }

}
