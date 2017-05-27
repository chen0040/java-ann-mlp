package com.github.chen0040.mlp.ann.classifiers;

import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import lombok.Getter;
import lombok.Setter;

import java.util.*;
import java.util.function.Supplier;
import java.util.logging.Level;
import java.util.logging.Logger;


/**
 * Created by memeanalytics on 21/8/15.
 */
public class MLPClassifier implements Cloneable {
    private MLPWithLabelOutput mlp;

    private final Logger logger = Logger.getLogger(String.valueOf(MLPClassifier.class));


    public static final String HIDDEN_LAYER1 = "hiddenLayer1";
    public static final String HIDDEN_LAYER2 = "hiddenLayer2";
    public static final String HIDDEN_LAYER3 = "hiddenLayer3";
    public static final String HIDDEN_LAYER4 = "hiddenLayer4";
    public static final String HIDDEN_LAYER5 = "hiddenLayer5";
    public static final String HIDDEN_LAYER6 = "hiddenLayer6";
    public static final String HIDDEN_LAYER7 = "hiddenLayer7";

    private List<String> classLabels = new ArrayList<>();

    @Getter
    @Setter
    private int epoches = 1000;

    @Getter
    @Setter
    private double learningRate = 0.2;

    private Map<String, Integer> hiddenLayer = new HashMap<>();

    public void copy(MLPClassifier rhs2) throws CloneNotSupportedException {
        mlp = rhs2.mlp == null ? null : (MLPWithLabelOutput)rhs2.mlp.clone();
        if(mlp != null){
            mlp.classLabelsModel = this::getClassLabels;
        }
    }

    public List<String> getClassLabels(){
        return classLabels;
    }

    @Override
    public Object clone() throws CloneNotSupportedException {
        MLPClassifier clone = (MLPClassifier)super.clone();
        clone.copy(this);

        return clone;
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
        double[] target = mlp.predict(tuple.toArray());

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
            logger.log(Level.SEVERE, "predict failed due to label not found");
        }

        return getClassLabels().get(selected_index);
    }

    private void scan4ClassLabels(DataFrame batch){
        int m = batch.rowCount();
        HashSet<String> set = new HashSet<>();
        for(int i=0; i < m; ++i){
            DataRow tuple = batch.row(i);
            if(!tuple.getCategoricalColumnNames().isEmpty()) {
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

        mlp = new MLPWithLabelOutput();
        mlp.classLabelsModel = () -> getClassLabels();

        int dimension = batch.row(0).toArray().length;

        List<Integer> hiddenLayers = parseHiddenLayers();

        mlp.setLearningRate(learningRate);
        mlp.createInputLayer(dimension);
        for (int hiddenLayerNeuronCount : hiddenLayers){
            mlp.addHiddenLayer(hiddenLayerNeuronCount);
        }
        mlp.createOutputLayer(getClassLabels().size());

        mlp.train(batch, epoches);
    }

}
