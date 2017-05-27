# java-ann-mlp
Package provides java implementation of multi-layer perceptron neural network with back-propagation learning algorithm 


# Usage

### MLP Multi-Class Classifier
The training of mlp classifier is done by calling:

```java
mlpClassifier.fit(trainingData);
```

The classification of mlp is done by calling:

```java
String predicted = mlpClassifier.classify(dataRow);
```

The sample code below shows how to use the MLP classifier to predict the labels of the heart-scale sample from libsvm.

The training data is loaded from a data frame connected to the "heart_scale" libsvm file (please refer to [here](https://github.com/chen0040/java-data-frame) for more example on how to create a data frame).
 
```java
InputStream inputStream = new FileInputStream("heart_scale");

DataFrame dataFrame = DataQuery.libsvm().from(inputStream).build();

dataFrame.unlock();
for(int i=0; i < dataFrame.rowCount(); ++i){
 DataRow row = dataFrame.row(i);
 row.setCategoricalTargetCell("category-label", "" + row.target());
}
dataFrame.lock();

MLPClassifier mlpClassifier = new MLPClassifier();
mlpClassifier.setHiddenLayers(6); // one hidden layer, to set two or more hidden layer call mlpClassifier.setHiddenLayer([layer1NeuronCunt], [layer2NeuronCunt], ...);
mlpClassifier.setEpoches(1000);
mlpClassifier.setLearningRate(0.2);
mlpClassifier.fit(dataFrame);

int correctnessCount = 0;
for(int i = 0; i < dataFrame.rowCount(); ++i){
 DataRow row = dataFrame.row(i);



 String predicted_label = mlpClassifier.classify(row);
 correctnessCount += (predicted_label.equals(row.categoricalTarget()) ? 1 : 0);

 if(i < 10) {
    System.out.println(row);
    System.out.println("predicted: " + predicted_label + "\tactual: " + row.categoricalTarget());
 }
}

System.out.println("Prediction Accuracy: "+(correctnessCount * 100 / dataFrame.rowCount()));
```

