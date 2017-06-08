# java-ann-mlp
Package provides java implementation of multi-layer perceptron neural network with back-propagation learning algorithm 

[![Build Status](https://travis-ci.org/chen0040/java-ann-mlp.svg?branch=master)](https://travis-ci.org/chen0040/java-ann-mlp) [![Coverage Status](https://coveralls.io/repos/github/chen0040/java-ann-mlp/badge.svg?branch=master)](https://coveralls.io/github/chen0040/java-ann-mlp?branch=master) 


# Features

* Regression + Classification
* Stochastic / Mini-batch / Steepest Descend Weight Update approaches
* Back-propagation / Resilient Back-propagation (rprop) / rmsprop
* Adaptive learning rate for individual weights
* Both numerical and categorical inputs

# Install

Add the following dependency to your POM file:

```xml
<dependency>
  <groupId>com.github.chen0040</groupId>
  <artifactId>java-ann-mlp</artifactId>
  <version>1.0.5</version>
</dependency>
```

# Usage

### MLP Regression

The training of mlp regression is done by calling:

```java
mlpRegression.fit(trainingData);
```

The regression of mlp is done by calling:

```java
double predicted = mlpRegression.transform(dataRow);
```

The sample code below shows how to use the MLP regression to predict the (-1, +1) numerical output of the heart-scale sample from libsvm.

The training data is loaded from a data frame connected to the "heart_scale" libsvm file (please refer to [here](https://github.com/chen0040/java-data-frame) for more example on how to create a data frame).
 
```java
InputStream inputStream = new FileInputStream("heart_scale");

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
```

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

As the heart-scale data frame has (-1, +1) numerical output, the codes first coverts the (-1, +1) as categorical output label "category-label".
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

# Configuration

### How often to update weights

By default the weight updating is done via online stochastic gradient descend, to change it to steepest gradient descent:

```java
mlpClassifier.setWeightUpdateMode(WeightUpdateMode.SteepestGradientDescend);
```

```java
regression.setWeightUpdateMode(WeightUpdateMode.SteepestGradientDescend);
```

To change it to mini-batch gradient descent:

```java
mlpClassifier.setWeightUpdateMode(WeightUpdateMode.MiniBatchGradientDescend);
```

```java
regression.setWeightUpdateMode(WeightUpdateMode.MiniBatchGradientDescend);
```

By default the mini-batch size is set to 50, this can be changed by calling:

```java
mlpClassifier.setMiniBatchSize(51);
```

```java
regression.setMiniBatchSize(51);
```

### How much to update weight

The default is by back-propagation, to change to other learning method such as rprop:

```java
mlpClassifier.setLearningMethod(LearningMethod.ResilientBackPropagation);
```

### Adaptive learning rate

The mlp can adapt the learning for individual weight, by default this is not enabled, to enable adaptive learning rate:

```java
mlpClassifier.enableAdaptiveLearningRate(true);
```

```java
regression.enableAdaptiveLearningRate(true);
```

To prevent the learning rate to increase out of bounds, we can also set the max learning rate (default is 1):

```java
mlpClassifier.setMaxLearningRate(1.1);
```

