
// COMMAND ----------

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler

val seed = 5043
val Array(trainingData, testData) = weatherFeaturesLabelDF.randomSplit(Array(0.7, 0.3), seed)

// train Random Forest model with training data set
val randomForestClassifier = new RandomForestClassifier()
  .setImpurity("gini")
  .setMaxDepth(3)
  .setNumTrees(20)
  .setMaxBins(64)
  .setFeatureSubsetStrategy("auto")
  .setSeed(seed)
val randomForestModel = randomForestClassifier.fit(trainingData)
println(randomForestModel.toDebugString)

val predictionDf = randomForestModel.transform(testData)
predictionDf.show(10)

// COMMAND ----------



// COMMAND ----------

val Array(pipelineTrainingData, pipelineTestingData) = weatherDF4_train.randomSplit(Array(0.7, 0.3), seed)
val cols1 = Array("MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Pressure9am", "Humidity9am", "Humidity3pm")


val assembler = new VectorAssembler()
  .setInputCols(cols1)
  .setOutputCol("features")
val featureDf = assembler.transform(weatherDF4_train)
featureDf.printSchema()


// COMMAND ----------


val indexer = new StringIndexer()
  .setInputCol("RainTomorrow")
  .setOutputCol("label")
val labelDf = indexer.fit(featureDf).transform(featureDf)
labelDf.printSchema()



// COMMAND ----------

import org.apache.spark.ml.Pipeline

val stages = Array(assembler, indexer, randomForestClassifier)

val pipeline = new Pipeline().setStages(stages)
val pipelineModel = pipeline.fit(pipelineTrainingData)

// test model with test data
val pipelinePredictionDf = pipelineModel.transform(pipelineTestingData)
pipelinePredictionDf.show(10)



// COMMAND ----------

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator


// evaluate model with area under ROC
val evaluator = new BinaryClassificationEvaluator()
  .setLabelCol("label")
  .setMetricName("areaUnderROC")

// measure the accuracy
val accuracy = evaluator.evaluate(predictionDf)
println(accuracy)
/*
 * output
0.7092862889323067
*/

// measure the accuracy of pipeline model
val pipelineAccuracy = evaluator.evaluate(pipelinePredictionDf)
println(pipelineAccuracy)
/*
 * output
0.7317904773399297
*/

// COMMAND ----------

import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidator
trainingData.show(5)
// parameters that needs to tune, we tune
//  1. max buns
//  2. max depth
//  3. impurity
val paramGrid = new ParamGridBuilder()
  .addGrid(randomForestClassifier.maxBins, Array(42, 70))
  .addGrid(randomForestClassifier.maxDepth, Array(6, 10))
  .addGrid(randomForestClassifier.impurity, Array( "gini"))
  .build()

// define cross validation stage to search through the parameters
// K-Fold cross validation with BinaryClassificationEvaluator
val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(5)

/*
 * output
0.7574678655482457
*/

// COMMAND ----------


// fit will run cross validation and choose the best set of parameters
// this will take some time to run
val cvModel = cv.fit(pipelineTrainingData)


// COMMAND ----------


// test cross validated model with test data
val cvPredictionDf = cvModel.transform(pipelineTestingData)
cvPredictionDf.show(10)


// COMMAND ----------


// measure the accuracy of cross validated model
// this model is more accurate than the old model
val cvAccuracy = evaluator.evaluate(cvPredictionDf)
println(cvAccuracy)
