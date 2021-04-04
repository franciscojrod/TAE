import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.{PipelineModel, Pipeline}

val seed = 5043
val Array(trainingData, testData) = weatherFeaturesLabelDF.randomSplit(Array(0.7, 0.3), seed)

val Array(pipelineTrainingData, pipelineTestingData) = weatherDF4_train.randomSplit(Array(0.7, 0.3), seed)
val cols1 = Array("MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Pressure9am", "Humidity9am", "Humidity3pm")


val assembler = new VectorAssembler()
  .setInputCols(cols1)
  .setOutputCol("features")
val featureDf = assembler.transform(weatherDF4_train)
featureDf.printSchema()


val indexer = new StringIndexer()
  .setInputCol("RainTomorrow")
  .setOutputCol("label")
val labelDf = indexer.fit(featureDf).transform(featureDf)
labelDf.printSchema()

val randomForestClassifier = new RandomForestClassifier()

val stages = Array(assembler, indexer, randomForestClassifier)

val pipeline = new Pipeline().setStages(stages)
val pipelineModel = pipeline.fit(pipelineTrainingData)

// test model with test data
val pipelinePredictionDf = pipelineModel.transform(pipelineTestingData)
pipelinePredictionDf.show(10)

// evaluate model with area under ROC
val evaluator = new BinaryClassificationEvaluator()
  .setLabelCol("label")
  .setMetricName("areaUnderROC")

// measure the accuracy
val accuracy = evaluator.evaluate(predictionDf)
println(accuracy)

// measure the accuracy of pipeline model
val pipelineAccuracy = evaluator.evaluate(pipelinePredictionDf)
println(pipelineAccuracy)


trainingData.show(5)



val paramGrid = new ParamGridBuilder().addGrid(randomForestClassifier.maxBins, Array(42, 49,  80)).addGrid(randomForestClassifier.maxDepth, Array(3, 6, 10)).addGrid(randomForestClassifier.impurity, Array( "gini", "entropy")).build()

val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(5)

// this will take some time to run
val cvModel = cv.fit(pipelineTrainingData)
// test cross validated model with test data
val cvPredictionDf = cvModel.transform(pipelineTestingData)
cvPredictionDf.show(10)

val cvAccuracy = evaluator.evaluate(cvPredictionDf)
println(cvAccuracy)

val bestModel = cvModel.bestModel
println(bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap)
