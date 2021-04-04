import org.apache.spark.sql.types.{IntegerType, DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.DataFrameNaFunctions
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{PipelineModel, Pipeline}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.evaluation.MultilabelMetrics
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidator


/*
 * CARGA DE DATOS
 *
 */

PATH = "/home/usuario/australia/"
:load PATH + "load_transformation.scala"


// MODELO 2: Random Forest 

// Selección del modelo



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

val Array(pipelineTrainingData, pipelineTestingData) = weatherDF4_train.randomSplit(Array(0.7, 0.3), seed)
val cols1 = Array("MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Pressure9am", "Humidity9am", "Humidity3pm")


val assembler = new VectorAssembler()
  .setInputCols(cols1)
  .setOutputCol("features")

val indexer = new StringIndexer()
  .setInputCol("RainTomorrow")
  .setOutputCol("label")

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

// measure the accuracy of pipeline model
val pipelineAccuracy = evaluator.evaluate(pipelinePredictionDf)
println(pipelineAccuracy)

val paramGrid_randomForest = new ParamGridBuilder().addGrid(randomForestClassifier.maxBins, Array(42, 49,  80)).addGrid(randomForestClassifier.maxDepth, Array(3, 6, 10)).addGrid(randomForestClassifier.impurity, Array( "gini", "entropy")).build()

val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid_randomForest).setNumFolds(5)

// this will take some time to run
val cvModel = cv.fit(pipelineTrainingData)
// test cross validated model with test data
val cvPredictionDf = cvModel.transform(pipelineTestingData)
cvPredictionDf.show(10)

val cvAccuracy = evaluator.evaluate(cvPredictionDf)
println(cvAccuracy)

val bestModel = cvModel.bestModel
println(bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap)
