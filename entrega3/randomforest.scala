import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.{PipelineModel, Pipeline}

val seed = 0
val randomForestClassifier = new RandomForestClassifier()
  .setImpurity("gini")
  .setMaxDepth(10)
  .setNumTrees(20)
  .setMaxBins(49)
  .setFeatureSubsetStrategy("auto")
  .setSeed(seed)
val randomForestModel = randomForestClassifier.fit(weatherFeaturesLabelDF)
//println(randomForestModel.toDebugString)

val predictionDf = randomForestModel.transform(weatherFeaturesLabelDF)
predictionDf.show(10)

// False positive and false negative True positive true negative

val TP = predictionDf.filter($"prediction"===1 && $"label"===$"prediction").count()
val FP = predictionDf.filter($"prediction"===1 && $"label"=!=$"prediction").count()
val TN = predictionDf.filter($"prediction"===0 && $"label"===$"prediction").count()
val FN = predictionDf.filter($"prediction"===0 && $"label"=!=$"prediction").count()


// Tasa de acierto [TP + TN]/[𝑃 + 𝑁]


val TA = (TP + TN)/(TP + FP + TN + FN).toDouble

// Tasa de ciertos positivos (Recall)

val TCP = TP/(TP + FN).toDouble

// Tasa de falsos positivos

val TFP = FP/(FP + TN).toDouble

// Precision

val prec = TP/(TP + FP).toDouble

// PR Curve: Plot of Recall (x) vs Precision (y).


// https://github.com/apache/spark/blob/master/examples/src/main/scala/org/apache/spark/examples/mllib/BinaryClassificationMetricsExample.scala


import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint

val lrModelLbFe = predictionDf.select("features", "label").rdd.map(row =>
  (row.getAs[Vector](0)(0), row.getAs[Double](1)))


val metrics = new BinaryClassificationMetrics(lrModelLbFe)

metrics.areaUnderPR()


/*-------------*/

val predictionsAndLabelsDFerror = predictionDf.select("prediction","label")

/* Creamos una instancia de clasificacion multiclass */
val LRmetrics_D = new MulticlassClassificationEvaluator()
/* Fijamos como mÃ©trica la tasa de error: accuracy */ 
LRmetrics_D.setMetricName("accuracy")
/* Calculamos la tasa de acierto */
val aciertoLR = LRmetrics_D.evaluate(predictionsAndLabelsDFerror)
/* Calculamos el error */
val errorLR = 1 - aciertoLR
// Lo mostramos
println(f"Tasa de error= $errorLR%1.3f")

/* DesviaciÃ³n estÃ¡ndar */

predictionDf.select(stddev(predictionDf("prediction"))).show()

/* Intervalo de confianza */

val IntConfianzaUp = errorLR + math.sqrt((errorLR*(1-errorLR))/1.96)
val IntConfianzaDown = errorLR - math.sqrt((errorLR*(1-errorLR))/1.96)
println(f"El intervalo de confianza está entre $IntConfianzaDown%1.3f y $IntConfianzaUp%1.3f" )
