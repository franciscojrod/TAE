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


PATH = ""
:load PATH + "load_transformation.scala"

// MODELO 1: Logistic Regression 

// Selección del modelo

/*

val lr = new LogisticRegression()
  
val pipeline = new Pipeline()
  .setStages(Array(lr))


val paramGrid = new ParamGridBuilder().addGrid(lr.maxIter, Array(10, 25, 50, 100)).addGrid(lr.regParam, Array(0.01, 0.02, 0.1, 0.5)).addGrid(lr.elasticNetParam, Array(0.5, 0.8)).addGrid(lr.fitIntercept, Array(true, false)).addGrid(lr.threshold, Array(0.3, 0.5, 0.7)).build()
                  

val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(new BinaryClassificationEvaluator).setEstimatorParamMaps(paramGrid)




val cvModel = cv.fit(weatherFeaturesLabelDF)

val bestModel = cvModel.bestModel

println(bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap)




/*
 * Resultados best model
 *
 * bestModel: org.apache.spark.ml.Model[_] = pipeline_04a76eb85f07
 * {
 *      logreg_3d812633886d-aggregationDepth: 2,
 *      logreg_3d812633886d-elasticNetParam: 0.5,
 *      logreg_3d812633886d-family: auto,
 *      logreg_3d812633886d-featuresCol: features,
 *      logreg_3d812633886d-fitIntercept: false,
 *      logreg_3d812633886d-labelCol: label,
 *      logreg_3d812633886d-maxIter: 100,
 *      logreg_3d812633886d-predictionCol: prediction,
 *      logreg_3d812633886d-probabilityCol: probability,
 *      logreg_3d812633886d-rawPredictionCol: rawPrediction,
 *      logreg_3d812633886d-regParam: 0.01,
 *      logreg_3d812633886d-standardization: true,
 *      logreg_3d812633886d-threshold: 0.5,
 *      logreg_3d812633886d-tol: 1.0E-6
 * }
*/

*/
// Evaluación del modelo


val lr = new LogisticRegression()
  .setMaxIter(100)
  .setRegParam(0.01)
  .setElasticNetParam(0.5)
  .setThreshold(0.5)
  .setFitIntercept(false)

//Adding family multimodal
// Fit the model

val lrModel = lr.fit(weatherFeaturesLabelDF_train)

val lrModelApplied=lrModel.transform(weatherFeaturesLabelDF_test)

// Print the coefficients and intercept for logistic regression
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")


// Metricas


// Tasa de acierto, su desviación estándar y su intervalo de confianza para una confianza del 95%. 

/* Tasa de acierto */

val predictionsAndLabelsDF_lr = lrModelApplied.select("prediction", "label")
val metrics_lrML = new MulticlassClassificationEvaluator()
metrics_lrML.setMetricName("accuracy")
val acierto_lrML = metrics_lrML.evaluate(predictionsAndLabelsDF_lr)
val error_lrML = 1 - acierto_lrML


/* Desviacian estandar */

predictionsAndLabelsDF_lr.select(stddev(predictionsAndLabelsDF_lr("prediction"))).show()

/* Intervalo de confianza */

val IntConfianzaUp = error_lrML + math.sqrt((error_lrML*(1-error_lrML))/1.96)
val IntConfianzaDown = error_lrML - math.sqrt((error_lrML*(1-error_lrML))/1.96)
println(f"El intervalo de confianza está entre $IntConfianzaDown%1.3f y $IntConfianzaUp%1.3f" )

// Binary evaluation - probability

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

val ML_binarymetrics= new BinaryClassificationEvaluator().setRawPredictionCol("probability")

val ML_auROC= ML_binarymetrics.evaluate(lrModelApplied)

println(f"con ML, métrica binaria,setRawPredictionCol('probability'): $ML_auROC%1.4f%n")

// ROC with MLlib



val probabilitiesAndLabelsRDD= lrModelApplied.select("label", "probability").rdd.map{row=> (row.getAs[Vector](1).toArray, row.getDouble(0))}.map{r => ( r._1(1), r._2)}

println(f"%nRDDprobabilitiesAndLabels:")
probabilitiesAndLabelsRDD.take(5).foreach(x => println(x))



val MLlib_binarymetrics= new BinaryClassificationMetrics(probabilitiesAndLabelsRDD,15)

val MLlib_auROC= MLlib_binarymetrics.areaUnderROC

println(f"%nAUCde la curva ROC para la clase SPAM")

println(f"conMLlib, métrica binaria, probabilitiesAndLAbels, 15 bins: $MLlib_auROC%1.4f%n")

val MLlib_curvaROC=MLlib_binarymetrics.roc

println("Puntos para construir curva ROC con MLlib, probabilitiesAndLabels, 15 bins:")

MLlib_curvaROC.take(17).foreach(x => println(x))

MLlib_curvaROC.coalesce(1,true).saveAsTextFile("lrROC")

// area under PR curve

val auPRC = MLlib_binarymetrics.areaUnderPR
println(f"Área bajo curva PR = $auPRC%1.4f%n")


// Matriz de confusión

val TP = lrModelApplied.filter($"prediction"===1 && $"label"===$"prediction").count()
val FP = lrModelApplied.filter($"prediction"===1 && $"label"=!=$"prediction").count()
val TN = lrModelApplied.filter($"prediction"===0 && $"label"===$"prediction").count()
val FN = lrModelApplied.filter($"prediction"===0 && $"label"=!=$"prediction").count()


println("Confusion Matrix")
println($" $TP  |  $FP")
println($" $FN  |  $TN")

// Tasa de acierto [TP + TN]/[𝑃 + 𝑁]

val TA = (TP + TN)/(TP + FP + TN + FN).toDouble

// Tasa de ciertos positivos (Recall)

val TCP = TP/(TP + FN).toDouble

// Tasa de falsos positivos

val TFP = FP/(FP + TN).toDouble

// Precision

val prec = TP/(TP + FP).toDouble
