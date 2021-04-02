// Load training data

import org.apache.spark.ml.classification.LogisticRegression

val lr = new LogisticRegression()
  .setMaxIter(10)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)

// Fit the model
val lrModel = lr.fit(weatherFeaturesLabelDF).transform(weatherFeaturesLabelDF)
// Print the coefficients and intercept for logistic regression
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")


// Metricas

// Binari evaluation - prediction

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

val ML_binarymetrics= new BinaryClassificationEvaluator().setRawPredictionCol("prediction")

val ML_auROC= ML_binarymetrics.evaluate(lrModel)

println(f"con ML, métrica binaria,setRawPredictionCol('prediction'): $ML_auROC%1.4f%n")

// Binari evaluation - probability

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

val ML_binarymetrics= new BinaryClassificationEvaluator().setRawPredictionCol("probability")

val ML_auROC= ML_binarymetrics.evaluate(lrModel)

println(f"con ML, métrica binaria,setRawPredictionCol('probability'): $ML_auROC%1.4f%n")

ROC with MLlib

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

val probabilitiesAndLabelsRDD= predictionsAndLabels.select("label", "probability").rdd.map{row=> (row.getAs[Vector](1).toArray, row.getDouble(0))}.map{r => ( r._1(1), r._2)}

println(f"%nRDDprobabilitiesAndLabels:")
probabilitiesAndLabelsRDD.take(5).foreach(x => println(x))

val MLlib_binarymetrics= new BinaryClassificationMetrics(probabilitiesAndLabelsRDD,15)

val MLlib_auROC= MLlib_binarymetrics.areaUnderROC

println(f"%nAUCde la curva ROC para la clase SPAM")

println(f"conMLlib, métrica binaria, probabilitiesAndLAbels, 15 bins: $MLlib_auROC%1.4f%n")

val MLlib_curvaROC=MLlib_binarymetrics.roc

println("Puntos para construir curva ROC con MLlib, probabilitiesAndLabels, 15 bins:")

MLlib_curvaROC.take(17).foreach(x => println(x))
