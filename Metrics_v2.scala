//Regression model

import org.apache.spark.ml.classification.LogisticRegression

val lr = new LogisticRegression()
  .setMaxIter(25)
  .setRegParam(0.02)
  .setElasticNetParam(0.8)

//.setFamily("multinomial")

//Adding family multimodal
// Fit the model

val lrModel = lr.fit(weatherFeaturesLabelDF)

val lrModelApplied=lrModel.transform(weatherFeaturesLabelDF)

// Print the coefficients and intercept for logistic regression
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")


// Metricas

// Binari evaluation - prediction

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

val ML_binarymetrics= new BinaryClassificationEvaluator().setRawPredictionCol("prediction")

val ML_auROC= ML_binarymetrics.evaluate(lrModelApplied)

println(f"con ML, métrica binaria,setRawPredictionCol('prediction'): $ML_auROC%1.4f%n")

// Binari evaluation - probability

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

val ML_binarymetrics= new BinaryClassificationEvaluator().setRawPredictionCol("probability")

val ML_auROC= ML_binarymetrics.evaluate(lrModelApplied)

println(f"con ML, métrica binaria,setRawPredictionCol('probability'): $ML_auROC%1.4f%n")

// ROC with MLlib

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

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

MLlib_curvaROC.coalesce(1,true).saveAsTextFile("Try1")



// Estimator

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

// val atrHash =  weatherDFnumeric.columns.diff(Array("RainTomorrow-num")).toSeq



//val tokenizer = new Tokenizer()
 // .setInputCol("text")
//  .setOutputCol("words")
// val hashingTF = new HashingTF()
.setInputCol()
 // .setOutputCol("features")
val lr = new LogisticRegression()
  
      .setElasticNetParam(0.8)
val pipeline = new Pipeline()
  .setStages(Array(lr))


val paramGrid = new ParamGridBuilder().addGrid(lr.maxIter, Array(10, 25, 50)).addGrid(lr.regParam, Array(0.1, 0.02)).build()

val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(new BinaryClassificationEvaluator).setEstimatorParamMaps(paramGrid)


 
val cvModel = cv.fit(weatherFeaturesLabelDF)

// Esta es la parte que no devuelve nada
cvModel.bestModel.extractParamMap()

// Not implemented yet. Here we need to use  the other df
cvModel.transform(weatherFeaturesLabelDF)
  .select("probability", "prediction")
  .collect()
  .foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
    println(s"($id, $text) --> prob=$prob, prediction=$prediction")
  }
