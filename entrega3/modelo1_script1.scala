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

/*

println("\n\n******************* CARGA DE DATOS *******************\n\n")

val PATH = "/home/usuario/australia6/"
val FILE_WEATHER = "weatherAUS.csv"

val weatherSchema = StructType(Array(
  StructField("Date", StringType, true),
  StructField("Location", StringType, true),
  StructField("MinTemp", DoubleType, true),
  StructField("MaxTemp", DoubleType, true),
  StructField("Rainfall", DoubleType, true),
  StructField("Evaporation", DoubleType, true),
  StructField("Sunshine", DoubleType, true),
  StructField("WindGustDir", StringType, true),
  StructField("WindGustSpeed", DoubleType, true),
  StructField("WindDir9am", StringType, true),
  StructField("WindDir3pm", StringType, true),
  StructField("WindSpeed9am", IntegerType, true),
  StructField("WindSpeed3pm", IntegerType, true),
  StructField("Humidity9am", IntegerType, true),
  StructField("Humidity3pm", IntegerType, true),
  StructField("Pressure9am", DoubleType, true),
  StructField("Pressure3pm", DoubleType, true),
  StructField("Cloud9am", IntegerType, true),
  StructField("Cloud3pm", IntegerType, true),
  StructField("Temp9am", DoubleType, true),
  StructField("Temp3pm", DoubleType, true),
  StructField("RainToday", StringType, true),
  StructField("RainTomorrow", StringType, true)));

val weatherRawDF = spark.read.format("csv").
  option("delimiter", ",").
  option("header", true).
  schema(weatherSchema).load(PATH + FILE_WEATHER)

val num_recordsRaw = weatherRawDF.count()
println("Numero de registros RAW: " + num_recordsRaw)

/*
 * LIMPIEZA DE DATOS (PREVIA A PARTICION)
 */

// Eliminación de atributos
// - Por porcentaje de valores ausentes: Evaporation, Cloud9am, Cloud3pm, Sunshine
// - Por correlación: Temp9am, Temp3pm, Pressure3pm
val weatherRawDF2 = weatherRawDF.drop("Temp9am", "Temp3pm", "Pressure3pm", "Evaporation", "Cloud9am", "Cloud3pm", "Sunshine")
                      

val weatherDF = weatherRawDF2.na.replace("MinTemp" :: "MaxTemp" :: "Rainfall" :: "WindGustDir" :: "WindGustSpeed" :: "WindDir9am" 
                                     :: "WindDir3pm" :: "WindSpeed9am":: "WindSpeed3pm" :: "Humidity9am" :: "Humidity3pm" 
                                     :: "Pressure9am" :: "RainToday" :: "RainTomorrow" :: Nil, Map("NA" -> null))


/*
 * CONJUNTOS DE TRAINING Y TEST
 */

val dataSplits = weatherDF.randomSplit(Array(0.7, 0.3), seed=0)
val weatherDF_train = dataSplits(0)
val weatherDF_test = dataSplits(1)

/*
 * LIMPIEZA DE DATOS TRAINING
 */


val columns = Seq("MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Pressure9am", 
                   "Humidity9am", "Humidity3pm")

val weatherDF_train_count = weatherDF_train.count

val weatherDF_train_duplicates = weatherDF_train.withColumn("UniqueID", concat(col("Date"), lit("-"), col("Location"))).dropDuplicates("UniqueID").drop("UniqueID")

val weatherDF_train_duplicate_count = weatherDF_train_count - weatherDF_train_duplicates.count

println(f"Número de valores duplicados eliminados $weatherDF_train_duplicate_count")

val weatherDF_train_empty = weatherDF_train_duplicates.na.drop("all")

val weatherDF_train_empty_count = weatherDF_train_count - weatherDF_train_empty.count

println(f"Número de registros completamente vacíos $weatherDF_train_empty_count")

val weatherDF_train_claseNull = weatherDF_train_empty.na.drop("all", Seq("RainTomorrow"))

val weatherDF_train_claseNull_count = weatherDF_train_count - weatherDF_train_claseNull.count

println(f"Número de registros con la clase ausente $weatherDF_train_claseNull_count")

val weatherDF_train_countAfterDrop = weatherDF_train_claseNull.count

println(f"Número de registros tras los drops $weatherDF_train_countAfterDrop")

val tasa_noClasificados = (weatherDF_train_count.toDouble - weatherDF_train_countAfterDrop)/ weatherDF_train_count

println(f"Tasa de no clasificados $tasa_noClasificados")


/*
 * TRANSFORMACIÓN DE DATOS
 */


val weatherDF2_train = weatherDF_train_claseNull.withColumn("Month",split(col("Date"),"-").getItem(1).cast("int")).drop("Date")

val weatherDF3_train = columns.foldLeft(weatherDF2_train) { 
  (tempDF, colName) => {
   
    val quantiles = weatherDF2_train.stat.approxQuantile(colName,Array(0.25, 0.5, 0.75),0.0)
    val Q1 = quantiles(0)
    val Q3 = quantiles(2)
    val median = quantiles(1)
      
    val IQR = Q3 - Q1

    val lowerRange = Q1 - 1.5*IQR
    val upperRange = Q3+ 1.5*IQR
   
    println(colName + " lower: " + lowerRange + " upper: " + upperRange + " median: " + median)
      
    tempDF.withColumn(
      colName,
      when(col(colName) > upperRange, upperRange)
      .when(col(colName) < lowerRange, lowerRange)
      .when(col(colName).isNull || col(colName) === "NA", median)
      .otherwise(col(colName))
    )
  }  
}

weatherDF3_train.limit(5).show()

val columns2 = Seq("WindGustDir", "WindDir9am", "WindDir3pm", "RainToday")

val weatherDF4_train = columns2.foldLeft(weatherDF3_train) { 
  (tempDF, colName) => {
   
    val moda_array = weatherDF3_train.groupBy(colName).count().orderBy($"count".desc).withColumnRenamed(colName, "value").filter("value != 'null'").filter("value != 'NA'").take(1)
    val moda = moda_array(0)(0)
    
    println(colName + " - moda : " + moda)
    
    tempDF.withColumn(
      colName,
      when(col(colName).isNull || col(colName) === "NA", moda)
      .otherwise(col(colName))
    )
  }  
}

weatherDF4_train.limit(5).show()



// Obtenemos el nombrede las columnasde carDF, salvo la clase

val attributeColumns= Seq("Month", "Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "RainTomorrow").toArray

// Generamos los nombres de las nuevas columnas
val outputColumns = attributeColumns.map(_ + "-num").toArray

val siColumns= new StringIndexer().setInputCols(attributeColumns).setOutputCols(outputColumns).setStringOrderType("alphabetDesc")

// Creamos el StringIndexerModel
val simColumns= siColumns.fit(weatherDF4_train)

val weatherDFnumeric= simColumns.transform(weatherDF4_train).drop(attributeColumns:_*)



// VectorAssembler

val va= new VectorAssembler().setOutputCol("features").setInputCols(weatherDFnumeric.columns.diff(Array("RainTomorrow-num"))) 

val weatherFeaturesClaseDF= va.transform(weatherDFnumeric).select("features", "RainTomorrow-num")


// creamos el StringIndexerpara la clase
val indiceClase= new StringIndexer().setInputCol("RainTomorrow-num").setOutputCol("label").setStringOrderType("alphabetDesc")


// Creamos el DataFramecarFeaturesLabelDFcon columnas features y label
val weatherFeaturesLabelDF= indiceClase.fit(weatherFeaturesClaseDF).transform(weatherFeaturesClaseDF).drop("RainTomorrow-num")

*/

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


//.setFamily("multinomial")

//Adding family multimodal
// Fit the model

val lrModel = lr.fit(weatherFeaturesLabelDF_train)

val lrModelApplied=lrModel.transform(weatherFeaturesLabelDF_test)

// Print the coefficients and intercept for logistic regression
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")


// Metricas

// Binary evaluation - prediction

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

val ML_binarymetrics= new BinaryClassificationEvaluator().setRawPredictionCol("prediction")

val ML_auROC= ML_binarymetrics.evaluate(lrModelApplied)

println(f"con ML, métrica binaria,setRawPredictionCol('prediction'): $ML_auROC%1.4f%n")

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

MLlib_curvaROC.coalesce(1,true).saveAsTextFile("Try1")

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

val prec = TP/(TP + FP)

// PR Curve: Plot of Recall (x) vs Precision (y).



import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint

val lrModelLbFe = lrModelApplied.select("features", "label").rdd.map(row =>
  (row.getAs[Vector](0)(0), row.getAs[Double](1)))


val metrics = new BinaryClassificationMetrics(lrModelLbFe)

metrics.areaUnderPR()

// random forest


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

val Array(pipelineTrainingData, pipelineTestingData) = weatherFeaturesLabelDF.randomSplit(Array(0.7, 0.3), seed)
val cols1 = Array("MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Pressure9am", "Humidity9am", "Humidity3pm")


val assembler = new VectorAssembler()
  .setInputCols(cols1)
  .setOutputCol("features")
val featureDf = assembler.transform(weatherFeaturesLabelDF)
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

// measure the accuracy of pipeline model
val pipelineAccuracy = evaluator.evaluate(pipelinePredictionDf)
println(pipelineAccuracy)

val paramGrid_randomForest = new ParamGridBuilder().addGrid(randomForestClassifier.maxBins, Array(42, 49,  80)).addGrid(randomForestClassifier.maxDepth, Array(3, 6, 10)).addGrid(randomForestClassifier.impurity, Array( "gini", "entropy")).build()

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
