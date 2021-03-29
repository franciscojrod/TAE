// Databricks notebook source
import org.apache.spark.sql.types.{IntegerType, DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.DataFrameNaFunctions
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.feature.OneHotEncoderModel
import org.apache.spark.ml.classification.DecisionTreeClassifier

println("\n\n******************* CARGA DE DATOS *******************\n\n")

// val PATH = "/home/usuario/australia/"
// val FILE_WEATHER = "weatherAUS.csv"

val df1 ="dbfs:/FileStore/shared_uploads/hola@franciscojrodriguez.es/weatherAUS.csv"

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
  schema(weatherSchema).load(df1)

val num_recordsRaw = weatherRawDF.count()
println("Numero de registros RAW: " + num_recordsRaw)

val weatherDF = weatherRawDF.na.replace("MinTemp" :: "MaxTemp" :: "Rainfall"
   :: "Evaporation" :: "Sunshine" :: "WindGustDir" :: "WindGustSpeed" :: "WindDir9am" :: "WindDir3pm"
   :: "WindSpeed9am":: "WindSpeed3pm" :: "Humidity9am" :: "Humidity3pm" :: "Pressure9am" :: "Pressure3pm" ::  "Cloud9am"
   :: "Cloud3pm" :: "Temp9am" :: "Temp3pm" :: "RainToday" :: "RainTomorrow" :: Nil, Map("NA" -> null))


// eliminar columnas de alta correlación y alta tasa de nulos
// Entre Sunshine - Clould9am - Cloud3pm, dejo la columna con menor numero de nulos

val weatherDF_count = weatherDF.count

val weatherDF1 = weatherDF.drop("Temp9am", "Temp3pm", "Pressure3pm", "Evaporation", "Cloud9am", "Cloud3pm", "Sunshine")
// weatherDF1.limit(5).show()


println("\n\n******************* Partición aleatoria *******************")

val dataSplits = weatherDF1.randomSplit(Array(0.7, 0.3), seed=0)
val weatherDF_train = dataSplits(0)
val weatherDF_test = dataSplits(1)

println("Numero de registros train: " + weatherDF_train.count())
println("Numero de registros test: " + weatherDF_test.count())


// Train

// Variables numéricas

val columns = Seq("MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Pressure9am", "Humidity9am", "Humidity3pm")

val weatherDF_train_count = weatherDF_train.count

val weatherDF_train_duplicates = weatherDF_train.withColumn("UniqueID", concat(col("Date"), lit("-"), col("Location"))).dropDuplicates("UniqueID").drop("UniqueID")

val weatherDF_train_duplicate_count = weatherDF_train_count - weatherDF_train_duplicates.count

println(f"Número de valores duplicados $weatherDF_train_duplicate_count")

val weatherDF_train_empty = weatherDF_train_duplicates.na.drop("all")

val weatherDF_train_empty_count = weatherDF_train_count - weatherDF_train_empty.count

println(f"Número de registros completamente vacíos $weatherDF_train_empty_count")

val weatherDF_train_claseNull = weatherDF_train_empty.na.drop("all", Seq("RainTomorrow"))

val weatherDF_train_claseNull_count = weatherDF_train_count - weatherDF_train_claseNull.count

println(f"Número de registros con la clase ausente $weatherDF_train_claseNull")

val weatherDF_train_countAfterDrop = weatherDF_train_claseNull.count

println(f"Número de registros tras los drops $weatherDF_train_countAfterDrop")

val tasa_noClasificados = (weatherDF_count.toDouble - weatherDF_train_countAfterDrop)/ weatherDF_count

println(f"Tasa de no clasificados $tasa_noClasificados")

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

// weatherDF3_train.limit(5).show()


//variables categóricas

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

// weatherDF4_train.limit(5).show()


// Obtenemos el nombrede las columnasde carDF, salvo la clase

val attributeColumns= Seq("Month", "Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "RainTomorrow").toArray

// Generamos los nombres de las nuevas columnas
val outputColumns = attributeColumns.map(_ + "-num").toArray

val siColumns= new StringIndexer().setInputCols(attributeColumns).setOutputCols(outputColumns).setStringOrderType("alphabetDesc")

// Creamos el StringIndexerModel
val simColumns= siColumns.fit(weatherDF4_train)

val weatherDFnumeric= simColumns.transform(weatherDF4_train).drop(attributeColumns:_*)

// weatherDFnumeric.show(5)


// VectorAssembler

val va= new VectorAssembler().setOutputCol("features").setInputCols(weatherDFnumeric.columns.diff(Array("RainTomorrow-num"))) 

val weatherFeaturesClaseDF= va.transform(weatherDFnumeric).select("features", "RainTomorrow-num")


// creamos el StringIndexerpara la clase
val indiceClase= new StringIndexer().setInputCol("RainTomorrow-num").setOutputCol("label").setStringOrderType("alphabetDesc")


// Creamos el DataFramecarFeaturesLabelDFcon columnas features y label
val weatherFeaturesLabelDF= indiceClase.fit(weatherFeaturesClaseDF).transform(weatherFeaturesClaseDF).drop("RainTomorrow-num")


// Instancia de decision tree
/* Importamos de ML*/

/* Creamos una instancia de DecisionTreeClassifier*/
val DTweather=new DecisionTreeClassifier()

// Lo he puesto para que no pite pero hay que revisarlo
 DTweather.setMaxBins(49)
 DTweather.setMaxDepth(15)

/* Entrenamos el modelo: Árbol de Decisión con los parámetros por defecto  */
val DTweatherModel=DTweather.fit(weatherFeaturesLabelDF)

// Para examinar el arbol
DTweatherModel.toDebugString



// COMMAND ----------

/* Predecimos la clase de los ejemplos de prueba*/
val predictionsAndLabelsDF= DTweatherModel.transform(weatherFeaturesLabelDF).select("prediction", "label")
predictionsAndLabelsDF.limit(5).show()
// Evaluación

/* Importamos de ML*/ 
/* Creamos una instancia de clasificación multiclass*/
val metrics= new MulticlassClassificationEvaluator()
/* Fijamos como métrica la tasa de error: accuracy*/
metrics.setMetricName("accuracy")

/* Calculamos la tasa de acierto*/
val acierto = metrics.evaluate(predictionsAndLabelsDF)
/* Calculamos el error  */
val error = 1 - acierto
// Lo mostramos
println(f" $error%1.3f")

/* Guardamos el modelo */
// Con opción overwrite//
//DTcarModel.write.overwrite().save(PATH + "DTcarModelML")

import org.apache.spark.ml.feature.IndexToString
val labelsDF= predictionsAndLabelsDF.select("label").distinct
val converter = new IndexToString().setInputCol("label").setOutputCol("Clase original")
val clasesDF = converter.transform(labelsDF)
println(f"%nMaping índices-etiquetas:")
clasesDF.show


// COMMAND ----------


import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
val ML_multiclasmetrics = new MulticlassClassificationEvaluator().setMetricName("accuracy")
// Calculamos la tasa de error
//
ML_multiclasmetrics.setMetricName("accuracy")
val tasa_acierto = ML_multiclasmetrics.evaluate(predictionsAndLabelsDF)
val tasa_error = 1.0 - tasa_acierto

ML_multiclasmetrics.setMetricName("truePositiveRateByLabel")
val labels= Array(0.0, 1.0)
println(f"%nTasa de ciertos positivos por etiqueta:")
labels.foreach {l =>
  ML_multiclasmetrics.setMetricLabel(l)
  val tp =
  ML_multiclasmetrics.evaluate(predictionsAndLabelsDF)
  println(f"truePositiveRateByLabel($l) = $tp%1.4f")
}

println("Tasas de acierto/error del clasificador con ML, métrica multiclase")
println(f"metricName=accuracy, resto de parámetros por defecto:%n Tasa de acierto = $tasa_acierto%1.4f%n Tasa de error = $tasa_error%1.4f %n")



// COMMAND ----------

// Calculamos truePositiveRate ponderada
//
ML_multiclasmetrics.setMetricName("weightedTruePositiveRate")
val ponderada = ML_multiclasmetrics.evaluate(predictionsAndLabelsDF)
println(f"f%nPrecision ponderada: $ponderada%1.4f")

// Calculamos falsePositiveRateByLabel para cada etiqueta
//

ML_multiclasmetrics.setMetricName("falsePositiveRateByLabel")
println(f"%nTasa de falsos positivos por etiqueta:")
labels.foreach {l =>
  ML_multiclasmetrics.setMetricLabel(l)
  val fp = ML_multiclasmetrics.evaluate(predictionsAndLabelsDF)
  println(f"falsePositiveRateByLabel($l) = $fp%1.4f")
}

// Calculamos falsePositiveRate ponderada
//
ML_multiclasmetrics.setMetricName("weightedFalsePositiveRate")
val ponderada_false =ML_multiclasmetrics.evaluate(predictionsAndLabelsDF)
println(f"%nTasa de falsos positivos ponderada: $ponderada_false%1.4f")

// COMMAND ----------

import org.apache.spark.ml.classification.RandomForestClassifier
weatherDF4_train.cache()
// Examinamos el Árbol
weatherDF4_train.toDebugString
