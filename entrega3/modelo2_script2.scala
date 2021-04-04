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
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.feature.OneHotEncoderModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.classification.LogisticRegressionModel



/*
 * CARGA DE DATOS
 *
 */


println("\n\n******************* CARGA DE DATOS *******************\n\n")
val PATH = "/home/usuario/australia/"
val FILE_WEATHER = "weatherAUS.csv"

// Load model
val MODEL_FOLDER = "modelo/modeloRF/"


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
 * LIMPIEZA DE DATOS 
 */

// Eliminación de atributos
// - Por porcentaje de valores ausentes: Evaporation, Cloud9am, Cloud3pm, Sunshine
// - Por correlación: Temp9am, Temp3pm, Pressure3pm
val weatherRawDF2 = weatherRawDF.drop("Temp9am", "Temp3pm", "Pressure3pm", "Evaporation", "Cloud9am", "Cloud3pm", "Sunshine")
                      
val weatherDF = weatherRawDF2.na.replace("MinTemp" :: "MaxTemp" :: "Rainfall" :: "WindGustDir" :: "WindGustSpeed" :: "WindDir9am" 
                                     :: "WindDir3pm" :: "WindSpeed9am":: "WindSpeed3pm" :: "Humidity9am" :: "Humidity3pm" 
                                     :: "Pressure9am" :: "RainToday" :: "RainTomorrow" :: Nil, Map("NA" -> null))

/*
 * LIMPIEZA DE DATOS 
 */
val columns = Seq("MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Pressure9am", 
                   "Humidity9am", "Humidity3pm")
val weatherDF_count = weatherDF.count
val weatherDF_duplicates = weatherDF.withColumn("UniqueID", concat(col("Date"), lit("-"), col("Location"))).dropDuplicates("UniqueID").drop("UniqueID")
val weatherDF_duplicate_count = weatherDF_count - weatherDF_duplicates.count
println(f"Número de valores duplicados eliminados $weatherDF_duplicate_count")
val weatherDF_empty = weatherDF_duplicates.na.drop("all")
val weatherDF_empty_count = weatherDF_count - weatherDF_empty.count
println(f"Número de registros completamente vacíos $weatherDF_empty_count")
val weatherDF_claseNull = weatherDF_empty.na.drop("all", Seq("RainTomorrow"))
val weatherDF_claseNull_count = weatherDF_count - weatherDF_claseNull.count
println(f"Número de registros con la clase ausente $weatherDF_claseNull_count")
val weatherDF_countAfterDrop = weatherDF_claseNull.count
println(f"Número de registros tras los drops $weatherDF_countAfterDrop")
val tasa_noClasificados = (weatherDF_count.toDouble - weatherDF_countAfterDrop)/ weatherDF_count
println(f"Tasa de no clasificados $tasa_noClasificados")


/*
 * TRANSFORMACIÓN DE DATOS
 */
val weatherDF2 = weatherDF_claseNull.withColumn("Month",split(col("Date"),"-").getItem(1).cast("int")).drop("Date")
val weatherDF3 = columns.foldLeft(weatherDF2) { 
  (tempDF, colName) => {
   
    val quantiles = weatherDF2.stat.approxQuantile(colName,Array(0.25, 0.5, 0.75),0.0)
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
weatherDF3.limit(5).show()
val columns2 = Seq("WindGustDir", "WindDir9am", "WindDir3pm", "RainToday")
val weatherDF4 = columns2.foldLeft(weatherDF3) { 
  (tempDF, colName) => {
   
    val moda_array = weatherDF3.groupBy(colName).count().orderBy($"count".desc).withColumnRenamed(colName, "value").filter("value != 'null'").filter("value != 'NA'").take(1)
    val moda = moda_array(0)(0)
    
    println(colName + " - moda : " + moda)
    
    tempDF.withColumn(
      colName,
      when(col(colName).isNull || col(colName) === "NA", moda)
      .otherwise(col(colName))
    )
  }  
}
weatherDF4.limit(5).show()

/*
 *  CREACION DF PARA ML
 */

// Obtenemos el nombrede las columnas, salvo la clase
val attributeColumns= Seq("Month", "Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "RainTomorrow").toArray
// Generamos los nombres de las nuevas columnas
val outputColumns = attributeColumns.map(_ + "-num").toArray
val siColumns= new StringIndexer().setInputCols(attributeColumns).setOutputCols(outputColumns).setStringOrderType("alphabetDesc")
// Creamos el StringIndexerModel
val simColumns= siColumns.fit(weatherDF4)
val weatherDFnumeric= simColumns.transform(weatherDF4).drop(attributeColumns:_*)
// VectorAssembler
val va= new VectorAssembler().setOutputCol("features").setInputCols(weatherDFnumeric.columns.diff(Array("RainTomorrow-num"))) 
val weatherFeaturesClaseDF= va.transform(weatherDFnumeric).select("features", "RainTomorrow-num")
// creamos el StringIndexerpara la clase
val indiceClase= new StringIndexer().setInputCol("RainTomorrow-num").setOutputCol("label").setStringOrderType("alphabetDesc")
// Creamos el DataFramecarFeaturesLabelDFcon columnas features y label
val weatherFeaturesLabelDF= indiceClase.fit(weatherFeaturesClaseDF).transform(weatherFeaturesClaseDF).drop("RainTomorrow-num")


println("\n\n******************* CARGA DE DATOS *******************\n\n")



val rfmodel = RandomForestClassificationModel.load(PATH + MODEL_FOLDER)
val rfModelApplied=rfmodel.transform(weatherFeaturesLabelDF)

// Metricas


// Tasa de acierto, su desviación estándar y su intervalo de confianza para una confianza del 95%. 

/* Tasa de acierto */

val predictionsAndLabelsDF_rf = rfModelApplied.select("prediction", "label")
val metrics_rfML = new MulticlassClassificationEvaluator()
metrics_rfML.setMetricName("accuracy")
val acierto_rfML = metrics_rfML.evaluate(predictionsAndLabelsDF_rf)
val error_rfML = 1 - acierto_rfML


/* Desviacian estandar */

predictionsAndLabelsDF_rf.select(stddev(predictionsAndLabelsDF_rf("prediction"))).show()

/* Intervalo de confianza */

val IntConfianzaUp = error_rfML + math.sqrt((error_rfML*(1-error_rfML))/1.96)
val IntConfianzaDown = error_rfML - math.sqrt((error_rfML*(1-error_rfML))/1.96)
println(f"El intervalo de confianza está entre $IntConfianzaDown%1.3f y $IntConfianzaUp%1.3f" )

// Binary evaluation - probability


val ML_binarymetrics= new BinaryClassificationEvaluator().setRawPredictionCol("probability")

val ML_auROC= ML_binarymetrics.evaluate(rfModelApplied)

println(f"con ML, métrica binaria,setRawPredictionCol('probability'): $ML_auROC%1.4f%n")

// ROC with MLlib



val probabilitiesAndLabelsRDD= rfModelApplied.select("label", "probability").rdd.map{row=> (row.getAs[Vector](1).toArray, row.getDouble(0))}.map{r => ( r._1(1), r._2)}

println(f"%nRDDprobabilitiesAndLabels:")
probabilitiesAndLabelsRDD.take(5).foreach(x => println(x))



val MLlib_binarymetrics= new BinaryClassificationMetrics(probabilitiesAndLabelsRDD,15)

val MLlib_auROC= MLlib_binarymetrics.areaUnderROC

println(f"%nAUCde la curva ROC para la clase SPAM")

println(f"conMLlib, métrica binaria, probabilitiesAndLAbels, 15 bins: $MLlib_auROC%1.4f%n")

val MLlib_curvaROC=MLlib_binarymetrics.roc

println("Puntos para construir curva ROC con MLlib, probabilitiesAndLabels, 15 bins:")

MLlib_curvaROC.take(17).foreach(x => println(x))

MLlib_curvaROC.coalesce(1,true).saveAsTextFile("rfROC")

// area under PR curve

val auPRC = MLlib_binarymetrics.areaUnderPR
println(f"Área bajo curva PR = $auPRC%1.4f%n")


// Matriz de confusión

val TP = rfModelApplied.filter($"prediction"===1 && $"label"===$"prediction").count()
val FP = rfModelApplied.filter($"prediction"===1 && $"label"=!=$"prediction").count()
val TN = rfModelApplied.filter($"prediction"===0 && $"label"===$"prediction").count()
val FN = rfModelApplied.filter($"prediction"===0 && $"label"=!=$"prediction").count()


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
