import org.apache.spark.sql.types.{IntegerType, DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.DataFrameNaFunctions
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.feature.VectorAssembler

println("\n\n******************* CARGA DE DATOS *******************\n\n")

val PATH = "/home/usuario/australia/"
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
  .na.replace("MinTemp" :: "MaxTemp" :: "Rainfall" 
   :: "Evaporation" :: "Sunshine" :: "WindGustDir" :: "WindGustSpeed" :: "WindDir9am" :: "WindDir3pm" 
   :: "WindSpeed9am":: "WindSpeed3pm" :: "Humidity9am" :: "Humidity3pm" :: "Pressure9am" :: "Pressure3pm" ::  "Cloud9am"
   :: "Cloud3pm" :: "Temp9am" :: "Temp3pm" :: "RainToday" :: "RainTomorrow" :: Nil, Map("NA" -> null))

val num_recordsRaw = weatherRawDF.count()
println("Numero de registros RAW: " + num_recordsRaw)

val weatherDF = weatherRawDF.na.replace("MinTemp" :: "MaxTemp" :: "Rainfall"
   :: "Evaporation" :: "Sunshine" :: "WindGustDir" :: "WindGustSpeed" :: "WindDir9am" :: "WindDir3pm"
   :: "WindSpeed9am":: "WindSpeed3pm" :: "Humidity9am" :: "Humidity3pm" :: "Pressure9am" :: "Pressure3pm" ::  "Cloud9am"
   :: "Cloud3pm" :: "Temp9am" :: "Temp3pm" :: "RainToday" :: "RainTomorrow" :: Nil, Map("NA" -> null))
.na.drop("all")


val num_records = weatherDF.count()
println("Numero de registros: " + num_records)


println("Se han eliminado " + (num_recordsRaw - num_records) + " líneas vacías.")


val primero = weatherDF.first()
println("Primer registro: " + primero)

val weatherDFnoNull = weatherDF.na.drop("all", Seq("RainTomorrow"))

// Chi square
//val categoricalColumns= Seq("Date", "Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday").toArray

//val outputColumns = categoricalColumns.map(_ + "-cat").toArray

//val siColumns= new StringIndexer().setInputCols(categoricalColumns).setOutputCols(outputColumns).setStringOrderType("alphabetDesc")

//val weatherDFnoNullChiSq = siColumns.setHandleInvalid("skip").fit(weatherDFnoNull).transform(weatherDFnoNull).drop(categoricalColumns:_*)

//val va = new VectorAssembler().setOutputCol("features").setInputCols(outputColumns)

//val weatherDFnoNullChiSq2 = va.transform(weatherDFnoNullChiSq).select("features", "RainTomorrow")

//val indiceClase= new StringIndexer().setInputCol("RainTomorrow").setOutputCol("label").setStringOrderType("alphabetDesc")

//val weatherDFnoNullChiSq3 = indiceClase.fit(weatherDFnoNullChiSq2).transform(weatherDFnoNullChiSq2).drop("RainTomorrow")

//ChiSquareTest.test(weatherDFnoNullChiSq3, "features", "label").head

println("\n\n******************* Partición aleatoria *******************")

val dataSplits = weatherDFnoNull.randomSplit(Array(0.7, 0.3), seed=0)
val weatherDF_train = dataSplits(0)
val weatherDF_test = dataSplits(1)

println("Numero de registros train: " + weatherDF_train.count())
println("Numero de registros test: " + weatherDF_test.count())
