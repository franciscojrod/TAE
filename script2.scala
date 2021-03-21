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
import org.apache.spark.ml.classification.DecisionTreeClassificationModel

println("\n\n******************* CARGA DE DATOS *******************\n\n")

val PATH = "/home/usuario/australia/"
val FILE_WEATHER = "weatherAUS.csv"
val MODEL_FOLDER = "modelo/"

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

val weatherDF = weatherRawDF.na.replace("MinTemp" :: "MaxTemp" :: "Rainfall"
   :: "Evaporation" :: "Sunshine" :: "WindGustDir" :: "WindGustSpeed" :: "WindDir9am" :: "WindDir3pm"
   :: "WindSpeed9am":: "WindSpeed3pm" :: "Humidity9am" :: "Humidity3pm" :: "Pressure9am" :: "Pressure3pm" ::  "Cloud9am"
   :: "Cloud3pm" :: "Temp9am" :: "Temp3pm" :: "RainToday" :: "RainTomorrow" :: Nil, Map("NA" -> null))

// Variables numéricas

val columns = Seq("MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
                   "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Pressure9am", 
                   "Pressure3pm", "Humidity9am", "Humidity3pm", "Temp9am", "Temp3pm")

val weatherDF_count = weatherDF.count

val weatherDF_duplicates = weatherDF.withColumn("UniqueID", concat(col("Date"), lit("-"), col("Location"))).dropDuplicates("UniqueID").drop("UniqueID")

val weatherDF_duplicate_count = weatherDF_count - weatherDF_duplicates.count

println(f"Número de valores duplicados $weatherDF_duplicate_count")

val weatherDF_empty = weatherDF_duplicates.na.drop("all")

val weatherDF_empty_count = weatherDF_count - weatherDF_empty.count

println(f"Número de registros completamente vacíos $weatherDF_empty_count")

val weatherDF_claseNull = weatherDF_empty.na.drop("all", Seq("RainTomorrow"))

val weatherDF_claseNull_count = weatherDF_count - weatherDF_claseNull.count

println(f"Número de registros con la clase ausente $weatherDF_claseNull")

val weatherDF_countAfterDrop = weatherDF_claseNull.count

println(f"Número de registros tras los drops $weatherDF_countAfterDrop")

val tasa_noClasificados = (weatherDF_count.toDouble - weatherDF_countAfterDrop)/ weatherDF_count

println(f"Tasa de no clasificados $tasa_noClasificados")

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



//variables categóricas

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



// variables categóricas 2 (cloud)

val columns3 = Seq("Cloud9am", "Cloud3pm")

val weatherDF5 = columns3.foldLeft(weatherDF4) { 
  (tempDF, colName) => {
   
    val moda_array = weatherDF4.groupBy(colName).count().orderBy($"count".desc).withColumnRenamed(colName, "value").filter("value is not null").take(1)
    
    val moda = moda_array(0)(0)
    
    println(colName + " - moda : " + moda)
    
    tempDF.withColumn(
      colName,
      when(col(colName).isNull || col(colName) === "NA" || col(colName) > 8 || col(colName) < 0, moda)
      .otherwise(col(colName))
    )
  }  
}

weatherDF5.limit(5).show()

// Obtenemos el nombrede las columnasde weatherDF, salvo la clase

val attributeColumns= Seq("Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday").toArray

// Generamos los nombres de las nuevas columnas
val outputColumns = attributeColumns.map(_ + "-num").toArray

val siColumns= new StringIndexer().setInputCols(attributeColumns).setOutputCols(outputColumns).setStringOrderType("alphabetDesc")

// Creamos el StringIndexerModel

val simColumns= siColumns.setHandleInvalid("skip").fit(weatherDF5)

val weatherDFnumeric= simColumns.transform(weatherDF5).drop(attributeColumns:_*)

// VectorAssembler

val attributeColumns_hot= Seq("Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday").toArray

val inputCol = outputColumns

// Generamos los nombres de las nuevas columnas

val outputColumns_hot = attributeColumns_hot.map(_ + "-hot").toArray

val hotColumns = new OneHotEncoder().setInputCols(inputCol).setOutputCols(outputColumns_hot)

val hotmColumns= hotColumns.fit(weatherDFnumeric)

val WeatherDFhot = hotmColumns.transform(weatherDFnumeric).drop(inputCol:_*)

val va = new VectorAssembler().setOutputCol("features").setInputCols(outputColumns_hot)

val WeatherFeaturesClaseDF = va.transform(WeatherDFhot).select("features", "RainTomorrow")

WeatherFeaturesClaseDF.show(2)

val indiceClase= new StringIndexer().setInputCol("RainTomorrow").setOutputCol("label").setStringOrderType("alphabetDesc")
val WeatherFeaturesClaseDFLabel = indiceClase.fit(WeatherFeaturesClaseDF).transform(WeatherFeaturesClaseDF).drop("RainTomorrow")

WeatherFeaturesClaseDFLabel.show(5)

// Load model

val loadModel = DecisionTreeClassificationModel.load(PATH + MODEL_FOLDER)
val predictionsAndLabelsDF_loadModel = loadModel.transform(WeatherFeaturesClaseDFLabel).select("prediction", "label")

predictionsAndLabelsDF_loadModel.show(3)

val errores_loadModel = predictionsAndLabelsDF_loadModel.map(x=>if(x(0)==x(1))0 else 1).collect.sum
val error_loadModel = errores_loadModel.toDouble/predictionsAndLabelsDF_loadModel.count
val acierto_loadModel = 1-error_loadModel

println(f"Tasa de error loadModel= $error_loadModel%1.3f")

