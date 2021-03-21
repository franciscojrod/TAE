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

val num_recordsRaw = weatherRawDF.count()
println("Numero de registros RAW: " + num_recordsRaw)

val weatherDF = weatherRawDF.na.replace("MinTemp" :: "MaxTemp" :: "Rainfall"
   :: "Evaporation" :: "Sunshine" :: "WindGustDir" :: "WindGustSpeed" :: "WindDir9am" :: "WindDir3pm"
   :: "WindSpeed9am":: "WindSpeed3pm" :: "Humidity9am" :: "Humidity3pm" :: "Pressure9am" :: "Pressure3pm" ::  "Cloud9am"
   :: "Cloud3pm" :: "Temp9am" :: "Temp3pm" :: "RainToday" :: "RainTomorrow" :: Nil, Map("NA" -> null))

println("\n\n******************* Partición aleatoria *******************")

val dataSplits = weatherDF.randomSplit(Array(0.7, 0.3), seed=0)
val weatherDF_train = dataSplits(0)
val weatherDF_test = dataSplits(1)

println("Numero de registros train: " + weatherDF_train.count())
println("Numero de registros test: " + weatherDF_test.count())


// Train

// Variables numéricas

val columns = Seq("MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
                   "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Pressure9am", 
                   "Pressure3pm", "Humidity9am", "Humidity3pm", "Temp9am", "Temp3pm")

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

val porcentaje_eliminados = ((weatherDF_train_count.toDouble - weatherDF_train_countAfterDrop) * 100)/ weatherDF_train_count

println(f"Porcentaje de registros eliminados $porcentaje_eliminados %%")

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

weatherDF4_train.limit(5).show()


// variables categóricas 2 (cloud)

val columns3 = Seq("Cloud9am", "Cloud3pm")

val weatherDF5_train = columns3.foldLeft(weatherDF4_train) { 
  (tempDF, colName) => {
   
    val moda_array = weatherDF4_train.groupBy(colName).count().orderBy($"count".desc).withColumnRenamed(colName, "value").filter("value is not null").take(1)
    
    val moda = moda_array(0)(0)
    
    println(colName + " - moda : " + moda)
    
    tempDF.withColumn(
      colName,
      when(col(colName).isNull || col(colName) === "NA" || col(colName) > 8 || col(colName) < 0, moda)
      .otherwise(col(colName))
    )
  }  
}

weatherDF5_train.limit(5).show()

// Obtenemos el nombrede las columnasde weatherDF, salvo la clase

val attributeColumns= Seq("Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday").toArray

// Generamos los nombres de las nuevas columnas
val outputColumns = attributeColumns.map(_ + "-num").toArray

val siColumns= new StringIndexer().setInputCols(attributeColumns).setOutputCols(outputColumns).setStringOrderType("alphabetDesc")

// Creamos el StringIndexerModel

val simColumns= siColumns.setHandleInvalid("skip").fit(weatherDF5_train)

val weatherDFnumeric= simColumns.transform(weatherDF5_train).drop(attributeColumns:_*)

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

// Fin train

// Test

val columns_test = Seq("MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
                   "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Pressure9am", 
                   "Pressure3pm", "Humidity9am", "Humidity3pm", "Temp9am", "Temp3pm")

val weatherDF_test_count = weatherDF_test.count

val weatherDF_test_duplicates = weatherDF_test.withColumn("UniqueID", concat(col("Date"), lit("-"), col("Location"))).dropDuplicates("UniqueID").drop("UniqueID")

val weatherDF_test_duplicate_count = weatherDF_test_count - weatherDF_test_duplicates.count

println(f"Número de valores duplicados $weatherDF_test_duplicate_count")

val weatherDF_test_empty = weatherDF_test_duplicates.na.drop("all")

val weatherDF_test_empty_count = weatherDF_test_count - weatherDF_test_empty.count

println(f"Número de registros completamente vacíos $weatherDF_test_empty_count")

val weatherDF_test_claseNull = weatherDF_test_empty.na.drop("all", Seq("RainTomorrow"))

val weatherDF_test_claseNull_count = weatherDF_test_count - weatherDF_test_claseNull.count

println(f"Número de registros con la clase ausente $weatherDF_test_claseNull")

val weatherDF_test_countAfterDrop = weatherDF_test_claseNull.count

println(f"Número de registros tras los drops $weatherDF_test_countAfterDrop")

val porcentaje_eliminados_test = ((weatherDF_test_count.toDouble - weatherDF_test_countAfterDrop) * 100)/ weatherDF_test_count

println(f"Porcentaje de registros eliminados $porcentaje_eliminados_test %%")

val weatherDF2_test = weatherDF_test_claseNull.withColumn("Month",split(col("Date"),"-").getItem(1).cast("int")).drop("Date")

val weatherDF3_test = columns_test.foldLeft(weatherDF2_test) { 
  (tempDF, colName) => {
   
    val quantiles = weatherDF2_test.stat.approxQuantile(colName,Array(0.25, 0.5, 0.75),0.0)
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

weatherDF3_test.limit(5).show()


//variables categóricas

val columns2_test = Seq("WindGustDir", "WindDir9am", "WindDir3pm", "RainToday")

val weatherDF4_test = columns2_test.foldLeft(weatherDF3_test) { 
  (tempDF, colName) => {
   
    val moda_array = weatherDF3_test.groupBy(colName).count().orderBy($"count".desc).withColumnRenamed(colName, "value").filter("value != 'null'").filter("value != 'NA'").take(1)
    val moda = moda_array(0)(0)
    
    println(colName + " - moda : " + moda)
    
    tempDF.withColumn(
      colName,
      when(col(colName).isNull || col(colName) === "NA", moda)
      .otherwise(col(colName))
    )
  }  
}

weatherDF4_test.limit(5).show()


// variables categóricas 2 (cloud)

val columns3_test = Seq("Cloud9am", "Cloud3pm")

val weatherDF5_test = columns3_test.foldLeft(weatherDF4_test) { 
  (tempDF, colName) => {
   
    val moda_array = weatherDF4_test.groupBy(colName).count().orderBy($"count".desc).withColumnRenamed(colName, "value").filter("value is not null").take(1)
    
    val moda = moda_array(0)(0)
    
    println(colName + " - moda : " + moda)
    
    tempDF.withColumn(
      colName,
      when(col(colName).isNull || col(colName) === "NA" || col(colName) > 8 || col(colName) < 0, moda)
      .otherwise(col(colName))
    )
  }  
}

weatherDF5_test.limit(5).show()

// Obtenemos el nombrede las columnasde weatherDF, salvo la clase

val attributeColumns_test= Seq("Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday").toArray

// Generamos los nombres de las nuevas columnas
val outputColumns_test = attributeColumns_test.map(_ + "-num").toArray

val siColumns_test= new StringIndexer().setInputCols(attributeColumns_test).setOutputCols(outputColumns_test).setStringOrderType("alphabetDesc")

// Creamos el StringIndexerModel

val simColumns_test= siColumns_test.setHandleInvalid("skip").fit(weatherDF5_test)

val weatherDFnumeric_test= simColumns_test.transform(weatherDF5_test).drop(attributeColumns_test:_*)

// VectorAssembler

val attributeColumns_hot_test= Seq("Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday").toArray

val inputCol_test = outputColumns_test

// Generamos los nombres de las nuevas columnas

val outputColumns_hot_test = attributeColumns_hot_test.map(_ + "-hot").toArray

val hotColumns_test = new OneHotEncoder().setInputCols(inputCol_test).setOutputCols(outputColumns_hot_test)

val hotmColumns_test= hotColumns_test.fit(weatherDFnumeric_test)

val WeatherDFhot_test = hotmColumns_test.transform(weatherDFnumeric_test).drop(inputCol_test:_*)

val va_test = new VectorAssembler().setOutputCol("features").setInputCols(outputColumns_hot_test)

val WeatherFeaturesClaseDF_test = va_test.transform(WeatherDFhot_test).select("features", "RainTomorrow")

WeatherFeaturesClaseDF_test.show(2)


val indiceClase_test= new StringIndexer().setInputCol("RainTomorrow").setOutputCol("label").setStringOrderType("alphabetDesc")
val WeatherFeaturesClaseDFLabel_test = indiceClase_test.fit(WeatherFeaturesClaseDF_test).transform(WeatherFeaturesClaseDF_test).drop("RainTomorrow")

WeatherFeaturesClaseDFLabel_test.show(5)

// Fin Test

// Modelo ML

//valores por defecto

val DTweather = new DecisionTreeClassifier()

val DTweatherAus = DTweather.fit(WeatherFeaturesClaseDFLabel)

val predictionsAndLabelsDF = DTweatherAus.transform(WeatherFeaturesClaseDFLabel_test).select("prediction", "label")

predictionsAndLabelsDF.show(3)

val metrics = new MulticlassClassificationEvaluator()
metrics.setMetricName("accuracy")
val acierto = metrics.evaluate(predictionsAndLabelsDF)
val error = 1 - acierto
println(f"Tasa de error= $error%1.3f")

//valores MaxBin, MaxProf y entropia

val impureza = "entropy"
val maxProf = 9
val maxBins = 150

val DTweather_ML1 = new DecisionTreeClassifier().setImpurity(impureza).setMaxDepth(maxProf).setMaxBins(maxBins)

val DTweatherAus_ML1 = DTweather_ML1.fit(WeatherFeaturesClaseDFLabel)

val predictionsAndLabelsDF_ML1 = DTweatherAus_ML1.transform(WeatherFeaturesClaseDFLabel_test).select("prediction", "label")

predictionsAndLabelsDF_ML1.show(3)

val errores_ML1 = predictionsAndLabelsDF_ML1.map(x=>if(x(0)==x(1))0 else 1).collect.sum
val error_ML1 = errores_ML1.toDouble/predictionsAndLabelsDF_ML1.count
val acierto_ML1 = 1-error_ML1
println(f"Tasa de error ML1= $error_ML1%1.3f")

val maxProf2 = 9
val maxBins2 = 2

val DTweather_ML2 = new DecisionTreeClassifier().setImpurity(impureza).setMaxDepth(maxProf2).setMaxBins(maxBins2)

val DTweatherAus_ML2 = DTweather_ML2.fit(WeatherFeaturesClaseDFLabel)

val predictionsAndLabelsDF_ML2 = DTweatherAus_ML2.transform(WeatherFeaturesClaseDFLabel_test).select("prediction", "label")

predictionsAndLabelsDF_ML2.show(3)


val errores_ML2 = predictionsAndLabelsDF_ML2.map(x=>if(x(0)==x(1))0 else 1).collect.sum
val error_ML2 = errores_ML2.toDouble/predictionsAndLabelsDF_ML2.count
val acierto_ML2 = 1-error_ML2
println(f"Tasa de error ML2= $error_ML2%1.3f")

// se comprueba que maxBins no modifica nada

val maxProf_ML3 = 11
val maxBins_ML3 = 2

val DTweather_ML3 = new DecisionTreeClassifier().setImpurity(impureza).setMaxDepth(maxProf_ML3).setMaxBins(maxBins_ML3)

val DTweatherAus_ML3 = DTweather_ML3.fit(WeatherFeaturesClaseDFLabel)

val predictionsAndLabelsDF_ML3 = DTweatherAus_ML3.transform(WeatherFeaturesClaseDFLabel_test).select("prediction", "label")

predictionsAndLabelsDF_ML3.show(3)

val errores_ML3 = predictionsAndLabelsDF_ML3.map(x=>if(x(0)==x(1))0 else 1).collect.sum
val error_ML3 = errores_ML3.toDouble/predictionsAndLabelsDF_ML3.count
val acierto_ML3 = 1-error_ML3
println(f"Tasa de error ML3= $error_ML3%1.3f")

val maxProf_ML4 = 15
val maxBins_ML4 = 2

val DTweather_ML4 = new DecisionTreeClassifier().setImpurity(impureza).setMaxDepth(maxProf_ML4).setMaxBins(maxBins_ML4)

val DTweatherAus_ML4 = DTweather_ML4.fit(WeatherFeaturesClaseDFLabel)

val predictionsAndLabelsDF_ML4 = DTweatherAus_ML4.transform(WeatherFeaturesClaseDFLabel_test).select("prediction", "label")

predictionsAndLabelsDF_ML4.show(3)

val errores_ML4 = predictionsAndLabelsDF_ML4.map(x=>if(x(0)==x(1))0 else 1).collect.sum
val error_ML4 = errores_ML4.toDouble/predictionsAndLabelsDF_ML4.count
val acierto_ML4 = 1-error_ML4 

println(f"Tasa de error ML4 = $error_ML4%1.3f")

// Guardando modelo

DTweatherAus_ML4.write.overwrite().save(PATH  + "modelo/DTweatherAus_ML4")

