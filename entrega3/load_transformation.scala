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



/*
 * CARGA DE DATOS
 *
 */


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
 * TRANSFORMACIÓN DE DATOS TRAINING
 */
val weatherDF2_test = weatherDF_train_claseNull.withColumn("Month",split(col("Date"),"-").getItem(1).cast("int")).drop("Date")
val weatherDF3_test = columns.foldLeft(weatherDF2_train) { 
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
val weatherDF4_test = columns2.foldLeft(weatherDF3_train) { 
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

/*
 *  CREACION TRAINING DF PARA ML
 */

// Obtenemos el nombrede las columnas, salvo la clase
val attributeColumns_train= Seq("Month", "Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "RainTomorrow").toArray
// Generamos los nombres de las nuevas columnas
val outputColumns_train = attributeColumns_train.map(_ + "-num").toArray
val siColumns_train= new StringIndexer().setInputCols(attributeColumns_train).setOutputCols(outputColumns_train).setStringOrderType("alphabetDesc")
// Creamos el StringIndexerModel
val simColumns_train= siColumns_train.fit(weatherDF4_train)
val weatherDFnumeric_train= simColumns.transform(weatherDF4_train).drop(attributeColumns_train:_*)
// VectorAssembler
val va_train= new VectorAssembler().setOutputCol("features").setInputCols(weatherDFnumeric_train.columns.diff(Array("RainTomorrow-num"))) 
val weatherFeaturesClaseDF_train= va_train.transform(weatherDFnumeric_train).select("features", "RainTomorrow-num")
// creamos el StringIndexerpara la clase
val indiceClase_train= new StringIndexer().setInputCol("RainTomorrow-num").setOutputCol("label").setStringOrderType("alphabetDesc")
// Creamos el DataFramecarFeaturesLabelDFcon columnas features y label
val weatherFeaturesLabelDF_train= indiceClase_train.fit(weatherFeaturesClaseDF_train).transform(weatherFeaturesClaseDF_train).drop("RainTomorrow-num")

/*
 * LIMPIEZA DE DATOS TEST
 */
val columns = Seq("MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Pressure9am", 
                   "Humidity9am", "Humidity3pm")
val weatherDF_test_count = weatherDF_test.count
val weatherDF_test_duplicates = weatherDF_test.withColumn("UniqueID", concat(col("Date"), lit("-"), col("Location"))).dropDuplicates("UniqueID").drop("UniqueID")
val weatherDF_test_duplicate_count = weatherDF_test_count - weatherDF_test_duplicates.count
println(f"Número de valores duplicados eliminados $weatherDF_test_duplicate_count")
val weatherDF_test_empty = weatherDF_test_duplicates.na.drop("all")
val weatherDF_test_empty_count = weatherDF_test_count - weatherDF_test_empty.count
println(f"Número de registros completamente vacíos $weatherDF_test_empty_count")
val weatherDF_test_claseNull = weatherDF_test_empty.na.drop("all", Seq("RainTomorrow"))
val weatherDF_test_claseNull_count = weatherDF_test_count - weatherDF_test_claseNull.count
println(f"Número de registros con la clase ausente $weatherDF_test_claseNull_count")
val weatherDF_test_countAfterDrop = weatherDF_test_claseNull.count
println(f"Número de registros tras los drops $weatherDF_test_countAfterDrop")
val tasa_noClasificados = (weatherDF_test_count.toDouble - weatherDF_test_countAfterDrop)/ weatherDF_test_count
println(f"Tasa de no clasificados $tasa_noClasificados")
/*
 * TRANSFORMACIÓN DE DATOS TEST
 */
val weatherDF2_test = weatherDF_test_claseNull.withColumn("Month",split(col("Date"),"-").getItem(1).cast("int")).drop("Date")
val weatherDF3_test = columns.foldLeft(weatherDF2_test) { 
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
val columns2 = Seq("WindGustDir", "WindDir9am", "WindDir3pm", "RainToday")
val weatherDF4_test = columns2.foldLeft(weatherDF3_test) { 
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


/*
 *  CREACION TRAINING DF PARA ML
 */

// Obtenemos el nombrede las columnas, salvo la clase
val attributeColumns_test= Seq("Month", "Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "RainTomorrow").toArray
// Generamos los nombres de las nuevas columnas
val outputColumns_test = attributeColumns_test.map(_ + "-num").toArray
val siColumns_test= new StringIndexer().setInputCols(attributeColumns_test).setOutputCols(outputColumns_test).setStringOrderType("alphabetDesc")
// Creamos el StringIndexerModel
val simColumns_test= siColumns_test.fit(weatherDF4_test)
val weatherDFnumeric_test= simColumns_test.transform(weatherDF4_test).drop(attributeColumns_test:_*)
// VectorAssembler
val va_test= new VectorAssembler().setOutputCol("features").setInputCols(weatherDFnumeric_test.columns.diff(Array("RainTomorrow-num"))) 
val weatherFeaturesClaseDF_test= va_test.transform(weatherDFnumeric_test).select("features", "RainTomorrow-num")
// creamos el StringIndexerpara la clase
val indiceClase_test= new StringIndexer().setInputCol("RainTomorrow-num").setOutputCol("label").setStringOrderType("alphabetDesc")
// Creamos el DataFramecarFeaturesLabelDFcon columnas features y label
val weatherFeaturesLabelDF_test= indiceClase_test.fit(weatherFeaturesClaseDF_test).transform(weatherFeaturesClaseDF_test).drop("RainTomorrow-num")

