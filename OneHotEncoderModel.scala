import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel

// Obtenemos el nombrede las columnasde carDF, salvo la clase

val attributeColumns= Seq("Date", "Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday").toArray

// Generamos los nombres de las nuevas columnas
val outputColumns = attributeColumns.map(_ + "-num").toArray

val siColumns= new StringIndexer().setInputCols(attributeColumns).setOutputCols(outputColumns).setStringOrderType("alphabetDesc")

// Creamos el StringIndexerModel
val simColumns= siColumns.fit(weatherDF4_train)

val weatherDFnumeric= simColumns.transform(weatherDF4_train).drop(attributeColumns:_*)


// VectorAssembler

import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.feature.OneHotEncoderModel

val attributeColumns_hot= Seq("Date", "Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday").toArray
val inputCol = outputColumns
// Generamos los nombres de las nuevas columnas
val outputColumns_hot = attributeColumns_hot.map(_ + "-hot").toArray

val hotColumns = new OneHotEncoder().setInputCols(inputCol).setOutputCols(outputColumns_hot)

val hotmColumns= hotColumns.fit(weatherDFnumeric)

val WeatherDFhot = hotmColumns.transform(weatherDFnumeric).drop(inputCol:_*)


/* Da NUll

import org.apache.spark.ml.classification.DecisionTreeClassifier

val DTweather = new DecisionTreeClassifier()

val DTweatherAus = DTweather.fit(WeatherFeaturesClaseDFLabel)

val predictionsAndLabelsDF = DTweatherAus.transform(WeatherFeaturesClaseDFLabel).select("prediction", "label")

predictionsAndLabelsDF.show(3)

*/


import org.apache.spark.ml.feature.VectorAssembler

val va = new VectorAssembler().setOutputCol("features").setInputCols(outputColumns_hot)

val WeatherFeaturesClaseDF = va.transform(WeatherDFhot).select("features", "RainTomorrow")

WeatherFeaturesClaseDF.show(5)


val indiceClase= new StringIndexer().setInputCol("RainTomorrow").setOutputCol("label").setStringOrderType("alphabetDesc")
val WeatherFeaturesClaseDFLabel = indiceClase.fit(WeatherFeaturesClaseDF).transform(WeatherFeaturesClaseDF).drop("RainTomorrow")

WeatherFeaturesClaseDFLabel.show(5)


