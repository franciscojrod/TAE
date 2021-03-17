// ---------

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel

val attributeColumns_hot= Seq("Date", "Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "RainTomorrow").toArray
// Generamos los nombres de las nuevas columnas
val outputColumns_hot = attributeColumns_hot.map(_ + "-hot").toArray

val hotColumns = new StringIndexer().setInputCols(attributeColumns_hot).setOutputCols(outputColumns_hot).setStringOrderType("alphabetDesc")

val hotmColumns= hotColumns.fit(weatherDF4_train)

val WeatherDFhot = hotmColumns.transform(weatherDF4_train).drop(attributeColumns_hot:_*)

WeatherDFhot.show(5)

import org.apache.spark.ml.feature.VectorAssembler

val va = new VectorAssembler().setOutputCol("features").setInputCols(outputColumns_hot)

val WeatherFeaturesClaseDF = va.transform(WeatherDFhot).select("features", "RainTomorrow-hot")

WeatherFeaturesClaseDF.show(5)
