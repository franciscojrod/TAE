import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.classification.DecisionTreeClassificationModel

// Obtenemos el nombrede las columnasde weatherDF, salvo la clase

val attributeColumns= Seq("Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday").toArray

// Generamos los nombres de las nuevas columnas
val outputColumns = attributeColumns.map(_ + "-num").toArray

val siColumns= new StringIndexer().setInputCols(attributeColumns).setOutputCols(outputColumns).setStringOrderType("alphabetDesc")

// Creamos el StringIndexerModel

val simColumns= siColumns.setHandleInvalid("skip").fit(weatherDF6_train)

val weatherDFnumeric= simColumns.transform(weatherDF6_train).drop(attributeColumns:_*)

// VectorAssembler

import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.feature.OneHotEncoderModel

val attributeColumns_hot= Seq("Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday").toArray
val inputCol = outputColumns
// Generamos los nombres de las nuevas columnas
val outputColumns_hot = attributeColumns_hot.map(_ + "-hot").toArray

val hotColumns = new OneHotEncoder().setInputCols(inputCol).setOutputCols(outputColumns_hot)

val hotmColumns= hotColumns.fit(weatherDFnumeric)

val WeatherDFhot = hotmColumns.transform(weatherDFnumeric).drop(inputCol:_*)


import org.apache.spark.ml.feature.VectorAssembler

val va = new VectorAssembler().setOutputCol("features").setInputCols(outputColumns_hot)

val WeatherFeaturesClaseDF = va.transform(WeatherDFhot).select("features", "RainTomorrow")

WeatherFeaturesClaseDF.show(2)


val indiceClase= new StringIndexer().setInputCol("RainTomorrow").setOutputCol("label").setStringOrderType("alphabetDesc")
val WeatherFeaturesClaseDFLabel = indiceClase.fit(WeatherFeaturesClaseDF).transform(WeatherFeaturesClaseDF).drop("RainTomorrow")

WeatherFeaturesClaseDFLabel.show(5)

//valores por defecto

import org.apache.spark.ml.classification.DecisionTreeClassifier

val DTweather = new DecisionTreeClassifier()

val DTweatherAus = DTweather.fit(WeatherFeaturesClaseDFLabel)

val predictionsAndLabelsDF = DTweatherAus.transform(WeatherFeaturesClaseDFLabel).select("prediction", "label")

predictionsAndLabelsDF.show(3)

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

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

val predictionsAndLabelsDF_ML1 = DTweatherAus_ML1.transform(WeatherFeaturesClaseDFLabel).select("prediction", "label")

predictionsAndLabelsDF_ML1.show(3)

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

val metrics_ML1 = new MulticlassClassificationEvaluator()
metrics_ML1.setMetricName("accuracy")
val acierto_ML1 = metrics.evaluate(predictionsAndLabelsDF_ML1)
val error_ML1 = 1 - acierto_ML1
println(f"Tasa de error ML1= $error_ML1%1.3f")

val maxProf2 = 9
val maxBins2 = 2

val DTweather_ML2 = new DecisionTreeClassifier().setImpurity(impureza).setMaxDepth(maxProf2).setMaxBins(maxBins2)

val DTweatherAus_ML2 = DTweather_ML2.fit(WeatherFeaturesClaseDFLabel)

val predictionsAndLabelsDF_ML2 = DTweatherAus_ML2.transform(WeatherFeaturesClaseDFLabel).select("prediction", "label")

predictionsAndLabelsDF_ML2.show(3)


val metrics_ML2 = new MulticlassClassificationEvaluator()
metrics_ML2.setMetricName("accuracy")
val acierto_ML2 = metrics.evaluate(predictionsAndLabelsDF_ML2)
val error_ML2 = 1 - acierto_ML2
println(f"Tasa de error ML2= $error_ML2%1.3f")

// se comprueba que maxBins no modifica nada

val maxProf_ML3 = 3
val maxBins_ML3 = 2

val DTweather_ML3 = new DecisionTreeClassifier().setImpurity(impureza).setMaxDepth(maxProf_ML3).setMaxBins(maxBins_ML3)

val DTweatherAus_ML3 = DTweather_ML3.fit(WeatherFeaturesClaseDFLabel)

val predictionsAndLabelsDF_ML3 = DTweatherAus_ML3.transform(WeatherFeaturesClaseDFLabel).select("prediction", "label")

predictionsAndLabelsDF_ML3.show(3)

val metrics_ML3 = new MulticlassClassificationEvaluator()
metrics.setMetricName("accuracy")
val acierto_ML3 = metrics_ML3.evaluate(predictionsAndLabelsDF_ML3)
val error_ML3 = 1 - acierto_ML3
println(f"Tasa de error ML3= $error_ML3%1.3f")

val maxProf_ML4 = 15
val maxBins_ML4 = 2

val DTweather_ML4 = new DecisionTreeClassifier().setImpurity(impureza).setMaxDepth(maxProf_ML4).setMaxBins(maxBins_ML4)

val DTweatherAus_ML4 = DTweather_ML4.fit(WeatherFeaturesClaseDFLabel)

val predictionsAndLabelsDF_ML4 = DTweatherAus_ML4.transform(WeatherFeaturesClaseDFLabel).select("prediction", "label")

predictionsAndLabelsDF_ML4.show(3)


val metrics_ML4 = new MulticlassClassificationEvaluator()
metrics_ML4.setMetricName("accuracy")
val acierto_ML4 = metrics.evaluate(predictionsAndLabelsDF_ML4)
val error_ML4 = 1 - acierto_ML4


// Sin librería ML
//val errores = predictionsAndLabelsDF_ML4.map(x=>if(x(0)==x(1))0 else 1).collect.sum
//val error_ML4 = errores.toDouble/predictionsAndLabelsDF_ML4.count
//val acierto_ML4 = 1-error_ML4

println(f"Tasa de error ML4= $error_ML4%1.3f")

//Save model

// DTweatherAus_ML4.write.overwrite().save(PATH + "DTweatherAus_ML4")

// Charge the model

// DecisionTreeClassificationModel.load(PATH + "DTweatherAus_ML4")
