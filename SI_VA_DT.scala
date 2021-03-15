import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel

// Obtenemos el nombrede las columnasde carDF, salvo la clase

val attributeColumns= Seq("Date", "Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "RainTomorrow").toArray

// Generamos los nombres de las nuevas columnas
val outputColumns = attributeColumns.map(_ + "-num").toArray

val siColumns= new StringIndexer().setInputCols(attributeColumns).setOutputCols(outputColumns).setStringOrderType("alphabetDesc")

// Creamos el StringIndexerModel
val simColumns= siColumns.fit(weatherDF4_train)

val weatherDFnumeric= simColumns.transform(weatherDF4_train).drop(attributeColumns:_*)

weatherDFnumeric.show(5)


// VectorAssembler

import org.apache.spark.ml.feature.VectorAssembler


val va= new VectorAssembler().setOutputCol("features").setInputCols(weatherDFnumeric.columns.diff(Array("RainTomorrow-num"))) 

val weatherFeaturesClaseDF= va.transform(weatherDFnumeric).select("features", "RainTomorrow-num")

import org.apache.spark.ml.feature.StringIndexer
// creamos el StringIndexerpara la clase
val indiceClase= new StringIndexer().setInputCol("RainTomorrow-num").setOutputCol("label").setStringOrderType("alphabetDesc")


// Creamos el DataFramecarFeaturesLabelDFcon columnas features y label
val weatherFeaturesLabelDF= indiceClase.fit(weatherFeaturesClaseDF).transform(weatherFeaturesClaseDF).drop("RainTomorrow-num")

val dataSplits= weatherFeaturesLabelDF.randomSplit(Array(0.66, 0.34), seed=0)
val trainWeatherDF= dataSplits(0)
val testWeatherDF= dataSplits(1)

Instancia de decision tree
/* Importamos de ML*/
import org.apache.spark.ml.classification.DecisionTreeClassifier
/* Creamos una instancia de DecisionTreeClassifier*/
val DTweather=new DecisionTreeClassifier()

// Lo he puesto para que no pite pero hay que revisarlo
 DTweather.setMaxBins(3395)

/* Entrenamos el modelo: Árbol de Decisión con los parámetros por defecto  */
val DTweatherModel=DTweather.fit(trainWeatherDF)

// Para examinar el arbol
DTweatherModel.toDebugString

/* Predecimos la clase de los ejemplos de prueba*/
val predictionsAndLabelsDF= DTweatherModel.transform(testWeatherDF).select("prediction", "label")

Evaluación

/* Importamos de ML*/ import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
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
