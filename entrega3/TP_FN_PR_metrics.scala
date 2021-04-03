// False positive and false negative True positive true negative

val TP = lrModelApplied.filter($"prediction"===1 && $"label"===$"prediction").count()
val FP = lrModelApplied.filter($"prediction"===1 && $"label"=!=$"prediction").count()
val TN = lrModelApplied.filter($"prediction"===0 && $"label"===$"prediction").count()
val FN = lrModelApplied.filter($"prediction"===0 && $"label"=!=$"prediction").count()





// Tasa de acierto [TP + TN]/[𝑃 + 𝑁]


val TA = (TP + TN)/(TP + FP + TN + FN).toDouble

// Tasa de ciertos positivos (Recall)

val TCP = TP/(TP + FN).toDouble

// Tasa de falsos positivos

val TFP = FP/(FP + TN).toDouble

// Precision

val prec = TP/(TP + FP)

// PR Curve: Plot of Recall (x) vs Precision (y).


// https://github.com/apache/spark/blob/master/examples/src/main/scala/org/apache/spark/examples/mllib/BinaryClassificationMetricsExample.scala


import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint

val lrModelLbFe = lrModelApplied.select("features", "label").rdd.map(row =>
  (row.getAs[Vector](0)(0), row.getAs[Double](1)))


val metrics = new BinaryClassificationMetrics(lrModelLbFe)

metrics.areaUnderPR()
