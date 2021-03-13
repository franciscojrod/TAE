println("\n\n******************* Partición aleatoria *******************")

val dataSplits = weatherDF.randomSplit(Array(0.7, 0.3), seed=0)
val rainaus_train = dataSplits(0)
val rainaus_test = dataSplits(1)

rainaus_train.count()
rainaus_test.count()
 import org.apache.spark.sql.{functions => F}

weatherDF.select("Temp9am").describe().show()

val quantiles = rainaus_train.stat.approxQuantile("Temp9am",Array(0.25,0.75),0.0)
val Q1 = quantiles(0)
val Q3 = quantiles(1)
val IQR = Q3 - Q1

val lowerRange = Q1 - 1.5*IQR
val upperRange = Q3+ 1.5*IQR

//val outliers = rainaus_train.filter(s"Temp9am < $lowerRange or Temp9am > $upperRange") 
var weatherDF2 = weatherDF
weatherDF2.dtypes.foreach {  f =>
  val fName = f._1
  val fType = f._2
  var n = 0
  if (fType  == "IntegerType" || fType  == "DoubleType") { 
    //println(s"STRING_TYPE")
    println(fName) 
    val quantiles = rainaus_train.stat.approxQuantile(fName,Array(0.25,0.75),0.0)
    val Q1 = quantiles(0)
    val Q3 = quantiles(1)
    val IQR = Q3 - Q1

    val lowerRange = Q1 - 1.5*IQR
    val upperRange = Q3+ 1.5*IQR
    println(lowerRange)
    println(upperRange)
    var n = weatherDF2.withColumn(fName + "_new", when(weatherDF2(fName) > upperRange, upperRange).when(weatherDF2(fName) < lowerRange, lowerRange).when(weatherDF2(fName).isNull, lowerRange).otherwise(col(fName)))
    weatherDF2.show()
    n++
    //weatherDF.withColumn("newcol" + fName, when(col(fName) > upperRange, upperRange).when(col(fName) < lowerRange, lowerRange).otherwise(weatherDF["newcol"+fName]))
  }
  if (fType  == "DoubleType") { 
    //println(s"STRING_TYPE")
    println(fName) 
    val quantiles = rainaus_train.stat.approxQuantile(fName,Array(0.25,0.75),0.0)
    val Q1 = quantiles(0)
    val Q3 = quantiles(1)
    val IQR = Q3 - Q1

    val lowerRange = Q1 - 1.5*IQR
    val upperRange = Q3+ 1.5*IQR
    println(lowerRange)
    println(upperRange)
  }
  //println("Name %s Type:%s - all:%s".format(fName , fType, f))
}

