println("\n\n******************* Partición aleatoria *******************")

val dataSplits = weatherDF.randomSplit(Array(0.7, 0.3), seed=0)
val rainaus_train = dataSplits(0)
val rainaus_test = dataSplits(1)

rainaus_train.count()
rainaus_test.count()


weatherDF.select("Temp9am").describe().show()

val quantiles = rainaus_train.stat.approxQuantile("Temp9am",Array(0.25,0.75),0.0)
val Q1 = quantiles(0)
val Q3 = quantiles(1)
val IQR = Q3 - Q1

val lowerRange = Q1 - 1.5*IQR
val upperRange = Q3+ 1.5*IQR

//val outliers = rainaus_train.filter(s"Temp9am < $lowerRange or Temp9am > $upperRange") 
weatherDF.withColumn("Temp9am_new", when(weatherDF("Temp9am") > upperRange, upperRange).when(weatherDF("Temp9am") < lowerRange, lowerRange).when(weatherDF("Temp9am").isNull, lowerRange).otherwise(col("Temp9am"))).show()
/*weatherDF.dtypes.foreach {  f =>
  val fName = f._1
  val fType = f._2
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
    weatherDF.withColumn("newcol" + fName, when(col(fName) > upperRange, upperRange).when(col(fName) < lowerRange, lowerRange).otherwise(weatherDF["newcol"+fName]))
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
}*/

