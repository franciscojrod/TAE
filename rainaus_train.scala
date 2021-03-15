// Variables numéricas

val columns = Seq("MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
                   "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Pressure9am", 
                   "Pressure3pm", "Humidity9am", "Humidity3pm", "Temp9am", "Temp3pm")

val weatherDF2_train = columns.foldLeft(weatherDF_train) { 
  (tempDF, colName) => {
   
    val quantiles = weatherDF_train.stat.approxQuantile(colName,Array(0.25, 0.5, 0.75),0.0)
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

weatherDF2_train.limit(5).show()



//variables categóricas

val columns2 = Seq("WindGustDir", "WindDir9am", "WindDir3pm", "RainToday")

val weatherDF3_train = columns2.foldLeft(weatherDF2_train) { 
  (tempDF, colName) => {
   
    val moda_array = weatherDF2_train.groupBy(colName).count().orderBy($"count".desc).withColumnRenamed(colName, "value").filter("value != 'null'").take(1)
    val moda = moda_array(0)(0)
    
    println(colName + " - moda : " + moda)
    
    tempDF.withColumn(
      colName,
      when(col(colName).isNull || col(colName) === "NA", moda)
      .otherwise(col(colName))
    )
  }  
}

weatherDF3_train.limit(5).show()



// variables categóricas 2 (cloud)

val columns3 = Seq("Cloud9am", "Cloud3pm")

val weatherDF4_train = columns3.foldLeft(weatherDF3_train) { 
  (tempDF, colName) => {
   
    val moda_array = weatherDF3_train.groupBy(colName).count().orderBy($"count".desc).withColumnRenamed(colName, "value").filter("value is not null").take(1)
    
    val moda = moda_array(0)(0)
    
    println(colName + " - moda : " + moda)
    
    tempDF.withColumn(
      colName,
      when(col(colName).isNull || col(colName) === "NA" || col(colName) > 8 || col(colName) < 0, moda)
      .otherwise(col(colName))
    )
  }  
}

weatherDF4_train.limit(5).show()
