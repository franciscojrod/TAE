println("\n\n******************* Partición aleatoria *******************")

val dataSplits = weatherDF.randomSplit(Array(0.7, 0.3), seed=0)
val rainaus_train = dataSplits(0)
val rainaus_test = dataSplits(1)

rainaus_train.count()
rainaus_test.count()

