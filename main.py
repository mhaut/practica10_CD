import matplotlib.pyplot as plt
import pandas as pd
from pyspark import SparkConf, SparkContext, sql
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import *

conf = SparkConf().setAppName("machine lerning API")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
sqlc = sql.SQLContext(sc)



def get_model(modeltr):
	if modeltr == "LR":
		from pyspark.ml.classification import LogisticRegression
		model = LogisticRegression(featuresCol="features", 
								labelCol="label", 
								regParam=0.1, 
								elasticNetParam=0.1, 
								maxIter=10000)
	elif modeltr == "MLP":
		pass
		#from pyspark.ml.classification import MultilayerPerceptronClassifier

		#layers = [784, 100, 20, 10]
		#perceptron = MultilayerPerceptronClassifier(maxIter=1000, layers=layers, blockSize=128, seed=1234)
		#perceptron_model = perceptron.fit(training)

		#from time import time

		#start_time = time()
		#perceptron_model = perceptron.fit(training)
		#test_pred = perceptron_model.transform(testing)
		#print("Accuracy:", evaluator.evaluate(test_pred))
		#print("Time taken: %d" % (time() - start_time))
	else:
		print("MODEL", modeltr, "don't exist")
		sc.stop()
		exit()
	return model




df_training = (sqlc
               .read
               .options(header = False, inferSchema = True)
               .csv("mnist_train.csv"))
print df_training.count()

#print("No of columns: ", len(df_training.columns), df_training.columns)

feature_culumns = ["_c" + str(i+1) for i in range(784)]
#print(feature_culumns)

vectorizer = VectorAssembler(inputCols=feature_culumns, outputCol="features")
training = (vectorizer
            .transform(df_training)
            .select("_c0", "features")
            .toDF("label", "features")
            .cache())
training.show()


a = training.take(1)[0].features.toArray()
plt.imshow(a.reshape(28, 28), cmap="Greys")
plt.show()


images = training.sample(False, 0.01, 1).take(25)
fig, _ = plt.subplots(5, 5, figsize = (10, 10))
for i, ax in enumerate(fig.axes):
    r = images[i]
    label = r.label
    features = r.features
    ax.imshow(features.toArray().reshape(28, 28), cmap = "Greys")
    ax.set_title("True: " + str(label))
plt.tight_layout()
plt.show()


counts = training.groupBy("label").count()
counts_df = counts.rdd.map(lambda r: {"label": r['label'],  "count": r['count']}).collect()
pd.DataFrame(counts_df).set_index("label").sort_index().plot.bar()
plt.show()


df_testing = (sqlc
              .read
              .options(header = False, inferSchema = True)
              .csv("mnist_test.csv"))
testing = (vectorizer
           .transform(df_testing)
           .select("_c0", "features")
           .toDF("label", "features")
           .cache())


model = get_model("LR")


model_trained = model.fit(training)
test_pred = model_trained.transform(testing).withColumn("matched", expr("label == prediction"))
test_pred.show()


evaluator = MulticlassClassificationEvaluator(labelCol="label", 
                                               predictionCol="prediction", 
                                               metricName="accuracy")

test_pred.withColumn("matched", expr("cast(matched as int)"))\
 .groupby("label")\
 .agg(avg("matched"))\
 .orderBy("label")\
 .show()

print("ACCURACY:", evaluator.evaluate(test_pred) * 100.0, "%")





