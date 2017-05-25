// Databricks notebook source
// MAGIC %md
// MAGIC 
// MAGIC #Power Plant Machine Learning Pipeline Application
// MAGIC 
// MAGIC This notebook is an end-to-end exercise of performing Extract-Transform-Load and Exploratory Data Analysis on a real-world dataset, and then applying several different machine learning algorithms to solve a supervised regression problem on the dataset.
// MAGIC 
// MAGIC ** This notebook covers: **
// MAGIC * *Part 1: Business Understanding*
// MAGIC * *Part 2: Load Your Data*
// MAGIC * *Part 3: Explore Your Data*
// MAGIC * *Part 4: Visualize Your Data*
// MAGIC * *Part 5: Data Preparation*
// MAGIC * *Part 6: Data Modeling*
// MAGIC * *Part 7: Tuning and Evaluation*
// MAGIC 
// MAGIC *Our goal is to accurately predict power output given a set of environmental readings from various sensors in a natural gas-fired power generation plant.*

// COMMAND ----------

// MAGIC %md
// MAGIC ## Part 2: Extract-Transform-Load (ETL) Your Data
// MAGIC 
// MAGIC Now that we understand what we are trying to do, the first step is to load our data into a format we can query and use.  This is known as ETL or "Extract-Transform-Load".  We will load our file from Amazon S3.
// MAGIC 
// MAGIC Note: Alternatively we could upload our data using "Databricks Menu > Tables > Create Table", assuming we had the raw files on our local computer.
// MAGIC 
// MAGIC Our data is available on Amazon s3 at the following path:
// MAGIC 
// MAGIC ```
// MAGIC dbfs:/databricks-datasets/power-plant/data
// MAGIC ```
// MAGIC 
// MAGIC **To Do:** Let's start by printing a sample of the data.
// MAGIC 
// MAGIC We'll use the built-in Databricks functions for exploring the Databricks filesystem (DBFS)
// MAGIC 
// MAGIC Use `display(dbutils.fs.ls("/databricks-datasets/power-plant/data"))` to list the files in the directory

// COMMAND ----------

display(dbutils.fs.ls("/databricks-datasets/power-plant/data"))

// COMMAND ----------

println(dbutils.fs.head("/databricks-datasets/power-plant/data/Sheet1.tsv")) // to look at the first a few lines in the first file

// COMMAND ----------

// MAGIC %md
// MAGIC ### Exercise 2(a)
// MAGIC 
// MAGIC Now use scala spark to print the first 5 lines of the data.
// MAGIC 
// MAGIC First, create an RDD from the data using sc.textFile() to read the data into RDD.
// MAGIC Second, figure out how to use RDD `take()` method to extract the first 5 lines of the RDD and print each line.

// COMMAND ----------

val rawTextRdd = sc.textFile("/databricks-datasets/power-plant/data")
rawTextRdd.take(5).foreach(println)

// COMMAND ----------

// MAGIC %md
// MAGIC Our schema definition from UCI appears below:
// MAGIC - AT = Atmospheric Temperature in C
// MAGIC - V = Exhaust Vacuum Speed
// MAGIC - AP = Atmospheric Pressure
// MAGIC - RH = Relative Humidity
// MAGIC - PE = Power Output.  This is the value we are trying to predict given the measurements above.

// COMMAND ----------

// MAGIC %md
// MAGIC ### Exercise 2(b): Create a DataFrame from the data
// MAGIC 
// MAGIC *Hint*: Use `sqlContext.read.format().options().load()`

// COMMAND ----------

val powerPlantDF = sqlContext.read.format("com.databricks.spark.csv").options(Map("delimiter" -> "\t", "header" -> "true", "inferschema" -> "true")).load("/databricks-datasets/power-plant/data")
powerPlantDF.show(10)

// COMMAND ----------

powerPlantDF.printSchema // powerPlantDF.dtypes

// COMMAND ----------

// MAGIC %md
// MAGIC #### Part 2: Alternative Method to Load your Data

// COMMAND ----------

import org.apache.spark.sql.types._

// COMMAND ----------

// Define a customized schema for the table

val customSchema = StructType(Array(
    StructField("AT", DoubleType, true),
    StructField("V", DoubleType, true),
    StructField("AP", DoubleType, true),
    StructField("RH", DoubleType, true),
    StructField("PE", DoubleType, true))
    )

val altPowerPlantDF = sqlContext.read.format("com.databricks.spark.csv").options(Map("header" -> "true", "delimiter" -> "\t", "inferschema" -> "true")).schema(customSchema).load("/databricks-datasets/power-plant/data")

altPowerPlantDF.show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Part 3: Explore Your Data

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC First, let's register our DataFrame as an SQL table named `power_plant`.  Because you may run this lab multiple times, we'll take the precaution of removing any existing tables first.
// MAGIC 
// MAGIC We can delete any existing `power_plant` SQL table using the SQL command: `DROP TABLE IF EXISTS power_plant` (we also need to to delete any Hive data associated with the table, which we can do with a Databricks file system operation).
// MAGIC 
// MAGIC Once any prior table is removed, we can register our DataFrame as a SQL table using `DataFrame.registerTempTable` (deprenciated now) or `DataFrame.createOrReplaceTempView` (**different from pyspark**)
// MAGIC 
// MAGIC ### (3a)
// MAGIC 
// MAGIC **ToDo:** Execute the prepared code in the following cell.

// COMMAND ----------

sqlContext.sql("DROP TABLE IF EXISTS power_plant")
dbutils.fs.rm("dbfs:/user/hive/warehouse/power_plant", true)
powerPlantDF.createOrReplaceTempView("power_plant")

// COMMAND ----------

// MAGIC %md
// MAGIC ### (3b) SQL on table
// MAGIC After the dataframe has been registered as SQL table, we could execute the SQL on top of it.

// COMMAND ----------

// MAGIC %sql 
// MAGIC select * from power_plant
// MAGIC limit 10

// COMMAND ----------

// MAGIC %sql
// MAGIC desc power_plant -- describe the schema

// COMMAND ----------

val df = sqlContext.table("power_plant") // get the dataframe associate with a sql table using sqlContex.table()
display(df.describe())

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ## Part 4: Visualization

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC select AT as Temperature, PE as Power from power_plant

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC select PE as Power, V as ExhaustVacuum from power_plant

// COMMAND ----------

// MAGIC %sql
// MAGIC select PE as Power, AP as Pressure from power_plant;

// COMMAND ----------

// MAGIC %sql
// MAGIC select PE as power, RH as Humidity from power_plant;

// COMMAND ----------

// MAGIC %md
// MAGIC ## Part 5: Data Preparation

// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler

//val datasetDF = sqlContext.sql("select * from power_plant")
val datasetDF = sqlContext.table("power_plant")

val vectorizer = new VectorAssembler()
                .setInputCols(Array("AT", "V", "AP", "RH"))
                .setOutputCol("features")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Part 6: Data Modeling

// COMMAND ----------

// MAGIC %md
// MAGIC ### Exercise 6(a)
// MAGIC 
// MAGIC Use the `randomSplit()` method to divide up `datasetDF` into a trainingSetDF (80% of the input DataFrame) and a testSetDF (20% of the input DataFrame), and for reproducibility, use the *seed 1800009193L*. The cache each DataFrame in memory to maximize the performance.

// COMMAND ----------

val Array(trainingSetDF, testSetDF) = datasetDF.randomSplit(Array(0.8, 0.2), seed = 1800009193L)
trainingSetDF.cache
testSetDF.cache

// COMMAND ----------

// MAGIC %md
// MAGIC ### Exercise 6(b)

// COMMAND ----------

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.LinearRegressionModel

val lr = new LinearRegression()

println(lr.explainParams)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Exercise 6(c)
// MAGIC 
// MAGIC The first step is to set the parameters for the method:
// MAGIC - Set the name of the prediction column to "Predicted_PE"
// MAGIC - Set the name of the label column to "PE"
// MAGIC - Set the maximum number of iterations to 100
// MAGIC - Set the regularization parameter to 0.1
// MAGIC 
// MAGIC Next, we create the `ML Pipeline` and set the stages to the Vectorizer and Linear Regression learner we created earlier.
// MAGIC 
// MAGIC Finally, we create a model by training on `trainingSetDF`.

// COMMAND ----------

import org.apache.spark.ml.Pipeline

lr.setPredictionCol("Predicted_PE")
  .setLabelCol("PE")
  .setMaxIter(100)
  .setRegParam(0.1)

val lrPipeline = new Pipeline()
                 .setStages(Array(vectorizer, lr))

val lrModel = lrPipeline.fit(trainingSetDF)

// COMMAND ----------

val model = lrModel.stages(1).asInstanceOf[LinearRegressionModel]
val coefficients = model.coefficients
val intercept = model.intercept

val featuresNolabel = List.tabulate(datasetDF.columns.length -1)(n => if (datasetDF.columns(n) != "PE") datasetDF.columns(n))

// COMMAND ----------

var equation = f"y = $intercept"
for (n <- 0 to featuresNolabel.length - 1){
  var weight = coefficients(n)
  var col_name = featuresNolabel(n)
  var symbol = if (weight > 0) "+" else "-"
  equation += f"${symbol}" + f"${weight.abs}" + f" * ${col_name}"
}

println("The linear regression model is: " + equation)

// COMMAND ----------

// MAGIC %md 
// MAGIC ### Exercise 6(e)
// MAGIC 
// MAGIC Now let's see what our predictions look like given this model. We apply our Linear Regression model to the 20% of the data that we split from the input dataset. The output of the model will be a predicted Power Output column named "Predicted_PE".
// MAGIC 
// MAGIC - Run the next cell
// MAGIC - Scroll through the resulting table and notice how the values in the Power Output (PE) column compare to the corresponding values in the predicted Power Output (Predicted_PE) column

// COMMAND ----------

//Apply our LR model to the test data and predict power output

val predictionsAndLabelsDF = lrModel.transform(testSetDF).select("*")

display(predictionsAndLabelsDF)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Exercise 6(f): Model Evaluations by RMSE and R-squared

// COMMAND ----------

import org.apache.spark.ml.evaluation.RegressionEvaluator

val regEval = new RegressionEvaluator()
              .setPredictionCol("Predicted_PE")
              .setLabelCol("PE")
              .setMetricName("rmse")

val rmse = regEval.evaluate(predictionsAndLabelsDF)
println(f"The root-mean-squared-error on the test set is: ${rmse}%.2f")

// COMMAND ----------

val r2 = regEval.setMetricName("r2").evaluate(predictionsAndLabelsDF)

println(f"The R-squared value is: ${r2}%.2f")

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ### Exercise 6(h): Analyze the fitting errors

// COMMAND ----------

sqlContext.sql("drop table if exists Power_Plant_RSME_Evaluation")
dbutils.fs.rm("dbfs:/user/hive/warehouse/Power_Plant_RSME_Evaluation", true)

predictionsAndLabelsDF.selectExpr("PE", "Predicted_PE", "PE - Predicted_PE Residual_Error", f"(PE - Predicted_PE)/${rmse} Within_RSME").createOrReplaceTempView("Power_Plant_RSME_Evaluation")

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC select * from Power_Plant_RSME_Evaluation

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ### Exercise 6(k)

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC SELECT case when Within_RSME <= 1.0 AND Within_RSME >= -1.0 then 1
// MAGIC             when  Within_RSME <= 2.0 AND Within_RSME >= -2.0 then 2 else 3
// MAGIC        end RSME_Multiple, COUNT(*) AS count
// MAGIC FROM Power_Plant_RSME_Evaluation
// MAGIC Group BY case when Within_RSME <= 1.0 AND Within_RSME >= -1.0 then 1  when  Within_RSME <= 2.0 AND Within_RSME >= -2.0 then 2 else 3 end
// MAGIC ORDER BY case when Within_RSME <= 1.0 AND Within_RSME >= -1.0 then 1  when  Within_RSME <= 2.0 AND Within_RSME >= -2.0 then 2 else 3 end

// COMMAND ----------

// MAGIC %md
// MAGIC ## Part 7: Tuning and Evaluation

// COMMAND ----------

import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}

val regParam = List.tabulate(10)(n => (n+1)/100.0)

val paramGrid = new ParamGridBuilder()
                    .addGrid(lr.regParam, regParam)
                    .build()

val crossval = new CrossValidator()
                   .setEstimator(lrPipeline)
                   .setEvaluator(regEval)
                   .setNumFolds(3)
                   .setEstimatorParamMaps(paramGrid)

val cvModel = crossval.fit(trainingSetDF)//.bestModel

// COMMAND ----------

// MAGIC %md
// MAGIC ### Exercise 7(b)

// COMMAND ----------

val predictionsAndLabelsDF = cvModel.transform(testSetDF).select("*")

val rmseNew = regEval.setMetricName("rmse").evaluate(predictionsAndLabelsDF)

val r2New = regEval.setMetricName("r2").evaluate(predictionsAndLabelsDF)

println(f"Original Root Mean Squared Error: ${rmse}%.2f")
println(f"New Root Mean Squared Error: ${rmseNew}%.2f")
println(f"Old r2: ${r2}%.2f")
println(f"New r2: ${r2New}%.2f")

// COMMAND ----------

//cvModel

// COMMAND ----------

import org.apache.spark.ml.PipelineModel

println("The best regularization parameter is: " + cvModel.bestModel.asInstanceOf[PipelineModel].stages.last.asInstanceOf[LinearRegressionModel].getRegParam)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Exercise 7(c): Fitting the data with Decision Tree Regressor

// COMMAND ----------

import org.apache.spark.ml.regression.{DecisionTreeRegressor, DecisionTreeRegressionModel}

val dt = new DecisionTreeRegressor()
         .setLabelCol("PE")
         .setPredictionCol("Predicted_PE")
         .setFeaturesCol("features")
         .setMaxBins(100)

//println(dt.explainParams)

val dtPipeline = new Pipeline()
                 .setStages(Array(vectorizer, dt))

// COMMAND ----------

// MAGIC %md
// MAGIC ### Exercise 7(d)

// COMMAND ----------

//reuse the CrossValidator with new dtPipeline

val paramGrid = new ParamGridBuilder()
                    .addGrid(dt.maxDepth, Array(2, 3))
                    .build()

crossval.setEstimator(dtPipeline)
        .setEstimatorParamMaps(paramGrid)


val dtModel = crossval.fit(trainingSetDF)

// COMMAND ----------

val predictionsAndLabelsDF = dtModel.transform(testSetDF).select("*")

val rmseDT = regEval.setMetricName("rmse").evaluate(predictionsAndLabelsDF)
val r2DT = regEval.setMetricName("r2").evaluate(predictionsAndLabelsDF)

println(f"LR Room Mean Squared Error: ${rmseNew}%.2f")
println(f"DT Room Mean Squared Error: ${rmseDT}%.2f")
println(f"LR r2: ${r2New}%.2f")
println(f"DT r2: ${r2DT}%.2f")

// COMMAND ----------

println(dtModel.bestModel.asInstanceOf[PipelineModel].stages(1).asInstanceOf[DecisionTreeRegressionModel].toDebugString)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Exercise 7(f): Fitting the data with Random Forest Regressor

// COMMAND ----------

import org.apache.spark.ml.regression.{RandomForestRegressor, RandomForestRegressionModel}

val rf = new RandomForestRegressor()
             .setLabelCol("PE")
             .setPredictionCol("Predicted_PE")
             .setFeaturesCol("features")
             .setSeed(100088121L)
             .setMaxDepth(8)
             .setNumTrees(100)

val rfPipeline = new Pipeline()
                     .setStages(Array(vectorizer, rf))

// COMMAND ----------

val paramGrid = new ParamGridBuilder()
                    .addGrid(rf.maxBins, Array(50, 100))
                    .build()

crossval.setEstimator(rfPipeline)
        .setEstimatorParamMaps(paramGrid)

val rfModel = crossval.fit(trainingSetDF).bestModel

// COMMAND ----------

val predictionsAndLabelsDF = rfModel.transform(testSetDF).select("*")

val rmseRF = regEval.setMetricName("rmse").evaluate(predictionsAndLabelsDF)
val r2RF = regEval.setMetricName("r2").evaluate(predictionsAndLabelsDF)

println(f"LR Room Mean Squared Error: ${rmseNew}%.2f")
println(f"DT Room Mean Squared Error: ${rmseDT}%.2f")
println(f"RF Room Mean Squared Error: ${rmseRF}%.2f")
println(f"LR r2: ${r2New}%.2f")
println(f"DT r2: ${r2DT}%.2f")
println(f"RF Room Mean Squared Error: ${r2RF}%.2f")

// COMMAND ----------

println(rfModel.asInstanceOf[PipelineModel].stages(1).asInstanceOf[RandomForestRegressionModel].toDebugString)

// COMMAND ----------

rfModel.asInstanceOf[PipelineModel].stages(1).asInstanceOf[RandomForestRegressionModel].extractParamMap

// COMMAND ----------

rfModel.asInstanceOf[PipelineModel].stages(1).asInstanceOf[RandomForestRegressionModel].explainParams

// COMMAND ----------

spark.stop

// COMMAND ----------


