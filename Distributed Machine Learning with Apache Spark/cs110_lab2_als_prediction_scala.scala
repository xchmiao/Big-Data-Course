// Databricks notebook source
// MAGIC %md
// MAGIC ## Part 0: Preliminaries

// COMMAND ----------

val dbfs_dir = "databricks-datasets/cs110x/ml-20m/data-001"
val ratings_filename = dbfs_dir + "/ratings.csv"
val movies_filename = dbfs_dir + "/movies.csv"

// COMMAND ----------

display(dbutils.fs.ls(dbfs_dir))

// COMMAND ----------

import org.apache.spark.sql.types._

val ratings_df_schema = new StructType(Array(
                        StructField("userId", IntegerType),
                        StructField("movieId", IntegerType),
                        StructField("rating", DoubleType))
                        )


val movies_df_schema = new StructType(Array(
                       StructField("ID", IntegerType),
                       StructField("title", StringType))
                       )

// COMMAND ----------

// MAGIC %md
// MAGIC ### Load and Cache

// COMMAND ----------

import org.apache.spark.sql.functions.regexp_extract

// COMMAND ----------

val ratings_df = sqlContext.read.format("com.databricks.spark.csv")
                                .options(Map("header" -> "true", "inferSchema" -> "false"))
                                .schema(ratings_df_schema)
                                .load(ratings_filename)

//val ratings_df = raw_ratings_df.drop("Timestamp") // since in the schema definition, timestamp is not defined as a column, so it won't be loaded

val movies_df = sqlContext.read.format("com.databricks.spark.csv")
                                .option("header", true)
                                .option("inferSchema", false)
                                .schema(movies_df_schema)
                                .load(movies_filename)
//Genres is not defined in the schema, so it won't be loaded

ratings_df.cache
movies_df.cache

println(f"Total rows in ratings_df is ${ratings_df.count}.")
println(f"Total rows in movies_df is ${movies_df.count}")

ratings_df.show(5, truncate = false)
movies_df.show(5, truncate = false)

// COMMAND ----------

// MAGIC %md
// MAGIC ### (1a) Movies with Highest Average Ratings

// COMMAND ----------

import org.apache.spark.sql.functions._

val movie_ids_with_avg_ratings = ratings_df
                                 .groupBy("movieId")
                                 .agg(avg("rating").alias("average"), count("*").alias("count"))
                                 .join(movies_df, ratings_df("movieId") === movies_df("ID"))
                                 .select("movieId", "title", "average", "count")

//movie_ids_with_avg_ratings.filter($"movieId".isin(1831, 431, 631)).show(5)
//movie_ids_with_avg_ratings.filter("movieId in (1831, 431, 631)").show(5)
//movie_ids_with_avg_ratings.where("movieId in (1831, 431, 631)").show(5)

movie_ids_with_avg_ratings.select($"movieId", $"title", format_number(col("average"), 2).as("average"), $"count").show(5, truncate = false)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Movies with Highest Average Ratings and at least 500 reviews

// COMMAND ----------

val movies_with_500_ratings_or_more = movie_ids_with_avg_ratings.orderBy($"average".desc).filter($"count" >= 500)
movies_with_500_ratings_or_more.show(10, false)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Part 2: Collaborative Filtering

// COMMAND ----------

// MAGIC %md
// MAGIC ### 2(a) Creating a Training Set

// COMMAND ----------

val Array(training_df, validation_df, test_df) = ratings_df.randomSplit(Array(0.6, 0.2, 0.2), seed = 1800009193L)

training_df.cache
validation_df.cache
test_df.cache

println(f"Traning: ${training_df.count}, validation: ${validation_df.count}, test: ${test_df.count}")

training_df.show(3)
validation_df.show(3)
test_df.show(3)

// COMMAND ----------

// MAGIC %md
// MAGIC ### (2b) Alternating Least Squares

// COMMAND ----------

import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator

val seed = 1800 //009193L

val als = new ALS()
              .setUserCol("userId")
              .setItemCol("movieId")
              .setRatingCol("rating")
              .setRegParam(0.1)
              .setSeed(seed)

//create an RMSE evaluation using the label and predicted columns

val reg_eval = new RegressionEvaluator()
                   .setPredictionCol("prediction")
                   .setLabelCol("rating")
                   .setMetricName("rmse")

val tolerance = 0.03
val ranks = Array(4, 8, 12)
//val ranks = Array(4)
val errors = new Array[Double](3)
val models = new Array[ALSModel](3)

var err = 0
var min_error = Double.PositiveInfinity
var best_rank_index = -1

for(rank <- ranks){
  als.setRank(rank)
  val model = als.fit(training_df)
  val predict_df = model.transform(validation_df).select("*")
  
  //Remove NaN values from prediction
  //val predicted_ratings_df = predict_df.filter(!col("prediction").isNaN)
  val predicted_ratings_df = predict_df.filter(col("prediction") =!= Double.NaN) //doesn't matter much
  //predicted_ratings_df.filter($"prediction".isNull).show(4)
  
  //Run RMSE evaluator
  errors(err) = reg_eval.evaluate(predicted_ratings_df)
  models(err) = model
  
  println(f"For rank ${rank}, the RMSE is ${errors(err)}")
  
  if (errors(err) < min_error){
    min_error = errors(err)
    best_rank_index = err
    //println(best_rank_index)
  }
    
  err = err + 1
}

als.setRank(ranks(best_rank_index))
println(f"The best model was trained with rank ${ranks(best_rank_index)}")
val my_model = models(best_rank_index)

// COMMAND ----------

// MAGIC %md
// MAGIC ### (2c) Testing Your Model

// COMMAND ----------

val predicted_test_df = my_model.transform(test_df).select("*").filter(col("prediction") =!= Double.NaN)

val test_RMSE = reg_eval.evaluate(predicted_test_df)

println(f"The model had a RMSE on the test set of ${test_RMSE}%.4f")

// COMMAND ----------

// MAGIC %md
// MAGIC ### (2d) Comparing Your Model

// COMMAND ----------

val training_avg_rating = training_df.groupBy().agg(avg("rating").as("avg_rating")).collect()(0)(0)
println(f"The average rating from the training set is ${training_avg_rating.asInstanceOf[Double]}%.2f") //Need to convert the datatype to double

val test_for_avg_df = test_df.withColumn("prediction", lit(training_avg_rating))

val test_avg_RMSE = reg_eval.evaluate(test_for_avg_df)
println(f"The RMSE on the average set is ${test_avg_RMSE}%.4f")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Part 3: Predictions for Yourself

// COMMAND ----------

// MAGIC %md
// MAGIC ### (3a) Your Movie Ratings

// COMMAND ----------

import org.apache.spark.sql.Row

val my_user_id = 0
case class My_record(userId: Int, movieId: Int, rating: Int)

val my_rated_movies = Seq(My_record(my_user_id, 318, 5), My_record(my_user_id, 858, 4),
                          My_record(my_user_id, 1221, 3), My_record(my_user_id, 527, 5),
                          My_record(my_user_id, 2959, 5), My_record(my_user_id, 58559, 5),
                          My_record(my_user_id, 593, 2), My_record(my_user_id, 2324, 3),
                          My_record(my_user_id, 930, 1), My_record(my_user_id, 260, 1)
                         )

val my_ratings_df = sc.parallelize(my_rated_movies).toDF()
println("My movie ratings:")
my_ratings_df.show(20)

// COMMAND ----------

// MAGIC %md
// MAGIC ### (3b) Add Your Movies to Training Dataset

// COMMAND ----------

val training_with_my_ratings_df = training_df.union(my_ratings_df)

training_with_my_ratings_df.count - training_df.count

// COMMAND ----------

// MAGIC %md
// MAGIC ### (3c) Train a Model with Your Ratings

// COMMAND ----------

//Reset the parameters for als

als.setPredictionCol("prediction")
   .setMaxIter(5)
   .setSeed(seed)
   .setUserCol("userId")
   .setItemCol("movieId")
   .setRatingCol("rating")

val my_ratings_model = als.fit(training_with_my_ratings_df)

// COMMAND ----------

// MAGIC %md
// MAGIC ### (3d) Check RMSE for the New Model with Your Ratings

// COMMAND ----------

val predicted_test_my_ratings_df = my_ratings_model.transform(test_df).select("*").filter($"prediction" =!= Double.NaN)

val test_RMSE_my_ratings = reg_eval.evaluate(predicted_test_my_ratings_df)

println(f"The model had a RMSE on the test set of ${test_RMSE_my_ratings}%.4f.")

// COMMAND ----------

// MAGIC %md
// MAGIC ### (3e) Predict Your Ratings

// COMMAND ----------

val my_rated_movies_list = my_rated_movies.map(x => x.movieId)

val not_rated_df = movies_df.filter(!$"ID".isin(my_rated_moives :_*))

val my_unrated_movies_df = not_rated_df.select($"ID".as("movieId"), lit(my_user_id).alias("userId"))

val predicted_rating_df = my_ratings_model.transform(my_unrated_movies_df).filter($"prediction" =!= Double.NaN)

// COMMAND ----------

predicted_rating_df.show(10)

// COMMAND ----------

val predicted_with_counts_df = predicted_rating_df.join(movie_ids_with_avg_ratings, movie_ids_with_avg_ratings("movieId") === predicted_rating_df("movieId"))
val predicted_highest_rated_movies_df = predicted_with_counts_df.orderBy($"prediction".desc).filter("count > 75")

predicted_highest_rated_movies_df.show(25, false)

// COMMAND ----------

spark.stop

// COMMAND ----------


