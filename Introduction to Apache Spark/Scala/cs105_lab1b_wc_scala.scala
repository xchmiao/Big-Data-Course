// Databricks notebook source
// MAGIC %md
// MAGIC This notebook is intented to rewrite the CS150_lab1b_wordcount homework in Scala.

// COMMAND ----------

// MAGIC %md
// MAGIC ####  Part1: Creating a base DataFrame and performing operations

// COMMAND ----------

// MAGIC %md
// MAGIC ##### (1a) Create a DataFrame

// COMMAND ----------

val l = List("cat", "elephant", "rat", "rat", "cat")
val wordsDF = sqlContext.sparkContext.parallelize(l).toDF("word")
import spark.implicits._
wordsDF.printSchema()

// COMMAND ----------

display(wordsDF)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### (1b) Using DataFrame functions to add an 's'

// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.sql.Column

// COMMAND ----------

val pluralDF = wordsDF.select(concat(wordsDF("word"), lit("s")).alias("word"))
pluralDF.show()

// COMMAND ----------

// MAGIC %md 
// MAGIC ##### (1c) Length of each word

// COMMAND ----------

val pluralLengthsDF = pluralDF.select($"word", length($"word").alias("length"))
pluralLengthsDF.show()

// COMMAND ----------

pluralDF.select("word").collect().map(y => y(0).toString.length).foreach(println)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Part 2: Counting with Spark SQL and DataFrames

// COMMAND ----------

// MAGIC %md
// MAGIC ##### (2a) Using `groupBy` and `count`

// COMMAND ----------

val wordCountsDF = (wordsDF
                    .groupBy("word").count())
wordCountsDF.show()

// COMMAND ----------

// MAGIC %md
// MAGIC #### Part 3: Finding unique words and a mean value

// COMMAND ----------

// MAGIC %md
// MAGIC ##### (3a) Unique Words

// COMMAND ----------

val uniqueWordsCount = wordsDF.distinct().count()
println(uniqueWordsCount)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### 3(b) Means of groups using DataFrames

// COMMAND ----------

val averageCount = (wordCountsDF
                    .groupBy()
                    .mean("count")).first()(0)
println(averageCount)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Part 4: Apply word count to a file

// COMMAND ----------

// MAGIC %md
// MAGIC ##### (4a) Define the `wordCount` Function

// COMMAND ----------

def wordCount(df: org.apache.spark.sql.DataFrame): org.apache.spark.sql.DataFrame = {df.groupBy("word").count()}

wordCount(wordsDF).show()

// COMMAND ----------

// MAGIC %md
// MAGIC ##### (4b) Capitalization and punctuation

// COMMAND ----------

wordsDF("word").getClass

// COMMAND ----------

import org.apache.spark.sql._
def removePunctuation(column: Column): Column = {
  var no_punc = regexp_replace(column, "\\p{Punct}", "")
  var trim_col = trim(no_punc)
  var low = lower(trim_col)
  return low
}

// COMMAND ----------

val sentenceDF = sqlContext.sparkContext.parallelize(Seq(("Hi, you!"), (" No under_score!"),(" *      Remove punctuation then spaces  * "))).toDF("sentence")

sentenceDF.select(removePunctuation(col("sentence"))).show(truncate = false)

// COMMAND ----------

// MAGIC %md
// MAGIC ** (4c) Load a text file **

// COMMAND ----------

val fileName = "dbfs:/databricks-datasets/cs100/lab1/data-001/shakespeare.txt"

val shakespeareDF = sqlContext.read.text(fileName).select(removePunctuation(col("value")).alias("value"))
shakespeareDF.show(15, truncate = false)

// COMMAND ----------

// MAGIC %md
// MAGIC ** (4d) Words from lines **
// MAGIC 
// MAGIC Beofe we can use the `wordcount()` function, we have to address two issues with the formate of the DataFrame:
// MAGIC   + The first issue is that we need to split each line by its spaces.
// MAGIC   + The second issue is we need to filter out empyt lines or words.
// MAGIC   
// MAGIC Apply a transformation that will split each 'sentence' in DataFrame by its space, and then transform from a DataFrame that contains lists of words into a DataFrame with each word in its own row. To accomplish these two task you can use `split` and `explode` functions. 
// MAGIC 
// MAGIC Once you have DataFrame with one word per row, you can apply the DataFrame operation `where` to remove the rows taht contain ''.

// COMMAND ----------

val shakeWordsDF = shakespeareDF
                   .select(split($"value", " ").alias("value"))
                   .select(explode($"value").alias("word"))
                   //.filter(length($"word") > 0)
                   //.filter($"word" =!= "") // In scala, for column wise string comparision, using === or =!=.
                   .filter("word != ''")
shakeWordsDF.show()
val shakeWordsDFCount = shakeWordsDF.count()

// COMMAND ----------

// MAGIC %md
// MAGIC ** (4e) Count the words **
// MAGIC 
// MAGIC We now have a DataFrame that is only words. Next, let's apply the wordCount() function to produce a list of word counts. We can view the first 20 words by using `show()` actions. Put the words in descending order by using `orderBy` or `sort` to first sort the DataFrame. 

// COMMAND ----------

val topWordsAndCountsDF = shakeWordsDF
                          .groupBy("word").count()
                          //.sort($"count".desc)
                          .orderBy(desc("count"))
topWordsAndCountsDF.show(20)

// COMMAND ----------


