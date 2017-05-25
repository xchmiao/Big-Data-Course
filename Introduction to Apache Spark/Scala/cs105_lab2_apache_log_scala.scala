// Databricks notebook source
import java.util.Calendar
import org.apache.spark.SparkContext._
import org.apache.spark.sql.functions._

// COMMAND ----------

val throwaway_df = sc.parallelize(Seq(("Anthony", 10), ("Julia", 20), ("Fred", 5))).toDF("name", "count")
display(throwaway_df)

// COMMAND ----------

val s = "abcdefabcok"
val pattern = "(?<=abc)\\w{2,3}".r
pattern.findAllIn(s).toList

// COMMAND ----------

println("This was last run on: %s".format(Calendar.getInstance().getTime()))
println("Local time is: %s".format(java.time.LocalDate.now))

// COMMAND ----------

// MAGIC %md
// MAGIC ### Part 2: Exploratory Data Analysis

// COMMAND ----------

// MAGIC %md
// MAGIC #### (2a) Loading the log file

// COMMAND ----------

val log_file_path = "dbfs:/databricks-datasets/cs100/lab2/data-001/apache.access.log.PROJECT"

// COMMAND ----------

val base_df = sqlContext.read.text(log_file_path)
base_df.printSchema()

// COMMAND ----------

base_df.show(truncate = false)

// COMMAND ----------

// MAGIC %md
// MAGIC #### (2b) Parsing the log file

// COMMAND ----------

val split_df = base_df.select(regexp_extract($"value", "^([^\\s]+)\\s", 1).alias("host"),
                              regexp_extract($"value", "^.*\\[(\\d\\d/\\w{3}/\\d{4}:\\d{2}:\\d{2}:\\d{2}\\s-\\d{4})", 1).alias("timestamp"),
                              //regexp_extract($"value", "\"\\w+\\s*(/\\w*.*)\\s+HTTP.*" , 1).alias("path")
                              regexp_extract($"value", "\"\\w+\\s*([^\\s]+)\\s+HTTP.*", 1).alias("path"),
                              regexp_extract($"value", "^.*\".*\"\\s+(\\d+)", 1).cast("integer").alias("status"),
                              regexp_extract($"value", "(\\d+)$", 1).cast("integer").alias("content_size")
                            )
split_df.show(truncate = false)

// COMMAND ----------

// MAGIC %md
// MAGIC #### (2c) Data Cleaning

// COMMAND ----------

base_df.filter(base_df("value").isNull).count()

// COMMAND ----------

// MAGIC %md
// MAGIC If our parsing worked properly, we'll have no rows with null column values. Let's check.

// COMMAND ----------

val bad_rows_df = split_df.filter(split_df("host").isNull||
                                  split_df("timestamp").isNull||
                                  split_df("path").isNull||
                                  split_df("status").isNull||
                                  split_df("content_size").isNull
                                 )
bad_rows_df.count()

// COMMAND ----------

// MAGIC %md
// MAGIC Let's find out which column(s) is(are) affectedl

// COMMAND ----------

def count_null(col_name: String): org.apache.spark.sql.Column ={
  sum(col(col_name).isNull.cast("integer")).alias(col_name)
}

var expr = Array[org.apache.spark.sql.Column]()
for (col_name <- split_df.columns){
  expr = expr :+ count_null(col_name)
}

split_df.agg(expr(0), expr(1), expr(2), expr(3), expr(4)).show()

// COMMAND ----------

// MAGIC %md
// MAGIC Diagnosie the issue: bad rows are ended with "_" instead of a real number

// COMMAND ----------

base_df.filter(!base_df("value").rlike("\\d+$")).show(false)

// COMMAND ----------

// MAGIC %md
// MAGIC #### (2d) Fix the rows with null content_size

// COMMAND ----------

val cleaned_df = split_df.na.fill(0, Seq("content_size"))
cleaned_df.filter($"content_size".isNull).count()

// COMMAND ----------

// MAGIC %md
// MAGIC #### (2e) Parsing the timestamp
// MAGIC 
// MAGIC Okay, now that we have a clean, parsed DataFrame, we have to parse the timestamp field into an actual timestamp. The Common Log Format time is somewhat non-standard. A User-Defined Function (UDF) is the most straightforward way to parse it.

// COMMAND ----------

val month_map = Map ("Jan" -> 1, 
                     "Feb" -> 2,
                     "Mar" -> 3,
                     "Apr" -> 4, 
                     "May" -> 5, 
                     "Jun" -> 6,
                     "Jul" -> 7,
                     "Aug" -> 8,
                     "Sep" -> 9,
                     "Oct" -> 10,
                     "Nov" -> 11,
                     "Dec" -> 12
                    )

def parse_clf_time(s: String): String = {
  "%1$02d-%2$02d-%3$04d %4$02d:%5$02d:%6$02d".format(
  s.substring(7, 11).toInt,
  month_map(s.substring(3, 6)),
  s.substring(0, 2).toInt,
  s.substring(12, 14).toInt,
  s.substring(15, 17).toInt,
  s.substring(18, 20).toInt
  )
}
var s = "01/Aug/1995:00:00:01 -0400"
println(parse_clf_time(s))

// COMMAND ----------

val u_parse_time = udf(parse_clf_time(_:String))

val logs_df = cleaned_df.select($"*", u_parse_time(split_df("timestamp")).cast("timestamp").alias("time")).drop("timestamp")
display(logs_df)

// COMMAND ----------

logs_df.cache() // cache the dataframe, since we are going to use it quite a bit from now on

// COMMAND ----------

// MAGIC %md
// MAGIC ### Part 3: Analysis Walk-Throught on the Web Server Log File

// COMMAND ----------

// MAGIC %md
// MAGIC #### (3a) Example: Content Size Statistics

// COMMAND ----------

logs_df.describe("content_size", "status").show()

// COMMAND ----------

logs_df.select(min($"content_size"), max($"content_size"), mean($"content_size")).show()

// COMMAND ----------

logs_df.agg(min($"content_size"), max($"content_size"), avg($"content_size")).show()

// COMMAND ----------

// MAGIC %md
// MAGIC #### (3b) Example: HTTP Status Analysis

// COMMAND ----------

val status_to_count_df = (logs_df
                          .groupBy("status")
                          .count()
                          //.sort($"count".desc)
                          .sort("status")
                          .cache()
                         )
status_to_count_df.show()

// COMMAND ----------

display(status_to_count_df)

// COMMAND ----------

val log_status_to_count_df = status_to_count_df.withColumn("log_count", log(status_to_count_df("count")))
display(log_status_to_count_df)

// COMMAND ----------

// MAGIC %md
// MAGIC #### (3d) Example: Frequent Host
// MAGIC 
// MAGIC Let's look at hosts that have accessed the server frequently (e.g., more than ten times). As with the response code analysis in (3b), we create a new DataFrame by grouping `successLogsDF` by the 'host' column and aggregating by count.
// MAGIC 
// MAGIC We then filter the result based on the count of accesses by each host being greater than ten.  Then, we select the 'host' column and show 20 elements from the result.

// COMMAND ----------

val host_sum_df = logs_df
                  .groupBy("host")
                  .count()
                  .filter($"count" > 10)
                  .orderBy($"count".desc)
                  .select("host")
println("The top 20 most frequently-visited hosts are:")
host_sum_df.show(truncate = false)

// COMMAND ----------

// MAGIC %md
// MAGIC #### (3e) Example: Visualizing Paths

// COMMAND ----------

val path_df = logs_df
              .groupBy("path")
              .count()
              .sort($"count".desc)
display(path_df)

// COMMAND ----------

// MAGIC %md
// MAGIC #### (3f) Example: Top Paths

// COMMAND ----------

path_df.show(10, truncate = false)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Part 4: Analyzing Web Server Log

// COMMAND ----------

// MAGIC %md
// MAGIC #### (4a) Top 10 Error Path
// MAGIC 
// MAGIC What are the top ten paths which did not have return code 200? Create a sorted list containing the paths and the number of times that they were accessed with a non-200 return code and show the top ten.
// MAGIC 
// MAGIC Think about the steps that you need to perform to determine which paths did not have a 200 return code, how you will uniquely count those paths and sort the list.

// COMMAND ----------

val error_path_df = logs_df
                    .filter($"status" =!= 200)
                    .groupBy("path")
                    .count()
                    .orderBy($"count".desc)
error_path_df.show(10, false)

// COMMAND ----------

// MAGIC %md
// MAGIC #### (4b) Number of Unique Hosts

// COMMAND ----------

printf("Number of unique hosts is : %d", logs_df.select("host").distinct.count)

// COMMAND ----------

// MAGIC %md
// MAGIC #### (4c) Exercise: Number of Unique Daily Hosts

// COMMAND ----------

val daily_host_df = logs_df.select($"host", dayofmonth($"time").alias("day"))
                    .distinct()
                    .groupBy("day")
                    .count()
                    .orderBy($"day")
display(daily_host_df)

// COMMAND ----------

// MAGIC %md
// MAGIC #### (4d) Exercise: Average Number of Daily Requests per Host
// MAGIC 
// MAGIC *Since the log only covers a single month, you can skip checking for the month.*

// COMMAND ----------

val avg_daily_req_per_host_df = logs_df.select($"host", dayofmonth($"time").alias("day"))
                                .groupBy("host", "day").count()
                                .groupBy("day").mean("count").sort($"day")
                                .select($"day", $"avg(count)".alias("req_per_host"))
avg_daily_req_per_host_df.show()

// COMMAND ----------

display(avg_daily_req_per_host_df)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Part 5: Exploring 404 Status Codes
// MAGIC 
// MAGIC Let's drill down and explore the error 404 status records. We've all seen those "404 Not Found" web pages. 404 errors are returned when the server cannot find the resource (page or object) the browser or client requested.

// COMMAND ----------

// MAGIC %md
// MAGIC ### (5a) Exercise: Counting 404 Response Codes
// MAGIC 
// MAGIC Create a DataFrame containing only log records with a 404 status code. Make sure you `cache()` `not_found_df` as we will use it in the rest of this exercise.
// MAGIC 
// MAGIC How many 404 records are in the log?

// COMMAND ----------

val not_found_df = logs_df.filter($"status" === 404)
not_found_df.count()
not_found_df.cache

// COMMAND ----------

// MAGIC %md
// MAGIC ### (5b) Exercise: Listing 404 Status Code Records
// MAGIC 
// MAGIC Using the DataFrame containing only log records with a 404 status code that you cached in part (5a), print out a list up to 40 _distinct_ paths that generate 404 errors.
// MAGIC 
// MAGIC **No path should appear more than once in your list.**

// COMMAND ----------

not_found_df.select("path").distinct.show(40, false)

// COMMAND ----------

List(row(0) for (row <-unique_not_found_paths_df.take(40)))

// COMMAND ----------

val unique_path_list = not_found_df.select("path").distinct.take(40)
val x = List.tabulate(40)(n => unique_path_list(n)(0).toString)
x.foreach(println)

// COMMAND ----------

// MAGIC %md
// MAGIC ### (5d) Exercise: Listing the Top Twenty-five 404 Response Code Hosts
// MAGIC 
// MAGIC Instead of looking at the paths that generated 404 errors, let's look at the hosts that encountered 404 errors. Using the DataFrame containing only log records with a 404 status codes that you cached in part (5a), print out a list of the top twenty-five hosts that generate the most 404 errors.

// COMMAND ----------

val hosts_404_count_df = logs_df
                         .filter($"status" === 404)
                         .groupBy("host")
                         .count()
                         .sort($"count".desc)
hosts_404_count_df.show(25, false)

// COMMAND ----------

// MAGIC %md
// MAGIC ### (5e) Exercise: Listing 404 Errors per Day
// MAGIC 
// MAGIC Let's explore the 404 records temporally. Break down the 404 requests by day (cache the `errors_by_date_sorted_df` DataFrame) and get the daily counts sorted by day in `errors_by_date_sorted_df`.
// MAGIC 
// MAGIC *Since the log only covers a single month, you can ignore the month in your checks.*

// COMMAND ----------

val errors_by_date_sorted_df = logs_df
                               .filter($"status" === 404)
                               .select($"host", dayofmonth($"time").alias("day"))
                               .groupBy("day")
                               .count()
                               .sort($"day")
errors_by_date_sorted_df.cache()
errors_by_date_sorted_df.show()

// COMMAND ----------

display(errors_by_date_sorted_df)

// COMMAND ----------

// MAGIC %md
// MAGIC #### (5g) Exercise: Top Five Days for 404 Errors

// COMMAND ----------

errors_by_date_sorted_df.sort($"count".desc).show(5, false)

// COMMAND ----------

// MAGIC %md
// MAGIC ### (5h) Exercise: Hourly 404 Errors
// MAGIC 
// MAGIC Using the DataFrame `not_found_df` you cached in the part (5a) and sorting by hour of the day in increasing order, create a DataFrame containing the number of requests that had a 404 return code for each hour of the day (midnight starts at 0). Cache the resulting DataFrame `hour_records_sorted_df` and print that as a list.

// COMMAND ----------

val hour_records_sorted_df = not_found_df
                             .select($"host", hour($"time").alias("hour"))
                             .groupBy("hour")
                             .count()
                             .sort($"hour")
hour_records_sorted_df.show()

// COMMAND ----------

display(hour_records_sorted_df)

// COMMAND ----------


