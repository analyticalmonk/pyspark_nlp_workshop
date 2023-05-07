# Databricks notebook source
# MAGIC %md
# MAGIC Today, we're going to dive into the basics of PySpark and the DataFrame API. Our goal is to set up and get familiar with PySpark API, focusing on the DataFrame API and advanced data operations such as filtering, joining, aggregating, and grouping. We'll also discuss what use cases PySpark is a good fit for.
# MAGIC
# MAGIC **Starting a SparkSession**
# MAGIC
# MAGIC To begin, let's start a SparkSession in Databricks:

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("PySpark Advanced Basics") \
    .getOrCreate()

# COMMAND ----------

# MAGIC %fs
# MAGIC
# MAGIC ls databricks-datasets/wine-quality

# COMMAND ----------

dbutils.fs.ls('/databricks-datasets')

# COMMAND ----------

# MAGIC %md
# MAGIC This code sets up a new SparkSession with the name "PySpark Advanced Basics" and makes it available as the spark variable.
# MAGIC
# MAGIC Now, let's load a dataset from the Databricks file system as a DataFrame. 
# MAGIC
# MAGIC The "Record Linkage Comparison Patterns" dataset consists of 5,749,132 record pairs derived from an epidemiological cancer registry. The dataset is split into 10 blocks, each containing information on the degree of agreement between record pairs on various attributes such as names, date of birth, and postal codes. We will start with 1 of these blocks.

# COMMAND ----------

data_path = "dbfs:/databricks-datasets/wine-quality/winequality-red.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True, sep=";")

# COMMAND ----------

# MAGIC %md
# MAGIC The above code reads a CSV file from the specified path and creates a DataFrame called df with the schema inferred from the data.

# COMMAND ----------

df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC **Nulls**

# COMMAND ----------

from pyspark.sql.functions import col, sum

null_counts = df.select([sum(col(c).isNull().cast('int')).alias(c) for c in df.columns]).collect()
null_counts = {c: null_counts[0][c] for c in df.columns}
print(null_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC 1. `col(c).isNull().cast('int')`: For each column `c`, the `isNull()` function returns a boolean value indicating whether the value is null or not. We then cast this boolean value to an integer, where `True` becomes `1` and `False` becomes `0`. This creates a DataFrame where the null values are represented as `1` and non-null values as `0`.
# MAGIC
# MAGIC 2. `[sum(col(c).isNull().cast('int')).alias(c) for c in df.columns]`: We use a list comprehension to apply the previous step for all columns in the DataFrame `df`. The `sum()` function is used to aggregate the 1s and 0s, calculating the total number of null values for each column. The `alias(c)` function is used to keep the original column name for the resulting DataFrame.
# MAGIC
# MAGIC 3. `df.select(...)`: We use the `select()` function to create a new DataFrame with the aggregated null counts for each column.
# MAGIC
# MAGIC 4. `null_counts = ...collect()`: The `collect()` function is used to retrieve the result of the null count calculation as a list of Row objects. In this case, there will only be one Row object because we've aggregated the data.
# MAGIC
# MAGIC 5. `{c: null_counts[0][c] for c in df.columns}`: We use a dictionary comprehension to convert the Row object into a dictionary, where the keys are the column names and the values are the null counts for each column.
# MAGIC
# MAGIC 6. `print(null_counts)`: Finally, we print the dictionary containing the null counts for each column.

# COMMAND ----------

# MAGIC %md
# MAGIC **Sorting**
# MAGIC
# MAGIC Sort the records based on the pH and residual sugar amount.

# COMMAND ----------

from pyspark.sql.functions import desc

sorted_df = df.orderBy(desc("pH"), desc("residual sugar"))
sorted_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Filtering**
# MAGIC
# MAGIC Filter the records to keep only those within the ideal pH range

# COMMAND ----------

filtered_df = df.filter((df["pH"] >= 3.4) & (df["pH"] <= 3.6))
filtered_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Window functions**
# MAGIC
# MAGIC Next, let's look at window functions. They allow us to perform calculations across sets of rows that are related to the current row.  
# MAGIC In this example, we use a window function:

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, desc

window_spec = Window.partitionBy("quality").orderBy(desc("alcohol"))
ranked_df = df.withColumn("rank", row_number().over(window_spec))
ranked_df.show()


# COMMAND ----------

# MAGIC %md
# MAGIC The output DataFrame contains an additional "rank" column. The rank column assigns a unique rank to each row within the group of rows with the same "quality". The ranking is determined by the "alcohol" column, with higher values receiving a higher rank (1 being the highest rank in each group).

# COMMAND ----------

# MAGIC %md
# MAGIC **Summary statistics**

# COMMAND ----------

summary = df.describe()

summary.select("summary", "fixed acidity", "residual sugar").show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Using SQL queries**
# MAGIC
# MAGIC Lastly, we'll cover how to use SQL queries with DataFrames.  
# MAGIC Here, we create a temporary view called "people" from our DataFrame df and then execute an SQL query to calculate the average age for each occupation group.

# COMMAND ----------

# Register the DataFrame as a temporary view
df.createOrReplaceTempView("wine")

# Example: Counting the number of matches and non-matches
results = spark.sql("SELECT quality, COUNT(*) as count FROM wine GROUP BY quality")
results.show()

# COMMAND ----------

# MAGIC %md
# MAGIC In this module, we've covered the basics of PySpark and the DataFrame API. We've learned how to set up a SparkSession, load data, perform data operations, and use SQL queries. Now that you're familiar with the DataFrame API, we'll move on to using PySpark for natural language processing tasks in the next module.