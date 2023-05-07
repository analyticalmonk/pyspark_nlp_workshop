# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC In this module, we'll discuss using PySpark for NLP tasks such as entity recognition and sentiment analysis. We'll cover how to load, preprocess, and analyze text data using PySpark. We'll also discuss when to use PySpark for NLP tasks and when to consider other Python NLP libraries.
# MAGIC
# MAGIC We'll introduce Spark NLP, a popular NLP library built on top of PySpark. The hands-on exercise will demonstrate how to perform text preprocessing and feature extraction with Spark NLP.
# MAGIC
# MAGIC **Install Spark NLP library**
# MAGIC
# MAGIC First, let's install the Spark NLP library in our Databricks environment:

# COMMAND ----------

!pip install spark-nlp

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's import the necessary libraries for this module:

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover, CountVectorizer, IDF
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import sparknlp
from sparknlp.annotator import *
from sparknlp.base import *

# COMMAND ----------

# MAGIC %fs
# MAGIC
# MAGIC ls databricks-datasets/amazon

# COMMAND ----------

dbutils.fs.ls('/amazon')

# COMMAND ----------

# MAGIC %md
# MAGIC **Load a text dataset as a DataFrame**
# MAGIC
# MAGIC Let's load a text dataset from the Databricks file system as a DataFrame:

# COMMAND ----------

text_data_path = "dbfs:/databricks-datasets/amazon/data20K/"
text_df = spark.read.parquet(text_data_path, header=True, inferSchema=True)

# COMMAND ----------

text_df.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC **Preprocess and analyze text data using PySpark**
# MAGIC
# MAGIC Now, we'll preprocess and analyze the text data using PySpark. First, tokenize the text.
# MAGIC
# MAGIC Tokenization is the process of breaking text into individual words or tokens. It's one of the essential steps in NLP to convert unstructured text data into a structured format.

# COMMAND ----------

from pyspark.ml.feature import Tokenizer

tokenizer = Tokenizer(inputCol="review", outputCol="tokens")
tokenized_df = tokenizer.transform(text_df)


# COMMAND ----------

tokenized_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC Stop words are common words that don't carry much meaning and are often removed from text data to reduce noise and computational complexity. Examples of stop words are "a", "an", "the", "and", etc.

# COMMAND ----------

remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
filtered_df = remover.transform(tokenized_df)

# COMMAND ----------

filtered_df.show(5)

# COMMAND ----------

cv = CountVectorizer(inputCol="filtered_tokens", outputCol="raw_features")
cv_model = cv.fit(filtered_df)
featurized_df = cv_model.transform(filtered_df)

idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(featurized_df)
result_df = idf_model.transform(featurized_df)

# COMMAND ----------

result_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Build a Spark NLP pipeline**
# MAGIC
# MAGIC A pipeline is a sequence of NLP operations applied to text data. In Spark NLP, you create a pipeline by chaining together various annotators and transformers.

# COMMAND ----------

document_assembler = DocumentAssembler.setInputCol("text").setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("tokens")

normalizer = Normalizer() \
    .setInputCols(["tokens"]) \
    .setOutputCol("normalized")

lemmatizer = LemmatizerModel.pretrained() \
    .setInputCols(["normalized"]) \
    .setOutputCol("lemmas")

finisher = Finisher() \
    .setInputCols(["lemmas"]) \
    .setCleanAnnotations(False)

pipeline = Pipeline(stages=[
    document_assembler,
    tokenizer,
    normalizer,
    lemmatizer,
    finisher
])

# COMMAND ----------

# MAGIC %md
# MAGIC - DocumentAssembler: It is the first stage in the Spark NLP pipeline, converting input text data into Spark NLP "Document" format.
# MAGIC - Tokenizer: It takes the "Document" as input and tokenizes it, breaking the text into individual words or tokens.
# MAGIC - Normalizer: It removes punctuations, numbers, and any other non-alphabetic characters from the tokens, resulting in clean tokens.
# MAGIC - Lemmatizer: It reduces words to their base or dictionary form, also known as lemmas. This step helps in standardizing words with similar meanings to their base form, which can improve text analysis.
# MAGIC - Finisher: It is the final stage in the Spark NLP pipeline, converting the output of the previous annotators and transformers into a DataFrame format that can be used for further analysis or machine learning tasks.
# MAGIC
# MAGIC
# MAGIC Transform the text DataFrame using the Spark NLP pipeline:

# COMMAND ----------

pipeline_model = pipeline.fit(text_df)
processed_df = pipeline

# COMMAND ----------

processed_df.sow()