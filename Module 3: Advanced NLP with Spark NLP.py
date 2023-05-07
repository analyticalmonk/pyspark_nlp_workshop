# Databricks notebook source
!pip install spark-nlp

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import sparknlp
from sparknlp.annotator import *
from sparknlp.base import *

# COMMAND ----------

text_data_path = "dbfs:/databricks-datasets/amazon/data20K"
text_df = spark.read.parquet(text_data_path, header=True, inferSchema=True)

# COMMAND ----------

# NER pipeline to identify and classify named entities in the reviews:

document_assembler = DocumentAssembler() \
    .setInputCol("reviews") \
    .setOutputCol("document")

sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentences")

tokenizer = Tokenizer() \
    .setInputCols(["sentences"]) \
    .setOutputCol("tokens")

ner_model = NerDLModel.pretrained("ner_dl", "en") \
    .setInputCols(["sentences", "tokens"]) \
    .setOutputCol("ner_tags")

ner_converter = NerConverter() \
    .setInputCols(["sentences", "tokens", "ner_tags"]) \
    .setOutputCol("entities")

pipeline = Pipeline(stages=[
    document_assembler,
    sentence_detector,
    tokenizer,
    ner_model,
    ner_converter
])


# COMMAND ----------

pipeline_model = pipeline.fit(reviews_df)
ner_result_df = pipeline_model.transform(reviews_df)


# COMMAND ----------

# Extract the most common named entities:

named_entities = ner_result_df.select("entities.result").withColumnRenamed("result", "named_entities")
flatten = udf(lambda x: [item for sublist in x for item in sublist], StringType())
flat_entities = named_entities.select(flatten(col("named_entities")).alias("entities"))
top_entities = flat_entities.groupBy("entities").agg(count("*").alias("count")).sort(desc("count")).limit(10)
top_entities.show()


# COMMAND ----------

# Create a DataFrame with only the reviews containing the most common named entities:

top_entities_list = [row.entities for row in top_entities.collect()]
filter_udf = udf(lambda entities: any(entity in top_entities_list for entity in entities), StringType())
filtered_reviews_df = ner_result_df.filter(filter_udf(col("entities.result"))).select("reviews")


# COMMAND ----------

document_assembler = DocumentAssembler() \
    .setInputCol("reviews") \
    .setOutputCol("document")

sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentences")

tokenizer = Tokenizer() \
    .setInputCols(["sentences"]) \
    .setOutputCol("tokens")

sentiment_model = SentimentDLModel.pretrained("sentimentdl_use_imdb", "en") \
    .setInputCols(["sentences", "tokens"]) \
    .setOutputCol("sentiment")

pipeline = Pipeline(stages=[
    document_assembler,
    sentence_detector,
    tokenizer,
    sentiment_model
])


# COMMAND ----------

pipeline_model = pipeline.fit(filtered_reviews_df)
sentiment_result_df = pipeline_model.transform(filtered_reviews_df)
