# PySpark NLP virtual workshop

Instructions and code for the workshop "From Big Data to NLP Insights: Unlocking the Power of PySpark and Spark NLP"

## Setup

### Databricks community edition

We will run the training code on Databricks Community Edition. Create your account by following the [instructions provided in the official documentation](https://docs.databricks.com/getting-started/community-edition.html). Please complete this step before moving forward.

### Databricks workspace

You can now create a Databricks workspace with the required Jupyter notebooks [using this link](). The steps for doing this can be seen in the below GIF.  

From the left-hand side navbar, click on `Workspace` > click on dropdown >  click on `Import` > choose `URL` option and enter the link > click on `Import`.

![](databricks_workspace_import_steps.gif)

### Compute cluster

#### Cluster creation

We will now create a compute cluster that we will run our code on.

- Click on the Compute tab on the navbar. Then click on "Create Compute" button. You will be taken to the "New Cluster" configuration view.
- Assign the cluster a name. From the "Databricks runtime version" dropdown, choose "Runtime: 12.2 LTS (Scala 2.12, Spark 3.3.2).
- Click on the "Spark" tab. Add the following lines to "Spark config" field.
```
spark.kryoserializer.buffer.max 2000M
spark.serializer org.apache.spark.serializer.KryoSerializer
```
- Click on "Create Cluster". It may take a few minutes before the cluster gets created.

![databricks_cluster_creation](https://user-images.githubusercontent.com/4419448/237058641-e67762bc-e459-4586-857c-0851f611a218.gif)

At this point, you can successfully run the code in module 1's notebook. For the next 2 modules, we need to install the Spark NLP library in our cluster.

#### Spark NLP installation

In Libraries tab inside your cluster you need to follow these steps:

- Install New -> PyPI -> spark-nlp -> Install
- Install New -> Maven -> Coordinates -> com.johnsnowlabs.nlp:spark-nlp_2.12:4.4.1 -> Install

Voila! You're all set to start now.

## Code

The workshop code is distributed across 3 Jupyter notebooks. Each of these correspond to a workshop module. They are:

- Module 1: Basics of PySpark and the DataFrame API
- Module 2: PySpark for NLP
- Module 3: Advanced NLP with Spark NLP

They should be in your workspace if you have successfully completed the setup steps. They are present in this repository too if you want to go through them after the workshop.

_Note: A conceptual introduction to Jupyter notebooks can be found [here](https://www.databricks.com/glossary/jupyter-notebook)._
