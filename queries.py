from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, when, collect_list, size, count, countDistinct, desc, trim

# Initialize Spark session
spark = SparkSession.builder.appName("Nodes_and_Edges").getOrCreate()

# Read data
nodes = spark.read.option("header", True).option("sep", "\t").csv("nodes.tsv")
edges = spark.read.option("header", True).option("sep", "\t").csv("edges.tsv")

# Filter compound (drugs) and disease nodes
compounds = nodes.filter(col('kind') == 'Compound')
diseases = nodes.filter(col('kind') == 'Disease')

# Treatments: all edges we care about
treatments = edges.filter(col('metaedge').isin('CtD', 'CbG', 'CuG', 'CdG', 'CpD'))

# Join compounds (drugs) with treatments
joined_df = compounds.join(treatments, compounds['id'] == treatments['Source'], 'inner')

# Also join diseases to get disease names if needed
disease_join = diseases.select(col('id').alias('disease_id'), col('name').alias('disease_name'))

# ---------------------------
# Q1: Drugs - number of genes and diseases
# ---------------------------
q1 = joined_df.groupBy('id', 'name').agg(
    sum(when(col('metaedge').isin('CtD', 'CpD'), 1).otherwise(0)).alias('num_diseases'),
    sum(when(col('metaedge').isin('CbG', 'CuG', 'CdG'), 1).otherwise(0)).alias('num_genes')
)

result_q1 = q1.orderBy(col('num_genes').desc())

print("\nQ1: Top 5 drugs by number of genes associated")
result_q1.select("id", "name", "num_genes", "num_diseases").show(5, truncate=False)

# ---------------------------
# Q2: Diseases - number of drugs associated
# ---------------------------
# Join treatments where target is disease
disease_edges = treatments.join(compounds, treatments['Source'] == compounds['id'], 'inner')

# Group by target (disease) and count distinct drugs
q2 = disease_edges.groupBy('target').agg(
    countDistinct('Source').alias('num_drugs')   # Count DISTINCT drugs per disease
)

# Group by the number of drugs and count how many diseases are associated with that number
q2_grouped = q2.groupBy('num_drugs').agg(
    count('target').alias('num_diseases')
)

# Now order by the number of diseases in descending order
result_q2 = q2_grouped.orderBy(col('num_diseases').desc())

print("\nQ2: Top 5 groups by number of diseases associated with x drugs")
result_q2.show(5)

# ---------------------------
# Q3: Names of drugs with top 5 gene associations
# ---------------------------
# Get top 5 drugs by number of genes associated
top_5_genes = result_q1.limit(5)

# Now get the names and the number of genes associated with each drug
result_q3 = top_5_genes.select("name", "num_genes")

print("\nQ3: Top 5 drug names by number of genes associated")
result_q3.show(truncate=False)

# Stop spark
spark.stop()
