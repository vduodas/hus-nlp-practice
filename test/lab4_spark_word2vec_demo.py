import re
from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec
from pyspark.sql.functions import col, lower, regexp_replace, split


def main():
    # Initialize Spark Session
    try:
        spark = SparkSession.builder \
        .appName("SparkWord2VecDemo") \
        .master("local[*]") \
        .getOrCreate()

        spark.sparkContext.setLogLevel("WARN")

        # Load dataset (JSON Lines format)
        data_path = "C:\\Users\\DoubleDD\\HUS\\NLP&DL\\datasets\\c4-train.00000-of-01024-30K.json"
        # check to ensure that the path is correct
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = f.readline()
                print("First line of dataset:", data)
        except FileNotFoundError:
            print(f"Dataset not found at path: {data_path}")
            return
        
        
        print("Loading dataset...")
        df = spark.read.json(data_path)

        # We are interested in the 'text' field
        text_df = df.select("text")

        # ===== Preprocessing =====
        # 1. lowercase
        # 2. remove punctuation
        # 3. split into words
        processed_df = (
            text_df
            .select(lower(col("text")).alias("text"))
            .select(regexp_replace(col("text"), r"[^a-z\s]", "").alias("text"))
            .select(split(col("text"), r"\s+").alias("words"))
        )

        # ===== Train Word2Vec =====
        word2vec = Word2Vec(
            vectorSize=100,
            windowSize=5,
            minCount=5,
            inputCol="words",
            outputCol="result",
            maxIter=5
        )

        print("Training Word2Vec model with Spark...")
        model = word2vec.fit(processed_df)

        # ===== Demonstration =====
        print("\nTop 5 words similar to 'computer':")
        model.findSynonyms("computer", 5).show(truncate=False)

    except Exception as e:
        print(f"Spark rror: {e}")
        
    finally:
        # Stop Spark
        spark.stop()


if __name__ == "__main__":
    main()
