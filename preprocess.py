from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, substring_index
from pyspark.sql.types import ArrayType, DoubleType, StringType
import numpy as np
from PIL import Image
import os
import glob

# --- CONFIGURATION ---
BASE_DIR = "/home/sat3812/final_project"
DATA_DIR = os.path.join(BASE_DIR, "mini_malware_dataset")
LABELS_PATH = "file://" + os.path.join(BASE_DIR, "trainLabels.csv")
OUTPUT_PATH = "file://" + os.path.join(BASE_DIR, "processed_malware_data")

def process_file_path(filepath):
    """
    Reads a file directly from disk and converts to image.
    """
    try:
        # Open file manually to avoid loading it into JVM
        with open(filepath, 'rb') as f:
            content = f.read()
            
        # Decode bytes to string
        content_str = content.decode('utf-8', errors='ignore')

        # Parse Hex
        hex_data = []
        # Limit to first 10MB to prevent crashes on massive files
        lines = content_str[:10000000].strip().split('\n') 
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 1:
                clean_bytes = [b for b in parts[1:] if b != '??']
                hex_data.extend([int(b, 16) for b in clean_bytes])

        if not hex_data: return None

        # Reshape & Resize
        width = int(len(hex_data)**0.5)
        if width == 0: return None
        
        rem = len(hex_data) % width
        if rem != 0: hex_data = hex_data[: -rem]
        
        img_array = np.array(hex_data, dtype=np.uint8).reshape(width, -1)
        img = Image.fromarray(img_array).resize((64, 64))
        
        # Return flattened array
        return [float(x) for x in np.array(img).flatten()]
        
    except Exception:
        return None

def run_preprocessing():
    spark = SparkSession.builder \
        .appName("MalwareClassification_LowMem") \
	.master("spark://hadoop1:7077") \
        .config("spark.driver.memory", "4g") \
	.config("spark.executor.memory", "4g") \
        .getOrCreate()

    print("========== STARTING V4 ==========")

    # 1. Get list of files using Python
    files = glob.glob(os.path.join(DATA_DIR, "*.bytes"))
    print(f"1. Found {len(files)} files in {DATA_DIR}")
    
    if len(files) == 0:
        print("ERROR: No files found! Check DATA_DIR path.")
        spark.stop()
        return

    # 2. Create DataFrame from Paths
    rdd = spark.sparkContext.parallelize(files, numSlices=32)
    df = rdd.map(lambda x: (x,)).toDF(["path"])

    # 3. Join Labels
    df = df.withColumn("Id", substring_index(substring_index(col("path"), "/", -1), ".", 1))
    
    labels_df = spark.read.option("header", "true").csv(LABELS_PATH)
    df = df.join(labels_df, on="Id", how="inner")
    
    # Rename Class column
    df = df.withColumnRenamed("Class", "label").withColumn("label", col("label").cast("string"))
    
    print(f"2. Labels joined. Processing {df.count()} files...")

    # 4. Apply UDF to Path (Not Content)
    process_udf = udf(process_file_path, ArrayType(DoubleType()))
    
    df_processed = df.withColumn("features", process_udf(col("path"))) \
                     .filter(col("features").isNotNull()) \
                     .select("label", "features")

    # 5. Save
    print("3. Saving to Parquet...")
    df_processed.write.mode("overwrite").parquet(OUTPUT_PATH)
    
    print("========== SUCCESS ==========")
    spark.stop()

if __name__ == "__main__":
    run_preprocessing()
