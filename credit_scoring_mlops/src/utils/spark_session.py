"""
Initialisation d'une session Spark pour le préprocessing et le traitement de données volumineuses.
"""

from pyspark.sql import SparkSession

def get_spark_session(app_name: str = "CreditScoringPipeline"):
    """
    Crée et retourne une session Spark configurée pour le projet.
    
    Args:
        app_name (str): Nom de l'application Spark.
    
    Returns:
        SparkSession: instance de SparkSession.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.executor.memory", "2g")
        .config("spark.driver.memory", "2g")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )
    
    spark.sparkContext.setLogLevel("WARN")  # réduire le bruit des logs Spark
    return spark

# -----------------------------
# Exemple d'utilisation
# -----------------------------
if __name__ == "__main__":
    spark = get_spark_session()
    print("✅ Session Spark initialisée :", spark)
