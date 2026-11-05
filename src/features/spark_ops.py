"""Small shared Spark column helpers for feature modules."""
from pyspark.sql import Column
from pyspark.sql import functions as F


def safe_ratio(numerator: Column, denominator: Column) -> Column:
    """Null out ratios instead of producing inf/NaN when the denominator is <= 0."""
    return F.when(denominator > 0, numerator / denominator).otherwise(F.lit(None).cast("double"))
