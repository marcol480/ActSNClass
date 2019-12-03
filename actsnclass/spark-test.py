from pyspark.sql import SparkSession
from pyspark import StorageLevel
from pyspark.sql import functions as F
from pyspark.sql.functions import randn
from pyspark.sql.types import IntegerType,FloatType
from pyspark.sql.functions import pandas_udf, PandasUDFType

spark = SparkSession.builder.getOrCreate()
sc=spark.sparkContext
print("Spark session started")

#a class for timing
from time import time
class Timer:
    """
    a simple class for printing time (s) since last call
    """
    def __init__(self):
        self.t0=time()

    def start(self):
        self.t0=time()

    def stop(self):
        t1=time()
        print("{:2.1f}s".format(t1-self.t0))

timer=Timer()


#import os
#dirfits=os.environ['FITSDIR']
timer.start()
#gal=spark.read.format("fits").option("hdu",1)\
#         .load(dirfits)\
#         .select(F.col("RA"), F.col("Dec"), (F.col("Z_COSMO")+F.col("DZ_RSD")).alias("z"))
timer.stop()


#gal.printSchema()

#timer.start()
#gal.show(5)
#timer.stop()

