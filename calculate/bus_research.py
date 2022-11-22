import numpy as np
import time

import dask
from dask.distributed import Client, progress

client = Client()
print( "Workers:" + str(len( list(client.scheduler_info()['workers']) )) )

cars_arr = np.arange(10)
    
def fun_step(x):
    time.sleep(1)
    return 3

results = []
for cars in cars_arr:
    fun = dask.delayed(fun_step)(cars)
    results.append(fun)

results = dask.persist(*results)
progress(results)
results.compute()

client.close()

