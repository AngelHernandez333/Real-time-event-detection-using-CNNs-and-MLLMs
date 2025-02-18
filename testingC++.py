import numpy as np
import time
def sum_python(arr):
    total = 0
    for x in arr:
        total += x
    return total

arr = np.random.rand(10**9)  # Un millón de elementos
start_time=time.time()
total=sum_python(arr)
time_py=time.time()-start_time
print(time_py*1000, ' ms')

import example

start_time=time.time()
total=example.sum_cpp(arr)
time_cpp=time.time()-start_time
print(time_cpp*1000, ' ms')

print(time_py/time_cpp)