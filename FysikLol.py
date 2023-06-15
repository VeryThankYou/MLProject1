import numpy as np
import math

data = np.array([1.242, 1.214, 1.213, 1.215, 1.251, 1.205, 1.220, 1.228, 1.219, 1.240])
mean = np.mean(data)
sd = np.std(data)
usikkerhed = sd/math.sqrt(10)
print(mean)
print(usikkerhed)

