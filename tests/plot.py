import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

x = np.linspace(0, 2, 200)

mu = 1.159218
variance = 1.454032 
sigma = math.sqrt(variance)
plt.plot(x, stats.norm.pdf(x, mu, sigma))

mu = 1.173241
variance = 1.454886 
sigma = math.sqrt(variance)
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))

mu = 1.149089
variance = 1.541856
sigma = math.sqrt(variance)
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.show()

# 1.159218 1.454032
# 1.173241 1.454886 
# 1.149089 1.541856