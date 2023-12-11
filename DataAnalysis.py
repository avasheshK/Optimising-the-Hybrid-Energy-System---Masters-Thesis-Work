import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.stats as stats
import numpy as np
from solarproductionforecast import *
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from fetchprices import *
import matplotlib.pyplot as plt

def generate_gamma_pdf(data, k=10):
    """
    Fit a gamma distribution to the data and plot the resulting PDF.
    """
    shape, loc, scale = stats.gamma.fit(data[data > 0], floc=0)  # Fit gamma distribution
    x = np.linspace(min(data), max(data), 1000)
    y = stats.gamma.pdf(x, shape, loc=0, scale=scale)  # PDF of the fitted gamma distribution
    
    plt.plot(x, y, label='Gamma PDF')
    plt.fill_between(x, 0, y, alpha=0.2)  # Optional: fill under the curve
    plt.title('Gamma Distribution of Solar Production Data')
    plt.xlabel('Solar Production (MW)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()


# Load your dataset
# Replace this with the path to your CSV file
prices = fetch_spot_prices()["Price(EUR/MW)"].values

# Define a list of distributions to test
distributions = [st.norm, st.lognorm, st.expon, st.gamma, st.beta, st.uniform]

# Function to calculate Akaike Information Criterion (AIC)
def calculate_aic(n, ll, k):
    return 2 * k - 2 * ll + 2 * k * (k + 1) / (n - k - 1)

# Store the results
results = []

# Estimate parameters and fit for each distribution
for distribution in distributions:
    # Fit the distribution to the data
    params = distribution.fit(prices)
    
    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    
    # Calculate the log likelihood
    ll = np.sum(distribution.logpdf(prices, loc=loc, scale=scale, *arg))
    
    # Calculate the number of parameters (including loc and scale)
    k = len(params)
    
    # Calculate AIC
    aic = calculate_aic(len(prices), ll, k)
    
    # Append the result
    results.append((distribution, aic, params))

# Sort results by AIC
results.sort(key=lambda x: x[1])

# Best fitting distribution
best_distribution, _, best_params = results[0]

print(best_distribution)
