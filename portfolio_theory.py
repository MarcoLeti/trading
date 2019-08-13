from __future__ import print_function
import config as cf
import intrinio_sdk
from intrinio_sdk.rest import ApiException
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tickers = ['GS', 'MSFT', 'AAPL']
n = len(tickers)

start_date = '2017-01-01'
end_date = '2017-12-31'

ticker_prices = {}
ticker_close_price = {}

intrinio_sdk.ApiClient().configuration.api_key['api_key'] = cf.sandbox_api_key

security_api = intrinio_sdk.SecurityApi()

for stock in tickers:
    identifier = stock # str | A Security identifier (Ticker, FIGI, ISIN, CUSIP, Intrinio ID)
    start_date = start_date # date | Return prices on or after the date (optional)
    end_date = end_date # date | Return prices on or before the date (optional)
    frequency = 'daily' # str | Return stock prices in the given frequency (optional) (default to daily)
    page_size = 100 # int | The number of results to return (optional) (default to 100)
    next_page = '' # str | Gets the next page of data from a previous API call (optional)

    try:
      api_response = security_api.get_security_stock_prices(identifier, start_date=start_date, end_date=end_date, frequency=frequency, page_size=page_size, next_page=next_page)
      pprint(api_response)
    except ApiException as e:
      print("Exception when calling SecurityApi->get_security_stock_prices: %s\r\n" % e)
    
    ticker_prices[stock] = pd.DataFrame(api_response.stock_prices_dict).sort_values('date')

    ticker_close_price[stock] = ticker_prices[stock][['date', 'close']]
    ticker_close_price[stock].rename(columns={'close':stock}, inplace=True)
    
prices = ticker_close_price[tickers[0]]
for i in range(1, n):
    prices = pd.merge(prices, ticker_close_price[tickers[i]], on='date', how='inner')
    
# calculate daily returns
for i in range(0, n):
    prices[tickers[i] + '_R'] = prices[tickers[i]][1:] / prices[tickers[i]][:-1].values - 1

# calculate expected return as average of daily return
R = np.array([prices[tickers[0] + '_R'].mean()])
for i in range(1, n):
    R = np.concatenate([R, np.array([prices[tickers[i] + '_R'].mean()])])
R.shape = (n, 1)

cov = prices.filter(like='_R').cov().values
cov_inv = np.linalg.inv(cov)

ones = np.ones(n)
ones.shape = (n, 1)

A = np.dot(np.transpose(ones), cov_inv).dot(ones)
B = np.dot(np.transpose(R), cov_inv).dot(ones)
C =  np.dot(np.transpose(R), cov_inv).dot(R)

Rg = B/A    # expected return of the global minimum variance portfolio
varg = 1/A  # global minimum portfolio variance
wg = np.dot((1/A) * cov_inv, ones) # weights of the global minimum variance portfolio

# create combination of mean-variance on the efficient frontier
r = np.arange(Rg, Rg + 0.01, 0.001)
r.shape = (len(r), 1)
var = ((A * r**2) - (2 * B * r) + C) / (A * C - B**2)

# calculate weigths for the choosen return r
r = 0.005
lmbda = (C - B * r) / (A * C - B**2)
gmma = ((A * r) - B) / (A * C - B**2)
w = np.dot(lmbda * cov_inv, ones) + np.dot(gmma * cov_inv, R)

#expected return
np.dot(np.transpose(w), R)

# book example
R = np.array([[1], [2], [3]])
cov = np.array([[1, 1, 0], [1, 4, 0], [0, 0, 9]])

plt.plot(np.sqrt(var), r, '-o')
plt.show()

def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)


def random_portfolio(returns):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))
    
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    
    # This recursion reduces outliers to keep plots pretty
    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma

n_portfolios = 500
means, stds = np.column_stack([
    random_portfolio(return_vec) 
    for _ in xrange(n_portfolios)
])