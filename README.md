# Portfolio Optimizatiion with Mosek

This code provides an example how to estimate the minimum variance
and the minimum Value-at-Risk portfolio.
It uses the [Mosek](https://mosek.com/) software package for which
a free personal academic licence can be obtained [here](https://license.mosek.com/academic/).

## Example

The file `portfolio_optimization.py` downloads the Industry Portfolio data
from Kenneth R. French's [website](http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/) using
the [pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest/)
for python.
It then estimates the sample mean and covariance of the returns and
proceeds with estimating the portfolio weights subject to
the required return, either with or without shortselling.

