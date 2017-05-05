#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from mosek.fusion import *
from scipy.stats import norm


def min_risk(mu, H, target_return=0.0, shortselling=True, verbose=False):
    """
    Minimum Variance Portfolio
         min w'Hw
            s.t.
                w'mu = target_return
                sum(w) = 1
                w >= 0 (if no shortselling)

    In conic form
        min s2
            s.t.
                w'mu = target_return
                sum(w) = 1
                w >= 0 (if no shortselling)
                s2 >= w'Hw <=> (1/2, s2, GW) \in Q_r^{n+2}
    """
    names = mu.index               # The asset names
    n = mu.size                    # Number of assets
    G = np.linalg.cholesky(H).T    # A matrix such that H = G'G
    if isinstance(mu, pd.Series):  # Store as plain np.array
        mu = mu.values

    M = Model('Min Risk')

    # Define portfolio weights
    if shortselling:
        w = M.variable('w', n, Domain.unbounded())
    else:
        w = M.variable('w', n, Domain.greaterThan(0.0))

    # Variance
    s2 = M.variable('s2', 1, Domain.greaterThan(0.0))

    # The objective
    M.objective('minvar', ObjectiveSense.Minimize, s2)

    # Full investment
    M.constraint('budget', Expr.sum(w), Domain.equalsTo(1.0))

    # Get at least the target return
    if target_return > 0.0:
        M.constraint('target', Expr.dot(mu, w), Domain.greaterThan(target_return))

    # Imposes a bound on the risk
    M.constraint('s2 > ||Gw||_2^2',
                 Expr.vstack(Expr.constTerm(1, 0.5),
                             s2.asExpr(),
                             Expr.mul(G, w)), Domain.inRotatedQCone())
    if verbose:
        M.setLogHandler(sys.stdout)

    M.solve()

    return pd.Series(w.level(), index=names)


def min_VaR(mu, H, target_return=0.0, shortselling=True, alpha=0.01, verbose=False):
    """
    Minimum Value-at-Risk Portfolio
         min -(w'mu + w'Hw * q)^1/2
            s.t.
                w'mu = target_return
                sum(w) = 1
                w >= 0 (if no shortselling)

    In conic form
        min t
            s.t.
                w'mu = target_return
                sum(w) = 1
                w >= 0 (if no shortselling)
                t >= -(w'mu + w'Hw * q)^1/2 <=> (-1/q * (t + w'mu), GW) \in Q^{n+1}
    """

    names = mu.index               # The asset names
    q = norm.ppf(alpha)            # The quantile
    n = mu.size                    # Number of assets
    G = np.linalg.cholesky(H).T    # A matrix such that H = G'G
    if isinstance(mu, pd.Series):  # Store as plain np.array
        mu = mu.values

    M = Model('Min VaR')

    # Define portfolio weights
    if shortselling:
        w = M.variable('w', n, Domain.unbounded())
    else:
        w = M.variable('w', n, Domain.greaterThan(0.0))

    # The objective
    t = M.variable('t', 1, Domain.greaterThan(0.0))
    M.objective('min risk', ObjectiveSense.Minimize, t)

    # Full investment
    M.constraint('budget', Expr.sum(w), Domain.equalsTo(1.))

    # Get at least the target return
    if target_return > 0.0:
        M.constraint('target', Expr.dot(mu, w), Domain.greaterThan(target_return))

    # Imposes a bound on the risk
    M.constraint('-1/q * (t + w`mu) > ||Gw||_2',
                 Expr.vstack(Expr.mul(-1/q, Expr.add(t, Expr.dot(w, mu))),
                             Expr.mul(G, w)), Domain.inQCone())
    if verbose:
        M.setLogHandler(sys.stdout)

    M.solve()

    return pd.Series(w.level(), index=names)


if __name__ == "__main__":

    # Load French's portfolio data from pandas-datareader
    data = web.DataReader("5_Industry_Portfolios", "famafrench", start=2000)
    r = data[1] / 100

    # Estimate the sample mean and covariance from the data
    mu = r.mean()
    H = r.cov()

    # Configuration
    target_return = 4e-04
    shortselling = False
    verbose = False

    # Estimate minimum variance portfolio
    w = min_risk(mu=mu, H=H, target_return=target_return, shortselling=shortselling, verbose=verbose)

    # Estimate minimum Value-at-Risk portfolio
    w = min_VaR(mu=mu, H=H, target_return=target_return, shortselling=shortselling, verbose=verbose)
