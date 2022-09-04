import pandas as pd
from typing import List

from ...utils.date import Date
from ...utils.global_vars import gBasisPoint
from ...market.curves.discount_curve import DiscountCurve
from ...market.curves.discount_curve_pwf_onf import DiscountCurvePWFONF
from ...market.curves.composite_discount_curve import CompositeDiscountCurve

from ...products.rates.ibor_fra import IborFRA
from ...products.rates.ibor_swap import IborSwap
from ...products.rates.ibor_deposit import IborDeposit
from ...products.rates.ibor_single_curve import IborSingleCurve
from ...products.rates.ibor_single_curve_par_shocker import IborSingleCurveParShocker
from ...products.rates.ibor_benchmarks_report import benchmarks_report, ibor_benchmarks_report


def par_rate_risk_report(base_curve: IborSingleCurve,  trades: list, trade_labels: list = None, bump_size=1.0*gBasisPoint):
    """Calculate deltas (change in value to 1bp bump) of the trades to all benchmarks in the base curve. Supported trades are depos, fras, swaps.  
    trade_labels are used to identify trades in the output, if not provided simple ones are generated 

    Args:
        base_curve (IborSingleCurve): Base curve to be bumped
        trades (list): a list of trades to calculate deltas of
        trade_labels (list, optional): trade labels to identify trades in the output. Defaults to None in which case these are auto generted
        bump_size (float, optional): How big of a bump to apply to bechmarks. Output always expressed as change in value per 1 bp. Defaults to 1.0*gBasisPoint.

    Returns:
        (base_values, risk_report): 
            base_value is a dictionary with trade_labels as keys and base trade values as values
            risk_report is a dataframe with benchmarks as rows and a column of par deltas per trade, and a total for all trades
    """
    curve_shocker = IborSingleCurveParShocker(base_curve)
    benchmarks_report = curve_shocker.benchmarks_report()
    n_benchmarks = curve_shocker.n_benchmarks()
    n_trades = len(trades)

    risk_report = benchmarks_report[[
        'type', 'start_date', 'maturity_date']].copy()

    if trade_labels is None:
        trade_labels = [f'trade_{n:03d}' for n in range(n_trades)]

    base_values = {}
    for trade, trade_label in zip(trades, trade_labels):
        base_values[trade_label] = trade.value(
            base_curve._valuation_date, base_curve)

    for benchmark_idx in range(n_benchmarks):
        bumped_curve = curve_shocker.apply_bump_to_benchmark(
            benchmark_idx, bump_size)

        for trade_idx, trade in enumerate(trades):
            trade_label = trade_labels[trade_idx]
            base_value = base_values[trade_label]
            bumped_value = trade.value(
                bumped_curve._valuation_date, bumped_curve)
            par_delta = (bumped_value - base_value)/bump_size*gBasisPoint
            risk_report.loc[benchmark_idx, trade_label] = par_delta

    risk_report['total'] = risk_report[trade_labels].sum(axis=1)
    return base_values, risk_report


def forward_rate_risk_report(base_curve: DiscountCurve,
                             grid_last_date: Date,
                             grid_bucket_tenor: str,
                             trades: list,  trade_labels: list = None, bump_size=1.0*gBasisPoint,):
    """Generate forward rate deltas risk report, which is the sensitivity of trades to bucketed
    shocks of the instantaneous (ON) forward rates. Here shock_i is applied to the ON forward rates
    over [t_i, t_{i+1}] where {t_i} is a time grid from base_curve.valuation_date to grid_last_date with
    buckets of size grid_bucket_tenor

    Args:
        base_curve (DiscountCurve): base curve to apply bumps to
        grid_last_date: the last date for the grid that defines shocks
        grid_bucket_tenor (str): spacing of grid points as a tenor string such as '3M'
        trades (list): a list of trades to calculate deltas of
        trade_labels (list, optional): trade labels to identify trades in the output. Defaults to None in which case these are auto generted
        bump_size (float, optional): How big of a bump to apply to bechmarks. Output always expressed as change in value per 1 bp. Defaults to 1.0*gBasisPoint.

    Returns:
        (dict, Dataframe): (base_values, risk_report) 
            base_value is a dictionary with trade_labels as keys and base trade values as values
            risk_report is a dataframe with bump details for rows and a column of forward rate deltas per trade, and a total for all trades

    """
    grid = [base_curve._valuation_date]
    d = grid[0]

    # the loop is structured so that the grid_last_date is captured by the last bucket
    while d < grid_last_date:
        d = d.add_tenor(grid_bucket_tenor)
        grid.append(d)
    grid[-1] = grid_last_date

    return forward_rate_risk_report_custom_grid(base_curve, grid, trades, trade_labels, bump_size)


def forward_rate_risk_report_custom_grid(base_curve: DiscountCurve,
                                         grid: List[Date],
                                         trades: list,  trade_labels: list = None, bump_size=1.0*gBasisPoint,):
    """Generate forward rate deltas risk report, which is the sensitivity of trades to bucketed
    shocks of the instantaneous (ON) forward rates. Here shock_i is applied to the ON forward rates
    over [t_i, t_{i+1}] where {t_i} is the 'grid' argument

    Args:
        base_curve (DiscountCurve): base curve to apply bumps to
        grid: Date grid that defines ON forward rate bumps. Note that curve.valuation_date must be explictly included (if so desired)
        trades (list): a list of trades to calculate deltas of
        trade_labels (list, optional): trade labels to identify trades in the output. Defaults to None in which case these are auto generted
        bump_size (float, optional): How big of a bump to apply to bechmarks. Output always expressed as change in value per 1 bp. Defaults to 1.0*gBasisPoint.

    Returns:
        (dict, Dataframe): (base_values, risk_report) 
            base_value is a dictionary with trade_labels as keys and base trade values as values
            risk_report is a dataframe with bump details for rows and a column of forward rate deltas per trade, and a total for all trades

    """
    risk_report = pd.DataFrame(columns=['type', 'start_date', 'maturity_date'])
    risk_report['start_date'] = grid[:-1]
    risk_report['maturity_date'] = grid[1:]
    risk_report['type'] = 'ForwardRate'

    n_trades = len(trades)
    if trade_labels is None:
        trade_labels = [f'trade_{n:03d}' for n in range(n_trades)]

    base_values = {}
    for trade, trade_label in zip(trades, trade_labels):
        base_values[trade_label] = trade.value(
            base_curve._valuation_date, base_curve)

    for fwdrate_idx in range(len(risk_report)):
        fwd_rate_shock = DiscountCurvePWFONF.brick_wall_curve(
            base_curve._valuation_date, risk_report.loc[fwdrate_idx, 'start_date'], risk_report.loc[fwdrate_idx, 'maturity_date'], bump_size)
        bumped_curve = CompositeDiscountCurve([base_curve, fwd_rate_shock])

        for trade_idx, trade in enumerate(trades):
            trade_label = trade_labels[trade_idx]
            base_value = base_values[trade_label]
            bumped_value = trade.value(
                bumped_curve._valuation_date, bumped_curve)
            par_delta = (bumped_value - base_value)/bump_size*gBasisPoint
            risk_report.loc[fwdrate_idx, trade_label] = par_delta

    risk_report['total'] = risk_report[trade_labels].sum(axis=1)
    return base_values, risk_report
