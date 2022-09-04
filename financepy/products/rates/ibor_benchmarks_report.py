import pandas as pd

from ...utils.date import Date
from ...market.curves.discount_curve import DiscountCurve
from ...products.rates.ibor_single_curve import IborSingleCurve


def benchmarks_report(benchmarks,
                      valuation_date: Date,
                      discount_curve: DiscountCurve,
                      index_curve: DiscountCurve = None,
                      include_objects=False):
    '''
    Generate a DataFrame with one row per bechmark. A benchmark is any object that has a function
    valuation_details(...) that returns a dictionary of the right shape. Allowed benchmarks at the moment
    are depos, fras and swaps. Various useful information is reported. This is a bit slow
    so do not use in performance-critical spots
    '''

    # benchmarks = depos + fras + swaps
    df_bmi = None
    for benchmark in benchmarks:
        res = benchmark.valuation_details(valuation_date, discount_curve, index_curve)

        if df_bmi is None:
            df_bmi = pd.DataFrame.from_dict(res, orient='index').T
        else:
            df_bmi = df_bmi.append(res, ignore_index=True)

    if include_objects:
        df_bmi['benchmark_objects'] = benchmarks
    return df_bmi


def ibor_benchmarks_report(iborCurve: IborSingleCurve, include_objects=False):
    '''
    Generate a DataFrame with one row per bechmark used in constructing a given iborCurve.
    Various useful information is reported. This is a bit slow so do not use in performance-critical spots
    '''

    benchmarks = iborCurve._usedDeposits + iborCurve._usedFRAs + iborCurve._usedSwaps
    return benchmarks_report(benchmarks,
                             iborCurve._valuation_date, iborCurve, include_objects=include_objects)
