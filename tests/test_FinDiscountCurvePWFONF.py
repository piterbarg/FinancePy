###############################################################################
# Copyright (C) 2018, 2019, 2020 Dominic O'Kane
###############################################################################

from financepy.market.curves.discount_curve_pwf_onf import DiscountCurvePWFONF
from financepy.market.curves.interpolator import InterpTypes
from financepy.utils.date import Date
import numpy as np


def test_FinDiscountCurvePCFONF_01():

    start_date = Date(1, 1, 2015)
    knot_dates = [Date(1, 1, 2015), Date(1, 6, 2016), Date(1, 12, 2017), Date(1, 4, 2018), Date(1, 8, 2019)]
    ondfwd_rates = [0, 0.02, 0.04, 0.06, 0.08]

    curve = DiscountCurvePWFONF(start_date,
                                knot_dates,
                                ondfwd_rates,)

    test_dates = [Date(1, 6, 2015), Date(1, 2, 2017), Date(1, 2, 2018), Date(1, 8, 2018), Date(1, 12, 2019)]
    expected_onfwd = [0.02, 0.04, 0.06, 0.08, 0.08]
    actual_onfwd = curve.fwd(test_dates)

    one_bp = 1e-4
    for d, e, a in zip(test_dates, expected_onfwd, actual_onfwd):
        assert abs(e-a) < one_bp/100, f'Mismatch for date {d}, expected = {e}, actual = {a}'


def test_FinDiscountCurvePCFONF_02():

    start_date = Date(1, 1, 2015)
    knot_dates = [Date(1, 6, 2017), Date(1, 6, 2018), Date(2, 6, 2018)]
    ondfwd_rates = [0, 0.01, 0.0]

    curve = DiscountCurvePWFONF(start_date,
                                knot_dates,
                                ondfwd_rates,)

    test_dates = [Date(1, 6, 2015), Date(1, 12, 2017), Date(1, 8, 2018), ]
    expected_onfwd = [0.0, 0.01, 0.0]
    actual_onfwd = curve.fwd(test_dates)

    one_bp = 1e-4
    for d, e, a in zip(test_dates, expected_onfwd, actual_onfwd):
        assert abs(e-a) < one_bp/100, f'Mismatch for date {d}, expected = {e}, actual = {a}'
