import numpy as np

from helpers import *
from financepy.utils.date import Date
from financepy.utils.global_vars import gBasisPoint
from financepy.utils.global_types import SwapTypes
from financepy.utils.calendar import CalendarTypes
from financepy.utils.day_count import DayCountTypes
from financepy.utils.frequency import FrequencyTypes
from financepy.market.curves.interpolator import InterpTypes
from financepy.products.rates.ibor_deposit import IborDeposit
from financepy.products.rates.ibor_fra import IborFRA
from financepy.products.rates.ibor_swap import IborSwap
from financepy.products.rates.ibor_single_curve import IborSingleCurve
from financepy.products.rates.ibor_curve_risk_engine import par_rate_risk_report, forward_rate_risk_report


def test_par_rate_risk_report_cubic_zero():
    valuation_date = Date(6, 10, 2001)
    cal = CalendarTypes.UNITED_KINGDOM
    interp_type = InterpTypes.FINCUBIC_ZERO_RATES

    depoDCCType = DayCountTypes.ACT_360
    fraDCCType = DayCountTypes.ACT_360
    swapType = SwapTypes.PAY
    fixedDCCType = DayCountTypes.THIRTY_E_360_ISDA
    fixedFreqType = FrequencyTypes.SEMI_ANNUAL

    settlement_date, base_curve = _generate_base_curve(
        valuation_date, cal, interp_type, depoDCCType, fraDCCType, swapType, fixedDCCType, fixedFreqType)
    trades = _generate_trades(valuation_date, cal, swapType,
                              fixedDCCType, fixedFreqType, settlement_date, base_curve)

    # size of bump to apply. In all cases par risk is reported as change in value to 1 bp rate bump
    par_rate_bump = 1*gBasisPoint

    # run the report
    base_values, risk_report = par_rate_risk_report(
        base_curve, trades, bump_size=par_rate_bump)

    expected_totals = [0.00122854, -0.25323828, -0.24271177, -0.01423219,  0.31617136,
                       4.0262114, 2.03409619, -0.3957559]
    actual_totals = risk_report['total'].values

    # trade_labels = list(base_values.keys())
    # np.set_printoptions(suppress=True)
    # print(base_values)
    # print(risk_report['total'].values)
    # print(risk_report[trade_labels + ['total']].sum(axis=0))

    assert max(np.abs(actual_totals - expected_totals)) <= 1e-4


def test_par_rate_risk_report_flat_forward():
    valuation_date = Date(6, 10, 2022)
    base_curve = buildIborSingleCurve(valuation_date, '10Y')
    settlement_date = base_curve._usedSwaps[0]._effective_date
    cal = base_curve._usedSwaps[0]._fixed_leg._calendar_type
    fixed_day_count = base_curve._usedSwaps[0]._fixed_leg._day_count_type
    fixed_freq_type = base_curve._usedSwaps[0]._fixed_leg._freq_type

    trades = _generate_trades(valuation_date, cal, SwapTypes.PAY,
                              fixed_day_count, fixed_freq_type, settlement_date, base_curve)

    # size of bump to apply. In all cases par risk is reported as change in value to 1 bp rate bump
    par_rate_bump = 1*gBasisPoint

    # run the report
    base_values, risk_report = par_rate_risk_report(
        base_curve, trades, bump_size=par_rate_bump)

    expected_totals = [-0.08629015, -0.20597528, -0.08628776, -0.07793533, -0.05012633,
                       -0.00005542, -0.00005726, -0.00005542, -0.00005726, -0.00005726, -0.00008393, -0.00013093,
                       -0.0001267, 1.01065078, 1.49626527, 3.99996586, 0, 0, 0, 0, 0, 0
                       ]
    actual_totals = risk_report['total'].values

    # trade_labels = list(base_values.keys())
    # np.set_printoptions(suppress=True)
    # print(base_values)
    # print(risk_report['total'].values)
    # print(risk_report[trade_labels + ['total']].sum(axis=0))

    assert max(np.abs(actual_totals - expected_totals)) <= 1e-4


def test_forward_rate_risk_report():
    valuation_date = Date(6, 10, 2001)
    cal = CalendarTypes.UNITED_KINGDOM
    interp_type = InterpTypes.FLAT_FWD_RATES

    depoDCCType = DayCountTypes.ACT_360
    fraDCCType = DayCountTypes.ACT_360
    swapType = SwapTypes.PAY
    fixedDCCType = DayCountTypes.THIRTY_E_360_ISDA
    fixedFreqType = FrequencyTypes.SEMI_ANNUAL

    settlement_date, base_curve = _generate_base_curve(
        valuation_date, cal, interp_type, depoDCCType, fraDCCType, swapType, fixedDCCType, fixedFreqType)
    trades = _generate_trades(valuation_date, cal, swapType,
                              fixedDCCType, fixedFreqType, settlement_date, base_curve)

    # the grid on which we generate the risk report
    grid_bucket = '3M'
    grid_last_date = max(t._maturity_date for t in trades)

    # size of bump to apply. In all cases par risk is reported as change in value to 1 bp rate bump
    forward_rate_bump = 1*gBasisPoint

    # run the report
    base_values, risk_report = forward_rate_risk_report(
        base_curve, grid_last_date, grid_bucket, trades, bump_size=forward_rate_bump)

    expected_totals = [0.24374713, 0.24648603, 0.47965093, 0.49286362, 0.48197167, 0.47113499,
                       0.46583163, 0.47058739, 0.46015979, 0.45481044, 0.23886436, 0.22408547,
                       0.21923005, 0.21423566, 0.21186086, 0.21396794, 0.0069773]
    actual_totals = risk_report['total'].values

    # trade_labels = list(base_values.keys())
    # np.set_printoptions(suppress=True)
    # print(base_values)
    # print(risk_report)
    # print(risk_report['total'].values)
    # print(risk_report[trade_labels + ['total']].sum(axis=0))

    assert max(np.abs(actual_totals - expected_totals)) <= 1e-4


def _generate_trades(valuation_date, cal, swapType, fixedDCCType, fixedFreqType, settlement_date, base_curve):
    trade1 = IborSwap(settlement_date, "4Y", swapType, 4.20 /
                      100.0, fixedFreqType, fixedDCCType, calendar_type=cal, notional=10000)
    atm = trade1.swap_rate(valuation_date, base_curve)
    trade1.set_fixed_rate(atm)
    trade2 = IborSwap(settlement_date.add_tenor('6M'), "2Y", swapType,
                      4.20/100.0, fixedFreqType, fixedDCCType, calendar_type=cal, notional=10000)
    atm = trade2.swap_rate(valuation_date, base_curve)
    trade2.set_fixed_rate(atm)
    trades = [trade1, trade2]
    return trades


def _generate_base_curve(valuation_date, cal, interp_type, depoDCCType, fraDCCType, swapType, fixedDCCType, fixedFreqType):
    depos = []
    spot_days = 2
    settlement_date = valuation_date.add_weekdays(spot_days)
    depo = IborDeposit(settlement_date, "3M", 4.2/100.0,
                       depoDCCType, calendar_type=cal)
    depos.append(depo)

    fras = []
    fra = IborFRA(settlement_date.add_tenor("3M"), "3M",
                  4.20/100.0, fraDCCType, calendar_type=cal)
    fras.append(fra)

    swaps = []
    swap = IborSwap(settlement_date, "1Y", swapType, 4.20/100.0,
                    fixedFreqType, fixedDCCType, calendar_type=cal)
    swaps.append(swap)
    swap = IborSwap(settlement_date, "2Y", swapType, 4.30/100.0,
                    fixedFreqType, fixedDCCType, calendar_type=cal)
    swaps.append(swap)
    swap = IborSwap(settlement_date, "3Y", swapType, 4.70/100.0,
                    fixedFreqType, fixedDCCType, calendar_type=cal)
    swaps.append(swap)
    swap = IborSwap(settlement_date, "5Y", swapType, 4.70/100.0,
                    fixedFreqType, fixedDCCType, calendar_type=cal)
    swaps.append(swap)
    swap = IborSwap(settlement_date, "7Y", swapType, 4.70/100.0,
                    fixedFreqType, fixedDCCType, calendar_type=cal)
    swaps.append(swap)

    base_curve = IborSingleCurve(
        valuation_date, depos, fras, swaps, interp_type, )

    return settlement_date, base_curve


# if __name__ == '__main__':
#    test_par_rate_risk_report_cubic_zero()
