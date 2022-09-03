##############################################################################
# Copyright (C) 2018, 2019, 2020 Dominic O'Kane
##############################################################################

import numpy as np
from typing import Union
from scipy import interpolate

from ...utils.date import Date
from ...utils.error import FinError
from ...utils.global_vars import gSmall
from ...utils.math import test_monotonicity
from ...utils.frequency import FrequencyTypes
from ...utils.helpers import label_to_string
from ...utils.helpers import check_argument_types
from ...utils.day_count import DayCountTypes
from ...utils.helpers import times_from_dates
from ...market.curves.discount_curve import DiscountCurve


###############################################################################


class DiscountCurvePWFONF(DiscountCurve):
    """ Curve with piece-wise flat instantaneous (ON) fwd rates. Curve is made up of a series of sections with each having
    a flat instantaneous forward rate. The default compounding assumption is
    continuous. The class inherits methods from FinDiscountCurve. """

    def __init__(self,
                 valuation_date: Date,
                 knot_dates: list,
                 onfwd_rates: Union[list, np.ndarray]):
        """ 
        Creates a discount curve using a vector of times and ON fwd rates
        The fwd rate is right-continuous i.e. a given value in the input is applied to the left of that date until the previous date exclusive
        The last fwd rate is extrapolated into the future
        """

        check_argument_types(self.__init__, locals())

        self._valuation_date = valuation_date

        if len(knot_dates) != len(onfwd_rates):
            raise FinError("Dates and rates vectors must have same length")

        if len(knot_dates) == 0:
            raise FinError("Dates vector must have length > 0")

        self._knot_dates = knot_dates
        self._onfwd_rates = np.asarray(onfwd_rates)

        self._freq_type = FrequencyTypes.CONTINUOUS
        self._day_count_type = DayCountTypes.ACT_ACT_ISDA

        dc_times = times_from_dates(self._knot_dates,
                                    self._valuation_date,
                                    self._day_count_type)

        self._times = np.asarray(dc_times)

        # it is easier to deal in log(dfs), log(df[Ti]) = -\int_0^T_i f(u) du
        self._logdfs = -np.cumsum(np.diff(self._times, prepend=0.0) * self._onfwd_rates)
        self._logdfs_interp = interpolate.interp1d(
            np.concatenate(([0.0], self._times)), np.concatenate(([0.0], self._logdfs)), kind='linear', bounds_error=False, fill_value='extrapolate')

        if test_monotonicity(self._times) is False:
            raise FinError("Times are not sorted in increasing order")

    ###############################################################################

    def _zero_rate(self,
                   times: Union[float, np.ndarray, list]):
        """ 
        Piecewise flat instantaneous (ON) fwd rate is the same as linear logDfs
        """

        times = np.asarray(times, dtype=float)

        if np.any(times < 0.0):
            raise FinError("All times must be positive")

        times = np.maximum(times, gSmall)
        ldfs = self._logdfs_interp(times)
        zero_rates = -ldfs/times
        return zero_rates

    ###############################################################################

    def df(self, dates: Union[Date, list]):
        """ Return discount factors given a single or vector of dates. The
        discount factor depends on the rate and this in turn depends on its
        compounding frequency and it defaults to continuous compounding. It
        also depends on the day count convention. This was set in the
        construction of the curve to be ACT_ACT_ISDA. """

        # Get day count times to use with curve day count convention
        dc_times = times_from_dates(dates,
                                    self._valuation_date,
                                    self._day_count_type)

        zero_rates = self._zero_rate(dc_times)

        df = self._zero_to_df(self._valuation_date,
                              zero_rates,
                              dc_times,
                              self._freq_type,
                              self._day_count_type)

        return df

    ###############################################################################

    def __repr__(self):

        s = label_to_string("OBJECT TYPE", type(self).__name__)
        s += label_to_string("DATE", "ONWD RATE")
        for i in range(0, len(self._knot_dates)):
            s += label_to_string(self._knot_dates[i], self._onfwd_rates[i])
        s += label_to_string("FREQUENCY", (self._freq_type))
        return s

    ###############################################################################

    def _print(self):
        """ Simple print function for backward compatibility. """
        print(self)

###############################################################################
