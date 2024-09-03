##############################################################################
# Copyright (C) 2018, 2019, 2020 Dominic O'Kane
##############################################################################

import numpy as np
import scipy.optimize as optimize

from ...utils.date import Date
from ...utils.error import FinError
from ...utils.global_vars import g_days_in_year
from ...market.curves.interpolator import _uinterpolate, InterpTypes
from ...utils.helpers import input_time, table_to_string
from ...utils.day_count import DayCount
from ...utils.frequency import annual_frequency, FrequencyTypes
from ...utils.helpers import check_argument_types, _func_name
from ...utils.helpers import label_to_string


###############################################################################


def f(q, *args):
    """Function that returns zero when the survival probability that gives a
    zero value of the CDS has been determined."""

    curve = args[0]
    value_dt = args[1]
    cds = args[2]
    recovery_rate = args[3]

    num_points = len(curve._times)
    curve._values[num_points - 1] = q
    # This is important - we calibrate a curve that makes the clean PV of the
    # CDS equal to zero and so we select the second element of the value tuple
    obj_fn = cds.value(value_dt, curve, recovery_rate)["clean_pv"]
    return obj_fn


###############################################################################


class CDSCurve:
    """Generate a survival probability curve implied by the value of CDS
    contracts given a Ibor curve and an assumed recovery rate. The recovery
    rate corresponds to the seniority of the debt for these CDS. A scheme for
    the interpolation of the survival probabilities is also required."""

    def __init__(
        self,
        value_dt: Date,
        cds_contracts: list,
        libor_curve,
        recovery_rate,
        use_cache: bool = False,
        interp_method: InterpTypes = InterpTypes.FLAT_FWD_RATES,
    ):
        """Construct a credit curve from a sequence of maturity-ordered CDS
        contracts and a Ibor curve using the same recovery rate and the
        same interpolation method."""

        check_argument_types(getattr(self, _func_name(), None), locals())

        if value_dt != libor_curve.value_dt:
            raise FinError(
                "Curve does not have same valuation date as Issuer curve."
            )

        self.value_dt = value_dt
        self.cds_contracts = cds_contracts
        self.recovery_rate = recovery_rate
        self.libor_curve = libor_curve
        self.interp_method = interp_method
        self.built_ok = False

        self._times = []
        self._values = []

        if len(self.cds_contracts) > 0:
            self._build_curve()
        else:
            pass  # In some cases we allow None to be passed

        return

    ###########################################################################

    def times(self):
        return self._times

    ###########################################################################

    def values(self):
        return self._values

    ###########################################################################

    def _validate(self, cds_contracts):
        """Ensure that contracts are in increasing maturity."""

        if len(cds_contracts) == 0:
            raise FinError("No CDS contracts have been supplied.")

        maturity_dt = cds_contracts[0].maturity_dt

        for cds in cds_contracts[1:]:
            if cds.maturity_dt <= maturity_dt:
                raise FinError("CDS contracts not in increasing maturity.")

            maturity_dt = cds.maturity_dt

    ###########################################################################

    def survival_prob(self, dt):
        """Extract the survival probability to date dt. This function
        supports vectorisation."""

        if isinstance(dt, Date):
            t = (dt - self.value_dt) / g_days_in_year
        elif isinstance(dt, list):
            t = np.array(dt)
        else:
            t = dt

        if np.any(t < 0.0):
            raise FinError("Survival Date before curve anchor date")

        if isinstance(t, np.ndarray):
            n = len(t)
            qs = np.zeros(n)
            for i in range(0, n):
                qs[i] = _uinterpolate(
                    t[i], self._times, self._values, self.interp_method.value
                )
            return qs
        elif isinstance(t, float):
            q = _uinterpolate(
                t, self._times, self._values, self.interp_method.value
            )
            return q
        else:
            raise FinError("Unknown time type")

    ###########################################################################

    def df(self, dt):
        """Extract the discount factor from the underlying Ibor curve. This
        function supports vectorisation."""

        if isinstance(dt, Date):
            t = (dt - self.value_dt) / g_days_in_year
        elif isinstance(dt, list):
            t = np.array(dt)
        else:
            t = dt

        df = self.libor_curve._df(t)

        return df

    ###########################################################################

    def _build_curve(self):
        """Construct the CDS survival curve from a set of CDS contracts"""

        self._validate(self.cds_contracts)
        num_times = len(self.cds_contracts)

        # we size the vectors to include time zero
        self._times = np.array([0.0])
        self._values = np.array([1.0])

        for i in range(0, num_times):

            maturity_dt = self.cds_contracts[i].maturity_dt

            argtuple = (
                self,
                self.value_dt,
                self.cds_contracts[i],
                self.recovery_rate,
            )

            t_mat = (maturity_dt - self.value_dt) / g_days_in_year
            q = self._values[i]

            self._times = np.append(self._times, t_mat)
            self._values = np.append(self._values, q)

            optimize.newton(
                f,
                x0=q,
                fprime=None,
                args=argtuple,
                tol=1e-7,
                maxiter=50,
                fprime2=None,
            )

    ###########################################################################

    def fwd(self, dt):
        """Calculate the instantaneous forward rate at the forward date dt
        using the numerical derivative."""

        t = input_time(dt, self)
        epsilon = 1e-8
        df1 = self.df(t) * self.survival_prob(t)
        df2 = self.df(t + epsilon) * self.survival_prob(t + epsilon)
        fwd = np.log(df1 / df2) / dt
        return fwd

    ###########################################################################

    def fwd_rate(self, date1, date2, day_count_type):
        """Calculate the forward rate according between dates date1 and date2
        according to the specified day count convention."""

        if date1 < self.value_dt:
            raise FinError("Date1 before curve value date.")

        if date2 < date1:
            raise FinError("Date2 must not be before Date1")

        day_count = DayCount(day_count_type)
        year_frac = day_count.year_frac(date1, date2)[0]
        df1 = self.df(date1)
        df2 = self.df(date2)
        fwd = (df1 / df2 - 1.0) / year_frac
        return fwd

    ###########################################################################

    def zero_rate(self, dt, freq_type=FrequencyTypes.CONTINUOUS):
        """Calculate the zero rate to date dt in the chosen compounding
        frequency where -1 is continuous is the default."""

        t = input_time(dt, self)
        f = annual_frequency(freq_type)
        df = self.df(t)
        q = self.survival_prob(t)
        dfq = df * q

        if f == 0:  # Simple interest
            zero_rate = (1.0 / dfq - 1.0) / t
        if f == -1:  # Continuous
            zero_rate = -np.log(dfq) / t
        else:
            zero_rate = (dfq ** (-1.0 / t) - 1) * f
        return zero_rate

    ###########################################################################

    def __repr__(self):
        """Print out the details of the survival probability curve."""
        s = label_to_string("OBJECT TYPE", type(self).__name__)
        header = "TIME,SURVIVAL_PROBABILITY"
        value_table = [self._times, self._values]
        precision = "10.7f"
        s += table_to_string(header, value_table, precision)
        return s

    ###########################################################################

    def _print(self):
        """Simple print function for backward compatibility."""
        print(self)


###############################################################################
