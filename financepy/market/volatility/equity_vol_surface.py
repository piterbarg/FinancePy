##############################################################################
# Copyright (C) 2018, 2019, 2020 Dominic O'Kane
##############################################################################

import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from numba import njit, float64, int64

from ...utils.error import FinError
from ...utils.date import Date
from ...utils.global_vars import gDaysInYear
from ...utils.global_types import OptionTypes
from ...models.option_implied_dbn import option_implied_dbn
from ...utils.helpers import check_argument_types, label_to_string
from ...market.curves.discount_curve import DiscountCurve

from ...models.volatility_fns import VolFuncTypes
from ...models.volatility_fns import vol_function_clark
from ...models.volatility_fns import vol_function_bloomberg
from ...models.volatility_fns import vol_function_svi
from ...models.volatility_fns import vol_function_ssvi
from ...models.sabr import vol_function_sabr
from ...models.sabr import vol_function_sabr_beta_one
from ...models.sabr import vol_function_sabr_beta_half

from ...utils.math import norminvcdf

from ...models.black_scholes_analytic import bs_delta

from ...utils.distribution import FinDistribution

from ...utils.solver_1d import newton_secant
from ...utils.solver_nm import nelder_mead
from ...utils.global_types import FinSolverTypes

###############################################################################
# ISSUES
###############################################################################

###############################################################################
# TODO: Speed up search for strike by providing derivative function to go with
#       delta fit.
###############################################################################


###############################################################################
# Do not cache this function - WRONG. IT WORKS BUT WHY WHEN IN FX IT FAILS ??
###############################################################################

@njit(fastmath=True, cache=True)
def _obj(params, *args):
    """ Return a value that is minimised when the ATM, MS and RR vols have
    been best fitted using the parametric volatility curve represented by
    params and specified by the vol_type_value. We fit at one time slice only.
    """

    s = args[0]
    t = args[1]
    r = args[2]
    q = args[3]
    strikes = args[4]
    index = args[5]
    volatility_grid = args[6]
    vol_type_value = args[7]

    f = s * np.exp((r-q)*t)

    num_strikes = len(volatility_grid[0])

    tot = 0.0

    for i in range(0, num_strikes):
        fitted_vol = vol_function(vol_type_value, params, f, strikes[i], t)
        mkt_vol = volatility_grid[index][i]
        diff = fitted_vol - mkt_vol
        tot += diff**2

    return tot

###############################################################################
# Do not cache this function as it leads to complaints
###############################################################################


def _solve_to_horizon(s, t, r, q,
                      strikes,
                      time_index,
                      volatility_grid,
                      vol_type_value,
                      x_inits,
                      finSolverType):

    ###########################################################################
    # Determine parameters of vol surface using minimisation
    ###########################################################################

    tol = 1e-6

    args = (s, t, r, q, strikes, time_index, volatility_grid, vol_type_value)

    # Nelder-Mead (both SciPy and Numba) is quicker, but occasionally fails
    # to converge, so for those cases try again with CG
    # Numba version is quicker, but can be slightly away from CG output
    try:
        if finSolverType == FinSolverTypes.NELDER_MEAD_NUMBA:
            xopt = nelder_mead(_obj, np.array(x_inits),
                               bounds=np.array([[], []]).T,
                               args=args, tol_f=tol,
                               tol_x=tol, max_iter=1000)
        elif finSolverType == FinSolverTypes.NELDER_MEAD:
            opt = minimize(_obj, x_inits, args, method="Nelder-Mead", tol=tol)
            xopt = opt.x
        elif finSolverType == FinSolverTypes.CONJUGATE_GRADIENT:
            opt = minimize(_obj, x_inits, args, method="CG", tol=tol)
            xopt = opt.x
    except Exception:
        # If convergence fails try again with CG if necessary
        if finSolverType != FinSolverTypes.CONJUGATE_GRADIENT:
            print('Failed to converge, will try CG')
            opt = minimize(_obj, x_inits, args, method="CG", tol=tol)
            xopt = opt.x

    params = np.array(xopt)
    return params

###############################################################################


@njit(float64(int64, float64[:], float64, float64, float64),
      cache=True, fastmath=True)
def vol_function(vol_function_type_value, params, f, k, t):
    """ Return the volatility for a strike using a given polynomial
    interpolation following Section 3.9 of Iain Clark book. """

    if vol_function_type_value == VolFuncTypes.CLARK.value:
        vol = vol_function_clark(params, f, k, t)
        return vol
    elif vol_function_type_value == VolFuncTypes.SABR_BETA_ONE.value:
        vol = vol_function_sabr_beta_one(params, f, k, t)
        return vol
    elif vol_function_type_value == VolFuncTypes.SABR_BETA_HALF.value:
        vol = vol_function_sabr_beta_half(params, f, k, t)
        return vol
    elif vol_function_type_value == VolFuncTypes.BBG.value:
        vol = vol_function_bloomberg(params, f, k, t)
        return vol
    elif vol_function_type_value == VolFuncTypes.SABR.value:
        vol = vol_function_sabr(params, f, k, t)
        return vol
    elif vol_function_type_value == VolFuncTypes.CLARK5.value:
        vol = vol_function_clark(params, f, k, t)
        return vol
    elif vol_function_type_value == VolFuncTypes.SVI.value:
        vol = vol_function_svi(params, f, k, t)
        return vol
    elif vol_function_type_value == VolFuncTypes.SSVI.value:
        vol = vol_function_ssvi(params, f, k, t)
        return vol
    else:
        return 0.0

###############################################################################


@njit(cache=True, fastmath=True)
def _delta_fit(k, *args):
    """ This is the objective function used in the determination of the
    option implied strike which is computed in the class below. I map it into
    inverse normcdf space to avoid the flat slope of this function at low vol
    and high K. It speeds up the code as it allows initial values close to
    the solution to be used. """

    vol_type_value = args[0]
    s = args[1]
    t = args[2]
    r = args[3]
    q = args[4]
    option_type_value = args[5]
    inverse_delta_target = args[6]
    params = args[7]

    f = s * np.exp((r-q)*t)
    v = vol_function(vol_type_value, params, f, k, t)
    delta_out = bs_delta(s, t, k, r, q, v, option_type_value)
    inverse_delta_out = norminvcdf(np.abs(delta_out))
    inv_obj_fn = inverse_delta_target - inverse_delta_out

    return inv_obj_fn

###############################################################################
# Unable to cache this function due to dynamic globals warning. Revisit.
###############################################################################


#@njit(float64(float64, float64, float64, float64, int64, int64, float64,
#              float64, float64[:]), fastmath=True)
def _solver_for_smile_strike(s, t, r, q,
                             option_type_value,
                             volatilityTypeValue,
                             delta_target,
                             initial_guess,
                             parameters):
    """ Solve for the strike that sets the delta of the option equal to the
    target value of delta allowing the volatility to be a function of the
    strike. """

    inverse_delta_target = norminvcdf(np.abs(delta_target))

    argtuple = (volatilityTypeValue, s, t, r, q,
                option_type_value,
                inverse_delta_target,
                parameters)

    K = newton_secant(_delta_fit, x0=initial_guess, args=argtuple,
                      tol=1e-8, maxiter=50)

    return K

###############################################################################
# Unable to cache function and if I remove njit it complains about pickle
###############################################################################


class EquityVolSurface:
    """ Class to perform a calibration of a chosen parametrised surface to the
    prices of equity options at different strikes and expiry tenors. There is0
    a choice of volatility function from cubic in delta to full SABR and SSVI.
    Check out VolFuncTypes. Visualising the volatility curve is useful.
    Also, there is no guarantee that the implied pdf will be positive."""

    def __init__(self,
                 value_dt: Date,
                 stock_price: float,
                 discount_curve: DiscountCurve,
                 dividend_curve: DiscountCurve,
                 expiry_dts: (list),
                 strikes: (list, np.ndarray),
                 volatility_grid: (list, np.ndarray),
                 volatility_function_type: VolFuncTypes = VolFuncTypes.CLARK,
                 finSolverType: FinSolverTypes = FinSolverTypes.NELDER_MEAD):
        """ Create the EquitySurface object by passing in market vol data
        for a list of strikes and expiry dates. """

        check_argument_types(self.__init__, locals())

        self._value_dt = value_dt
        self._stock_price = stock_price

        self._discount_curve = discount_curve
        self._dividend_curve = dividend_curve

        nExpiryDates = len(expiry_dts)
        nStrikes = len(strikes)
        n = len(volatility_grid)
        m = len(volatility_grid[0])

        if n != nExpiryDates:
            raise FinError("1st dimension of vol grid is not nExpiryDates")

        if m != nStrikes:
            raise FinError("2nd dimension of the vol matrix is not nStrikes")

        self._strikes = strikes
        self._num_strikes = len(strikes)

        self._expiry_dts = expiry_dts
        self._numExpiryDates = len(expiry_dts)

        self._volatility_grid = volatility_grid
        self._volatility_function_type = volatility_function_type

        self._build_vol_surface(finSolverType=finSolverType)

###############################################################################

    def vol_from_strike_dt(self, K, expiry_dt):
        """ Interpolates the Black-Scholes volatility from the volatility
        surface given call option strike and expiry date. Linear interpolation
        is done in variance space. The smile strikes at bracketed dates are
        determined by determining the strike that reproduces the provided delta
        value. This uses the calibration delta convention, but it can be
        overriden by a provided delta convention. The resulting volatilities
        are then determined for each bracketing expiry time and linear
        interpolation is done in variance space and then converted back to a
        lognormal volatility."""

        t_exp = (expiry_dt - self._value_dt) / gDaysInYear

        vol_type_value = self._volatility_function_type.value

        index0 = 0  # lower index in bracket
        index1 = 0  # upper index in bracket

        num_curves = self._numExpiryDates

        if num_curves == 1:

            index0 = 0
            index1 = 0

        # If the time is below first time then assume a flat vol
        elif t_exp <= self._t_exp[0]:

            index0 = 0
            index1 = 0

        # If the time is beyond the last time then extrapolate with a flat vol
        elif t_exp >= self._t_exp[-1]:

            index0 = len(self._t_exp) - 1
            index1 = len(self._t_exp) - 1

        else:  # Otherwise we look for bracketing times and interpolate

            for i in range(1, num_curves):

                if t_exp <= self._t_exp[i] and t_exp > self._t_exp[i-1]:
                    index0 = i-1
                    index1 = i
                    break

        fwd0 = self._F0T[index0]
        fwd1 = self._F0T[index1]

        t0 = self._t_exp[index0]
        t1 = self._t_exp[index1]

        vol0 = vol_function(vol_type_value, self._parameters[index0],
                            fwd0, K, t0)

        if index1 != index0:

            vol1 = vol_function(vol_type_value, self._parameters[index1],
                                fwd1, K, t1)

        else:

            vol1 = vol0

        # In the expiry time dimension, both volatilities are interpolated
        # at the same strikes but different deltas.
        vart0 = vol0*vol0*t0
        vart1 = vol1*vol1*t1

        if np.abs(t1-t0) > 1e-6:
            vart = ((t_exp-t0) * vart1 + (t1-t_exp) * vart0) / (t1 - t0)

            if vart < 0.0:
                raise FinError("Negative variance.")

            volt = np.sqrt(vart/t_exp)

        else:
            volt = vol1

        return volt

###############################################################################

    # def delta_to_strike(self, call_delta, expiry_dt, delta_method):
    #     """ Interpolates the strike at a delta and expiry date. Linear
    #     interpolation is used in strike."""

    #     t_exp = (expiry_dt - self._value_dt) / gDaysInYear

    #     vol_type_value = self._volatility_function_type.value

    #     s = self._spot_fx_rate

    #     if delta_method is None:
    #         delta_method_value = self._delta_method.value
    #     else:
    #         delta_method_value = delta_method.value

    #     index0 = 0 # lower index in bracket
    #     index1 = 0 # upper index in bracket

    #     num_curves = self._num_vol_curves

    #     # If there is only one time horizon then assume flat vol to this time
    #     if num_curves == 1:

    #         index0 = 0
    #         index1 = 0

    #     # If the time is below first time then assume a flat vol
    #     elif t_exp <= self._t_exp[0]:

    #         index0 = 0
    #         index1 = 0

    #     # If the time is beyond the last time then extrapolate with a flat vol
    #     elif t_exp > self._t_exp[-1]:

    #         index0 = len(self._t_exp) - 1
    #         index1 = len(self._t_exp) - 1

    #     else: # Otherwise we look for bracketing times and interpolate

    #         for i in range(1, num_curves):

    #             if t_exp <= self._t_exp[i] and t_exp > self._t_exp[i-1]:
    #                 index0 = i-1
    #                 index1 = i
    #                 break

    #     #######################################################################

    #     t0 = self._t_exp[index0]
    #     t1 = self._t_exp[index1]

    #     initial_guess = self._K_ATM[index0]

    #     K0 = _solver_for_smile_strike(s, t_exp, self._rd[index0], self._rf[index0],
    #                               OptionTypes.EUROPEAN_CALL.value,
    #                               vol_type_value, call_delta,
    #                               delta_method_value,
    #                               initial_guess,
    #                               self._parameters[index0],
    #                               self._strikes[index0],
    #                               self._gaps[index0])

    #     if index1 != index0:

    #         K1 = _solver_for_smile_strike(s, t_exp,
    #                                   self._rd[index1],
    #                                   self._rf[index1],
    #                                   OptionTypes.EUROPEAN_CALL.value,
    #                                   vol_type_value, call_delta,
    #                                   delta_method_value,
    #                                   initial_guess,
    #                                   self._parameters[index1],
    #                                   self._strikes[index1],
    #                                   self._gaps[index1])
    #     else:

    #         K1 = K0

    #     # In the expiry time dimension, both volatilities are interpolated
    #     # at the same strikes but different deltas.

    #     if np.abs(t1-t0) > 1e-6:

    #         K = ((t_exp-t0) * K1 + (t1-t_exp) * K1) / (t1 - t0)

    #     else:

    #         K = K1

    #     return K

###############################################################################

    def vol_from_delta_dt(self, call_delta, expiry_dt, delta_method=None):
        """ Interpolates the Black-Scholes volatility from the volatility
        surface given a call option delta and expiry date. Linear interpolation
        is done in variance space. The smile strikes at bracketed dates are
        determined by determining the strike that reproduces the provided delta
        value. This uses the calibration delta convention, but it can be
        overriden by a provided delta convention. The resulting volatilities
        are then determined for each bracketing expiry time and linear
        interpolation is done in variance space and then converted back to a
        lognormal volatility."""

        t_exp = (expiry_dt - self._value_dt) / gDaysInYear

        vol_type_value = self._volatility_function_type.value

        s = self._stock_price

        index0 = 0  # lower index in bracket
        index1 = 0  # upper index in bracket

        num_curves = self._numExpiryDates

        # If there is only one time horizon then assume flat vol to this time
        if num_curves == 1:

            index0 = 0
            index1 = 0

        # If the time is below first time then assume a flat vol
        elif t_exp <= self._t_exp[0]:

            index0 = 0
            index1 = 0

        # If the time is beyond the last time then extrapolate with a flat vol
        elif t_exp > self._t_exp[-1]:

            index0 = len(self._t_exp) - 1
            index1 = len(self._t_exp) - 1

        else:  # Otherwise we look for bracketing times and interpolate

            for i in range(1, num_curves):

                if t_exp <= self._t_exp[i] and t_exp > self._t_exp[i-1]:
                    index0 = i-1
                    index1 = i
                    break

        fwd0 = self._F0T[index0]
        fwd1 = self._F0T[index1]

        t0 = self._t_exp[index0]
        t1 = self._t_exp[index1]

        initial_guess = self._stock_price

        K0 = _solver_for_smile_strike(s,
                                      t_exp,
                                      self._r[index0],
                                      self._q[index0],
                                      OptionTypes.EUROPEAN_CALL.value,
                                      vol_type_value, call_delta,
                                      initial_guess,
                                      self._parameters[index0])

        vol0 = vol_function(vol_type_value, self._parameters[index0],
                            fwd0, K0, t0)

        if index1 != index0:

            K1 = _solver_for_smile_strike(s, t_exp,
                                          self._r[index1],
                                          self._q[index1],
                                          OptionTypes.EUROPEAN_CALL.value,
                                          vol_type_value, call_delta,
                                          initial_guess,
                                          self._parameters[index1])

            vol1 = vol_function(vol_type_value, self._parameters[index1],
                                fwd1, K1, t1)
        else:
            vol1 = vol0

        # In the expiry time dimension, both volatilities are interpolated
        # at the same strikes but different deltas.
        vart0 = vol0*vol0*t0
        vart1 = vol1*vol1*t1

        if np.abs(t1-t0) > 1e-6:

            vart = ((t_exp-t0) * vart1 + (t1-t_exp) * vart0) / (t1 - t0)
            kt = ((t_exp-t0) * K1 + (t1-t_exp) * K0) / (t1 - t0)

            if vart < 0.0:
                raise FinError(
                    "Failed interpolation due to negative variance.")

            volt = np.sqrt(vart/t_exp)

        else:

            volt = vol0
            kt = K0

        return volt, kt

###############################################################################

    def _build_vol_surface(self, finSolverType=FinSolverTypes.NELDER_MEAD):
        """ Main function to construct the vol surface. """

        s = self._stock_price

        numExpiryDates = self._numExpiryDates

        if self._volatility_function_type == VolFuncTypes.CLARK:
            num_parameters = 3
            self._parameters = np.zeros([numExpiryDates, num_parameters])
        elif self._volatility_function_type == VolFuncTypes.SABR_BETA_ONE:
            num_parameters = 3
            self._parameters = np.zeros([numExpiryDates, num_parameters])
        elif self._volatility_function_type == VolFuncTypes.SABR_BETA_HALF:
            num_parameters = 3
            self._parameters = np.zeros([numExpiryDates, num_parameters])
        elif self._volatility_function_type == VolFuncTypes.BBG:
            num_parameters = 3
            self._parameters = np.zeros([numExpiryDates, num_parameters])
        elif self._volatility_function_type == VolFuncTypes.SABR:
            num_parameters = 4
            self._parameters = np.zeros([numExpiryDates, num_parameters])
        elif self._volatility_function_type == VolFuncTypes.CLARK5:
            num_parameters = 5
            self._parameters = np.zeros([numExpiryDates, num_parameters])
        elif self._volatility_function_type == VolFuncTypes.SVI:
            num_parameters = 5
            self._parameters = np.zeros([numExpiryDates, num_parameters])
        elif self._volatility_function_type == VolFuncTypes.SSVI:
            num_parameters = 5
            self._parameters = np.zeros([numExpiryDates, num_parameters])
            self._parameters[:, 0] = 0.2  # sigma
            self._parameters[:, 1] = 0.8  # gamma
            self._parameters[:, 2] = -0.7  # rho
            self._parameters[:, 3] = 0.3
            self._parameters[:, 4] = 0.048
        else:
            print(self._volatilityFunctionType)
            raise FinError("Unknown Model Type")

        self._t_exp = np.zeros(numExpiryDates)

        self._F0T = np.zeros(numExpiryDates)
        self._r = np.zeros(numExpiryDates)
        self._q = np.zeros(numExpiryDates)

        #######################################################################
        # TODO: ADD SPOT DAYS
        #######################################################################

        spot_dt = self._value_dt

        for i in range(0, numExpiryDates):

            expiry_dt = self._expiry_dts[i]
            t_exp = (expiry_dt - spot_dt) / gDaysInYear

            dis_df = self._discount_curve._df(t_exp)
            div_df = self._dividend_curve._df(t_exp)
            f = s * div_df / dis_df

            self._t_exp[i] = t_exp
            self._r[i] = -np.log(dis_df) / t_exp
            self._q[i] = -np.log(div_df) / t_exp
            self._F0T[i] = f

        #######################################################################
        # THE ACTUAL COMPUTATION LOOP STARTS HERE
        #######################################################################

        vol_type_value = self._volatility_function_type.value

        x_inits = []
        x_init = np.zeros(num_parameters)
        x_inits.append(x_init)

        for i in range(0, numExpiryDates):

            t = self._t_exp[i]
            r = self._r[i]
            q = self._q[i]

            res = _solve_to_horizon(s, t, r, q,
                                    self._strikes,
                                    i,
                                    self._volatility_grid,
                                    vol_type_value,
                                    x_inits[i],
                                    finSolverType)

            self._parameters[i, :] = res

            x_init = res
            x_inits.append(x_init)

###############################################################################

    def check_calibration(self, verbose: bool, tol: float = 1e-6):
        """ Compare calibrated vol surface with market and output a report
        which sets out the quality of fit to the ATM and 10 and 25 delta market
        strangles and risk reversals. """

        if verbose:

            print("==========================================================")
            print("VALUE DATE:", self._value_dt)
            print("STOCK PRICE:", self._stock_price)
            print("==========================================================")

        for i in range(0, self._numExpiryDates):

            expiry_dt = self._expiry_dts[i]
            print("==========================================================")

            for j in range(0, self._num_strikes):

                strike = self._strikes[j]

                fitted_vol = self.vol_from_strike_dt(strike,
                                                       expiry_dt)

                mkt_vol = self._volatility_grid[i][j]

                diff = fitted_vol - mkt_vol

                print("%s %12.3f %7.4f %7.4f %7.5f" %
                      (expiry_dt, strike,
                       fitted_vol*100.0, mkt_vol*100, diff*100))

        print("==========================================================")

###############################################################################

    def implied_dbns(self, lowS, highS, numIntervals):
        """ Calculate the pdf for each tenor horizon. Returns a list of
        FinDistribution objects, one for each tenor horizon. """

        dbns = []

        for iTenor in range(0, self._numExpiryDates):

            f = self._F_0T[iTenor]
            t = self._t_exp[iTenor]

            dS = (highS - lowS) / numIntervals

            disDF = self._discount_curve._df(t)
            div_df = self._dividend_curve._df(t)

            r = -np.log(disDF) / t
            q = -np.log(div_df) / t

            Ks = []
            vols = []

            for iK in range(0, numIntervals):

                k = lowS + iK*dS

                vol = vol_function(self._volatility_function_type.value,
                                   self._parameters[iTenor],
                                   f, k, t)

                Ks.append(k)
                vols.append(vol)

            Ks = np.array(Ks)
            vols = np.array(vols)

            density = option_implied_dbn(self._stock_price, t, r, q, Ks, vols)

            dbn = FinDistribution(Ks, density)
            dbns.append(dbn)

        return dbns

###############################################################################

    def plot_vol_curves(self):
        """ Generates a plot of each of the vol discount implied by the market
        and fitted. """

        lowK = self._strikes[0] * 0.9
        highK = self._strikes[-1] * 1.1

        for tenor_index in range(0, self._numExpiryDates):

            expiry_dt = self._expiry_dts[tenor_index]
            plt.figure()

            ks = []
            fitted_vols = []

            numIntervals = 30
            K = lowK
            dK = (highK - lowK)/numIntervals

            for i in range(0, numIntervals):

                ks.append(K)
                fitted_vol = self.vol_from_strike_dt(K, expiry_dt) * 100.
                fitted_vols.append(fitted_vol)
                K = K + dK

            label_str = "FITTED AT " + str(self._expiry_dts[tenor_index])
            plt.plot(ks, fitted_vols, label=label_str)

            label_str = "MARKET AT " + str(self._expiry_dts[tenor_index])
            mkt_vols = self._volatility_grid[tenor_index] * 100.0
            plt.plot(self._strikes, mkt_vols, 'o', label=label_str)

            plt.xlabel("Strike")
            plt.ylabel("Volatility")

            title = str(self._volatility_function_type)
            plt.title(title)
            plt.legend()

###############################################################################

    def __repr__(self):
        s = label_to_string("OBJECT TYPE", type(self).__name__)
        s += label_to_string("VALUE DATE", self._value_dt)
        s += label_to_string("STOCK PRICE", self._stock_price)
        s += label_to_string("VOL FUNCTION", self._volatility_function_type)

        for i in range(0, self._numExpiryDates):
            s += label_to_string("EXPIRY DATE", self._expiry_dts[i])

        for i in range(0, self._num_strikes):
            s += label_to_string("STRIKE", self._strikes[i])

        s += label_to_string("EQUITY VOL GRID", self._volatility_grid)

        return s

###############################################################################

    def _print(self):
        """ Print a list of the unadjusted coupon payment dates used in
        analytic calculations for the bond. """
        print(self)

###############################################################################
