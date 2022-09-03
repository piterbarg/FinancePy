###############################################################################
# Copyright (C) 2018, 2019, 2020 Dominic O'Kane
###############################################################################


import numpy as np
from typing import List, Union

###############################################################################

from ...utils.date import Date
from ...utils.helpers import label_to_string
from ...utils.helpers import check_argument_types
from ...market.curves.discount_curve import DiscountCurve


class CompositeDiscountCurve(DiscountCurve):
    """ 
    A discount curve that is a sum (in rates) of 'children' discount curves
    """

###############################################################################

    def __init__(self, child_curves: List[DiscountCurve]):
        """ 
        Create a discount curve that is a sum (in rates) of other discount curves
        """

        check_argument_types(self.__init__, locals())

        assert child_curves, 'Empty list of child curves is not supported'

        self._children = child_curves

        self._valuation_date = self._children[0]._valuation_date
        assert all(c._valuation_date ==
                   self._valuation_date for c in self._children), 'Child curves must all have the same vlauation date'

###############################################################################

    def df(self,
           dates: Union[Date, list]):
        """ 
        Return discount factors given a single or vector of dates. 
        ParentRate = Sum of children rates => Parent DF = product of children dfs 
        """

        dfs = np.ones_like(np.asarray(dates), dtype=float)
        for c in self._children:
            dfs *= c.df(dates)

        return dfs

###############################################################################

    def __repr__(self):
        s = label_to_string("OBJECT TYPE", type(self).__name__)
        s += label_to_string("CHILDREN", (self._children))
        return s

###############################################################################

    def _print(self):
        """ Simple print function for backward compatibility. """
        print(self)

###############################################################################
