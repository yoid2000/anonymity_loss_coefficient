import numpy as np
from typing import Optional


class AnonymityLossCoefficient:
    '''
    AnonymityLossCoefficient is used to generate an anonymity loss coefficient (ALC).
    The max ALC is 1.0, which corresponds to complete anonymity loss, and is equivalent
    to publishing the original data. An ALC of 0.0 means that the there is
    no anonymity loss. What this means in practice is that the quality of
    attribute inferences about individuals in the synthetic dataset is
    statistically equivalent to the quality of attribute inferences made
    from a non-anonymized dataset about individuals that are not in that dataset.
    The ALC can be negative.  An ALC of 0.5 can be regarded conservatively as a
    safe amount of loss. In other words, the loss is little enough that it 
    eliminates attacker incentive.
    '''
    def __init__(self) -> None:
        # _prc_abs_weight is the weight given to the absolute PRC difference
        self._prc_abs_weight: float = 0.0
        # _recall_adjust_min_intercept is the recall value below which precision
        # has no effect on the PRC
        self._recall_adjust_min_intercept: float = 1/10000
        # Higher _recall_adjust_strength leads to lower recall adjustment
        self._recall_adjust_strength: float = 3.0

    def set_param(self, param: str, value: float) -> None:
        if param == 'prc_abs_weight':
            self._prc_abs_weight = value
        if param == 'recall_adjust_min_intercept':
            self._recall_adjust_min_intercept = value
        if param == 'recall_adjust_strength':
            self._recall_adjust_strength = value

    def get_param(self, param: str) -> Optional[float]:
        if param == 'prc_abs_weight':
            return self._prc_abs_weight
        if param == 'recall_adjust_min_intercept':
            return self._recall_adjust_min_intercept
        if param == 'recall_adjust_strength':
            return self._recall_adjust_strength
        return None

    def _recall_adjust(self, recall: float) -> float:
        adjust = (np.log10(recall) / np.log10(self._recall_adjust_min_intercept)) ** self._recall_adjust_strength
        return 1 - adjust

    def _prc_improve_absolute(self, prc_base: float, prc_attack: float) -> float:
        return prc_attack - prc_base

    def _prc_improve_relative(self, prc_base: float, prc_attack: float) -> float:
        if prc_base >= 1.0:
            prc_base = 0.99999999
        if prc_attack >= 1.0:
            prc_attack = 0.99999999
        return (prc_attack - prc_base) / (1.0 - prc_base)

    def _prc_improve(self, prc_base: float, prc_attack: float) -> float:
        prc_rel = self._prc_improve_relative(prc_base, prc_attack)
        prc_abs = self._prc_improve_absolute(prc_base, prc_attack)
        prc_improve = (self._prc_abs_weight * prc_abs) + ((1 - self._prc_abs_weight) * prc_rel)
        return prc_improve

    def prc(self, prec: float, recall: float) -> float:
        ''' Generates the precision-recall-coefficient, PRC.
            prev is the precision of the attack, and recall is the recall.
        '''
        # We do this adjusting because of floating point inaccuraccy
        prec = min(prec, 1.0)
        prec = max(prec, 0.0)
        recall = min(recall, 1.0)
        recall = max(recall, 0.0)
        if recall <= self._recall_adjust_min_intercept:
            return recall
        Rmin = self._recall_adjust_min_intercept
        alpha = self._recall_adjust_strength
        R = recall
        P = prec
        return (1 - ((np.log10(R) / np.log10(Rmin)) ** alpha)) * P

    def alc(self,
            p_base: Optional[float] = None,
            r_base: Optional[float] = None,
            p_attack: Optional[float] = None,
            r_attack: Optional[float] = None,
            prc_base: Optional[float] = None,
            prc_attack: Optional[float] = None
            ) -> Optional[float]:
        ''' alc can be called with either p_x and c_x, or prc_x
        '''
        if prc_base is None and p_base is not None and r_base is not None:
            # Adjust the precision based on the recall to make the
            # precision-recall-coefficient prc
            prc_base = self.prc(p_base, r_base)
        if prc_attack is None and p_attack is not None and r_attack is not None:
            prc_attack = self.prc(p_attack, r_attack)
        if prc_base is not None and prc_attack is not None:
            return self._prc_improve(prc_base, prc_attack)
        return None

    # The following aren't necessary for the AnonymityLossCoefficient, but are just
    # for testing
    def prec_from_prc_recall(self, prc: float, recall: float) -> float:
        ''' Given a PRC and recall, return the precision.
        '''
        Rmin = self._recall_adjust_min_intercept
        alpha = self._recall_adjust_strength
        R = recall
        PRC = prc
        return PRC / (1 - (np.log10(R) / np.log10(Rmin)) ** alpha)

    def recall_from_prc_prec(self, prc: float, prec: float) -> float:
        ''' Given a PRC and precision, return the recall.
        '''
        Rmin = self._recall_adjust_min_intercept
        alpha = self._recall_adjust_strength
        P = prec
        PRC = prc
        return 10 ** (np.log10(Rmin) * (1 - PRC / P) ** (1 / alpha))

    def prcatk_from_prcbase_alc(self, prc_base: float, alc: float) -> float:
        ''' Given a base PRC and ALC, return the PRC of the attack.
        '''
        prc_atk = (alc * (1.0-prc_base)) + prc_base
        return prc_atk
