import numpy as np

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
    def __init__(self):
        # _pcc_abs_weight is the weight given to the absolute PCC difference
        self._pcc_abs_weight = 0.5
        # _cov_adjust_min_intercept is the coverage value below which precision
        # has no effect on the PCC
        self._cov_adjust_min_intercept = 1/10000
        # Higher _cov_adjust_strength leads to lower coverage adjustment
        self._cov_adjust_strength = 3.0

    def set_param(self, param, value):
        if param == 'pcc_abs_weight':
            self._pcc_abs_weight = value
        if param == 'cov_adjust_min_intercept':
            self._cov_adjust_min_intercept = value
        if param == 'cov_adjust_strength':
            self._cov_adjust_strength = value

    def get_param(self, param):
        if param == 'pcc_abs_weight':
            return self._pcc_abs_weight
        if param == 'cov_adjust_min_intercept':
            return self._cov_adjust_min_intercept
        if param == 'cov_adjust_strength':
            return self._cov_adjust_strength
        return None

    def _cov_adjust(self, cov):
        adjust = (np.log10(cov)/ np.log10(self._cov_adjust_min_intercept)) ** self._cov_adjust_strength
        return 1 - adjust

    def _pcc_improve_absolute(self, pcc_base, pcc_attack):
        return pcc_attack - pcc_base

    def _pcc_improve_relative(self, pcc_base, pcc_attack):
        return (pcc_attack - pcc_base) / (1.00001 - pcc_base)

    def _pcc_improve(self, pcc_base, pcc_attack):
        pcc_rel = self._pcc_improve_relative(pcc_base, pcc_attack)
        pcc_abs = self._pcc_improve_absolute(pcc_base, pcc_attack)
        pcc_improve = (self._pcc_abs_weight * pcc_abs) + ((1-self._pcc_abs_weight) * pcc_rel)
        return pcc_improve

    def pcc(self, prec, cov):
        ''' Generates the precision-coverage-coefficient, PCC.
            prev is the precision of the attack, and cov is the coverage.
        '''
        if cov <= self._cov_adjust_min_intercept:
            return cov
        Cmin = self._cov_adjust_min_intercept
        alpha = self._cov_adjust_strength
        C = cov
        P = prec
        return (1-((np.log10(C)/ np.log10(Cmin)) ** alpha)) * P

    def alc(self,
                p_base = None,
                c_base = None,
                p_attack = None,
                c_attack = None,
                pcc_base = None,
                pcc_attack = None
                ):
        ''' alc can be called with either p_x and c_x, or pcc_x
        '''
        if pcc_base is None and p_base is not None and c_base is not None:
            # Adjust the precision based on the coverage to make the
            # precision-coverage-coefficient pcc
            pcc_base = self.pcc(p_base, c_base)
        if pcc_attack is None and p_attack is not None and c_attack is not None:
            pcc_attack = self.pcc(p_attack, c_attack)
        if pcc_base is not None and pcc_attack is not None:
            return self._pcc_improve(pcc_base, pcc_attack)
        return None

    # The following aren't necessary for the AnonymityLossCoefficient, but are just
    # for testing
    def prec_from_pcc_cov(self, pcc, cov):
        ''' Given a PCC and coverage, return the precision.
        '''
        Cmin = self._cov_adjust_min_intercept
        alpha = self._cov_adjust_strength
        C = cov
        PCC = pcc
        return PCC / (1 - (np.log10(C) / np.log10(Cmin)) ** alpha)

    def cov_from_pcc_prec(self, pcc, prec):
        ''' Given a PCC and precision, return the coverage.
        '''
        Cmin = self._cov_adjust_min_intercept
        alpha = self._cov_adjust_strength
        P = prec
        PCC = pcc
        return 10 ** (np.log10(Cmin) * (1 - PCC / P) ** (1 / alpha))

    def pccatk_from_pccbase_alc(self, pcc_base, alc):
        ''' Given a PCC and anonymity loss coefficient, return the PCC of the attack.
        '''
        pcc_atk = ((2*alc) - (2*alc)*pcc_base + (2*pcc_base) - (pcc_base**2)) / (2 - pcc_base)

        return pcc_atk