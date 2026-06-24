import pytest
import numpy as np

from anonymity_loss_coefficient import AnonymityLossCoefficient


def test_prc_defaults_to_fbeta():
    alc = AnonymityLossCoefficient()

    beta = 0.05
    expected = (1 + beta ** 2) * (0.8 * 0.4) / ((beta ** 2 * 0.8) + 0.4)

    assert alc.prc(prec=0.8, recall=0.4) == round(expected, 4)


def test_fbeta_uses_configured_beta():
    alc = AnonymityLossCoefficient(beta=2.0)

    expected = (1 + 2.0 ** 2) * (0.8 * 0.4) / ((2.0 ** 2 * 0.8) + 0.4)

    assert alc.prc(prec=0.8, recall=0.4) == round(expected, 4)


def test_prc_type_can_use_adjusted_prc():
    alc = AnonymityLossCoefficient(prc_type='prc')

    expected = (1 - ((np.log10(0.4) / np.log10(1/10000)) ** 3)) * 0.8

    assert alc.prc(prec=0.8, recall=0.4) == round(expected, 4)


def test_invalid_prc_type_raises():
    with pytest.raises(ValueError, match='prc_type'):
        AnonymityLossCoefficient(prc_type='unknown')
