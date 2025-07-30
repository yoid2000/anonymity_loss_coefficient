import pytest
from anonymity_loss_coefficient.alc.params import ALCParams

def test_add_group_and_access():
    ap = ALCParams()
    ap.add_group('mygroup', foo=123, bar='baz')
    assert hasattr(ap, 'mygroup')
    assert ap.mygroup.foo == 123
    assert ap.mygroup.bar == 'baz'

def test_set_param_existing_group():
    ap = ALCParams()
    ap.set_param(ap.alcm, "halt_low_acl", 0.5)
    assert ap.alcm.halt_low_acl == 0.5

def test_set_param_new_param_in_group():
    ap = ALCParams()
    ap.set_param(ap.alcm, "new_param", 42)
    assert hasattr(ap.alcm, "new_param")
    assert ap.alcm.new_param == 42

def test_set_param_none_value():
    ap = ALCParams()
    ap.set_param(ap.alcm, "halt_low_acl", None)
    # Should not change the value
    assert ap.alcm.halt_low_acl == 0.25
            
def test_add_group_then_set_params():
    ap = ALCParams()
    ap.add_group('atk')
    ap.set_param(ap.atk, 'new_param', 42)
    assert ap.atk.new_param == 42