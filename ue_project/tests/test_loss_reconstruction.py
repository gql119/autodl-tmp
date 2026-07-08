from tests.test_supervision_decomposer import make_decomposition


def test_loss_reconstruction_relative_error_is_tiny():
    dec = make_decomposition()
    assert dec.statistics["relative_reconstruction_error"] < 1.0e-5
    assert dec.statistics["absolute_reconstruction_error"] < 1.0e-5
