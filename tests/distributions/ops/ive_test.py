import torch

from hyperspherical_vae.distributions.ops import ive


def test_ive_gradcheck():
    test_input = torch.tensor(
        [0.6373, 1.5867, 0.1988, 1.4114, 0.7618],
        requires_grad=True,
        dtype=torch.float64,
    )
    assert torch.autograd.gradcheck(lambda x: ive(0.0, x), (test_input,))
