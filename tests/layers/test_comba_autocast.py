# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import copy

import pytest
import torch

from fla.layers.comba import Comba
from fla.utils import device


@pytest.mark.parametrize('length', [17, 65])
@pytest.mark.parametrize('use_short_conv', [False, True], ids=['no_conv', 'conv'])
@pytest.mark.parametrize(
    ('use_inner_decay', 'use_output_correction'),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_comba_autocast(length, use_short_conv, use_inner_decay, use_output_correction):
    torch.manual_seed(42)
    layer = Comba(
        hidden_size=128,
        head_dim=32,
        num_heads=2,
        expand_v=2,
        use_short_conv=use_short_conv,
        use_inner_decay=use_inner_decay,
        use_output_correction=use_output_correction,
    ).to(device=device)
    reference = copy.deepcopy(layer)
    for module in reference.modules():
        if isinstance(module, torch.nn.Linear):
            module.to(torch.bfloat16)
    x = torch.randn(1, length, 128, device=device, requires_grad=True)
    xr = x.detach().to(torch.bfloat16).requires_grad_()
    do = torch.randn_like(x)
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
        actual = layer(x)[0]
    expected = reference(xr)[0]
    assert actual.dtype == expected.dtype == torch.bfloat16
    actual_grads = torch.autograd.grad((actual * do).sum(), (x, *layer.parameters()))
    expected_grads = torch.autograd.grad((expected * do).sum(), (xr, *reference.parameters()))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for expected_grad, actual_grad in zip(expected_grads, actual_grads):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad.float(), expected_grad.float(), rtol=0, atol=0)
    assert all(parameter.dtype == torch.float32 for parameter in layer.parameters())
