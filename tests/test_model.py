"""Shapes and parameter count of the pillar reader."""

import pytest
import torch

pytest.importorskip("kymatio")

from pillar_reader import PillarReader  # noqa: E402
from pillar_reader.model import scattering_channels  # noqa: E402


def test_scattering_channels():
    assert scattering_channels(3, 8) == 28


def test_output_has_the_input_size():
    model = PillarReader(size=(64, 64))
    y = model(torch.randn(2, 1, 12, 64, 64))
    assert y.shape == (2, 1, 64, 64)
    assert torch.isfinite(y).all()


def test_parameter_count():
    assert sum(p.numel() for p in PillarReader().parameters()) == 121_369
