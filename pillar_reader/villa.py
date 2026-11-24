"""Run the pillar reader inside the ScrollPrize/villa ink-detection pipeline.

villa (https://github.com/ScrollPrize/villa, MIT) provides data loading, augmentation, the loss,
the optimiser, checkpoints and sliding-window inference. This module only swaps the model that
villa builds: nothing else in the pipeline is changed.

    python -m pillar_reader.villa train <config.json>
    python -m pillar_reader.villa infer <surface-volume.zarr> <checkpoint.pth> <output.tif> [...]

Run it from villa's `vesuvius` folder, with this repository on PYTHONPATH (see README.md).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from torch import nn

from pillar_reader.model import PillarReader


class PillarReaderForVilla(nn.Module):
    """villa expects a dictionary of outputs per target; ours is called "ink"."""

    def __init__(self, size: tuple[int, int]) -> None:
        super().__init__()
        self.reader = PillarReader(size=size)

    def forward(self, x):
        return {"ink": self.reader(x)}


def make_model(config) -> nn.Module:
    """Replaces villa's `make_model`. The villa config is only read for the patch size."""
    size = (256, 256)
    try:
        size = tuple(int(v) for v in config.data.patch_size[-2:])
    except AttributeError:
        pass
    return PillarReaderForVilla(size=size)


def plug_in() -> None:
    import vesuvius.ink_detection.inference.infer as inference
    import vesuvius.ink_detection.models.model as models

    models.make_model = make_model
    inference.make_model = make_model  # imported by name when the module loads


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in {"train", "infer"}:
        print(__doc__)
        return 2
    command, rest = argv[0], argv[1:]
    plug_in()
    if command == "train":
        from vesuvius.ink_detection.training.train import main as train

        return int(train(rest) or 0)
    from vesuvius.ink_detection.inference.infer import main as infer

    return int(infer(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
