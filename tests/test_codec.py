import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from common import codec


def test_encode_decode_round_trip():
    tensor = torch.randint(0, 256, (1, 2, 3), dtype=torch.uint8)
    encoded = codec.encode(tensor)
    dtype = getattr(tensor.dtype, "name", tensor.dtype)
    decoded = codec.decode(encoded, dtype=dtype, shape=tensor.shape)
    assert torch.equal(decoded, tensor)

