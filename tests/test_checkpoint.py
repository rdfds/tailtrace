import pytest

torch = pytest.importorskip("torch")

from tailtrace.checkpoint import restore, save
from tailtrace.config import TrainConfig
from tailtrace.model import CausalTransformer, make_batch, token_loss


def update(model, optim, config, ids):
    optim.zero_grad(set_to_none=True)
    x, y = make_batch(ids, [3, 8, 4], 32, config.seed, torch.device("cpu"))
    loss, _ = token_loss(model(x), y, sum([3, 8, 4][i] - 1 for i in ids), 1)
    loss.backward()
    optim.step()


def test_restore_optimizer_and_next_update(tmp_path):
    torch.set_num_threads(1)
    config = TrainConfig(width=16, heads=2, layers=1, max_length=8, min_length=2, vocab_size=32)
    torch.manual_seed(2)
    model = CausalTransformer(config)
    optim = torch.optim.AdamW(model.parameters(), lr=config.lr)
    update(model, optim, config, [0, 1])
    save(tmp_path / "checkpoint", model, optim, config, 1, 1, 0)
    restored = CausalTransformer(config)
    restored_optim = torch.optim.AdamW(restored.parameters(), lr=config.lr)
    assert restore(tmp_path / "checkpoint", restored, restored_optim, config, 1) == 1
    update(model, optim, config, [2, 0])
    update(restored, restored_optim, config, [2, 0])
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_incomplete_checkpoint_rejected(tmp_path):
    with pytest.raises(ValueError, match="commit marker"):
        restore(tmp_path, None, None, TrainConfig(), 1)
