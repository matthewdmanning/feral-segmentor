"""TDD tests for ``single_epoch`` in ``feral_segmentor.training._train``.

The function under test does NOT exist yet. These tests define its contract.

================================================================================
STEP 1 — SUCCESS CRITERIA
================================================================================
``single_epoch(model, dataloader, optimizer, criterion, epoch) -> float`` must:

Correct-input behavior:
  C1. Return a Python ``float`` (the average loss over batches).
  C2. Return a *finite* float for a normal run (real model + real batches).
  C3. Return ``average = total_loss / num_batches`` (mean of per-batch losses).
  C4. With a non-None optimizer: call ``optimizer.step()`` once per batch
      (i.e. backward pass happens and weights are stepped).
  C5. With a non-None optimizer: call ``optimizer.zero_grad()`` once per batch.
  C6. With optimizer is None: NEVER call ``optimizer.step()`` (forward only,
      no backward). Verified via a spy optimizer that must not be touched, and
      by asserting no grads accumulate on model params.
  C7. Model mode (``model.training``) is NOT changed by single_epoch: whatever
      the caller set (train() or eval()) is preserved on return.
  C8. single_epoch does not itself call ``model.train()`` / ``model.eval()``.
  C9. Empty dataloader (0 batches) returns ``math.inf``.
  C10. optimizer is None path runs under ``torch.no_grad()``: model params
       have no ``.grad`` populated after the call.

Incorrect / edge-input behavior:
  E1. Empty dataloader returns math.inf regardless of optimizer being None
      or not (covered by C9 + a variant).

Deliberately NOT tested (would crash on first forward — out of contract):
  - Mismatched input/target tensor shapes.
  - Non-iterable dataloader.

================================================================================
STEP 2 — INPUTS PER CRITERION
================================================================================
  C1  : mock_model, tiny_dataloader, mock_optimizer, CrossEntropy criterion, epoch=0
  C2  : same as C1 — assert math.isfinite(result)
  C3  : constant criterion returning known constant per batch -> avg == constant
  C4  : spy_optimizer wrapping SGD; assert step_count == len(dataloader)
  C5  : spy_optimizer; assert zero_grad_count == len(dataloader)
  C6  : optimizer=None; assert no param.grad set
  C7  : set model.eval() then call; assert model.training is False after.
        set model.train() then call; assert model.training is True after.
  C8  : mode_recording_model; assert train()/eval() never invoked.
  C9  : empty_dataloader ([]) -> math.inf
  C10 : optimizer=None -> all params .grad is None
"""

import importlib.util
import math

import pytest
import torch
import torch.nn as nn

# TDD: the implementation may not exist yet. Guard the import so collection
# succeeds and all named tests are visible (as skipped) until it lands.
_TRAIN_AVAILABLE = (
    importlib.util.find_spec("feral_segmentor.training._train") is not None
)
if _TRAIN_AVAILABLE:
    from feral_segmentor.training._train import single_epoch

pytestmark = pytest.mark.skipif(
    not _TRAIN_AVAILABLE,
    reason="feral_segmentor.training._train.single_epoch not implemented yet (TDD)",
)


# ---------------------------------------------------------------------------
# Local fixtures / helpers (self-contained; do not touch broken conftest ones)
# ---------------------------------------------------------------------------


@pytest.fixture
def criterion():
    """Mean-reduction loss on raw logits vs. int64 masks (CrossEntropy)."""
    return nn.CrossEntropyLoss()


@pytest.fixture
def empty_dataloader():
    """A dataloader with zero batches."""
    return []


class _ConstantLoss(nn.Module):
    """Criterion that always returns the same scalar loss (grad-connected)."""

    def __init__(self, value: float, model_param: nn.Parameter):
        super().__init__()
        self.value = value
        self._param = model_param

    def forward(self, outputs, targets):
        # Multiply zero-tensors by outputs and the model param so the graph is
        # connected (backward is exercisable), then add the constant value.
        base = (outputs.sum() * 0.0) + (self._param.sum() * 0.0)
        return base + self.value


class _SpyOptimizer:
    """Wraps a real optimizer, recording step()/zero_grad() call counts."""

    def __init__(self, wrapped: torch.optim.Optimizer):
        self._wrapped = wrapped
        self.step_count = 0
        self.zero_grad_count = 0

    def step(self, *args, **kwargs):
        self.step_count += 1
        return self._wrapped.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        self.zero_grad_count += 1
        return self._wrapped.zero_grad(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


class _ModeRecordingModel(nn.Module):
    """Records whether train()/eval() were invoked; returns zero logits."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self._p = nn.Parameter(torch.zeros(1))
        self.train_called = False
        self.eval_called = False

    def train(self, mode: bool = True):
        # nn.Module.eval() calls self.train(False); distinguish by mode.
        if mode:
            self.train_called = True
        else:
            self.eval_called = True
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        return self._p * torch.zeros(b, self.num_classes, h, w)


@pytest.fixture
def spy_optimizer(mock_model):
    return _SpyOptimizer(torch.optim.SGD(mock_model.parameters(), lr=1e-3))


# ---------------------------------------------------------------------------
# STEP 4 — TESTS
# ---------------------------------------------------------------------------


def test_returns_float(mock_model, tiny_dataloader, mock_optimizer, criterion):
    """C1: returns a Python float."""
    result = single_epoch(mock_model, tiny_dataloader, mock_optimizer, criterion, 0)
    assert isinstance(result, float)


def test_return_is_finite(mock_model, tiny_dataloader, mock_optimizer, criterion):
    """C2: normal run yields a finite average loss."""
    result = single_epoch(mock_model, tiny_dataloader, mock_optimizer, criterion, 0)
    assert math.isfinite(result)


def test_average_equals_mean_of_batch_losses(mock_model, tiny_dataloader):
    """C3: return value is total_loss / num_batches."""
    const = 2.5
    constant_criterion = _ConstantLoss(const, mock_model._p)
    result = single_epoch(mock_model, tiny_dataloader, None, constant_criterion, 0)
    assert result == pytest.approx(const)


def test_optimizer_step_called_per_batch(
    mock_model, tiny_dataloader, spy_optimizer, criterion
):
    """C4: with an optimizer, step() runs once per batch."""
    single_epoch(mock_model, tiny_dataloader, spy_optimizer, criterion, 0)
    assert spy_optimizer.step_count == len(tiny_dataloader)


def test_optimizer_zero_grad_called_per_batch(
    mock_model, tiny_dataloader, spy_optimizer, criterion
):
    """C5: with an optimizer, zero_grad() runs once per batch."""
    single_epoch(mock_model, tiny_dataloader, spy_optimizer, criterion, 0)
    assert spy_optimizer.zero_grad_count == len(tiny_dataloader)


def test_no_optimizer_no_grad_accumulated(mock_model, tiny_dataloader, criterion):
    """C6/C10: optimizer=None -> forward only, no grads on params."""
    single_epoch(mock_model, tiny_dataloader, None, criterion, 0)
    assert all(p.grad is None for p in mock_model.parameters())


def test_model_stays_in_eval_mode(
    mock_model, tiny_dataloader, mock_optimizer, criterion
):
    """C7: model.eval() set by caller is preserved after the call."""
    mock_model.eval()
    single_epoch(mock_model, tiny_dataloader, mock_optimizer, criterion, 0)
    assert mock_model.training is False


def test_model_stays_in_train_mode(
    mock_model, tiny_dataloader, mock_optimizer, criterion
):
    """C7: model.train() set by caller is preserved after the call."""
    mock_model.train()
    single_epoch(mock_model, tiny_dataloader, mock_optimizer, criterion, 0)
    assert mock_model.training is True


def test_does_not_toggle_model_mode(tiny_dataloader):
    """C8: single_epoch never calls model.train()/model.eval() itself."""
    model = _ModeRecordingModel()
    model.train()  # caller sets mode
    model.train_called = False  # reset after caller's own call
    model.eval_called = False
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    # Shape-agnostic loss so class-count / mask-value mismatches never mask the
    # behavior under test (whether single_epoch toggles model mode).
    mode_criterion = _ConstantLoss(1.0, model._p)
    single_epoch(model, tiny_dataloader, optimizer, mode_criterion, 0)
    assert model.train_called is False
    assert model.eval_called is False


def test_empty_dataloader_returns_inf_with_optimizer(
    mock_model, empty_dataloader, mock_optimizer, criterion
):
    """C9: empty dataloader returns math.inf (optimizer present)."""
    result = single_epoch(mock_model, empty_dataloader, mock_optimizer, criterion, 0)
    assert result == math.inf


def test_empty_dataloader_returns_inf_no_optimizer(
    mock_model, empty_dataloader, criterion
):
    """C9/E1: empty dataloader returns math.inf (optimizer None)."""
    result = single_epoch(mock_model, empty_dataloader, None, criterion, 0)
    assert result == math.inf
