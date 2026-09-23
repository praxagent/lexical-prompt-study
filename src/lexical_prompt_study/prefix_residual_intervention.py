"""One-call CPU FP32 post-block residual capture or supplied-prefix replacement.

This is an apparatus primitive, not an experiment runner or a causal claim.
The caller owns model/block provenance, native token audits and source-vector
provenance. Importing this module does not import Torch or load a model.
"""
from __future__ import annotations

from dataclasses import dataclass


class ResidualInterventionError(ValueError):
    """Fixed error codes never include tensor contents or private inputs."""


def _require(condition, code):
    if not condition:
        raise ResidualInterventionError("prefix_residual_" + code)


@dataclass(frozen=True)
class PrefixResidualSite:
    """Absolute position of the last supplied-prefix token in prompt + target.

    ``target_token_count`` includes the native final EOS. Native token identity
    and EOS correctness remain the caller's responsibility. Requiring the
    explicit position to equal P + q - 1 rejects full-input end offsets and
    accidentally selecting the input EOS or its preceding answer-ending token.
    """

    prompt_token_count: int
    target_token_count: int
    prefix_token_count: int
    hidden_width: int
    absolute_position: int

    def __post_init__(self):
        _require(all(type(value) is int for value in (
            self.prompt_token_count, self.target_token_count,
            self.prefix_token_count, self.hidden_width, self.absolute_position,
        )), "site_integer_types")
        _require(self.prompt_token_count > 0 and self.hidden_width > 0
                 and 0 < self.prefix_token_count < self.target_token_count,
                 "site_dimensions")
        _require(self.absolute_position == self.prompt_token_count + self.prefix_token_count - 1,
                 "supplied_prefix_position")

    @property
    def sequence_length(self):
        return self.prompt_token_count + self.target_token_count


class PrefixResidualIntervention:
    """Single-use context around one normal module-dispatched model forward.

    With no replacement, the hook returns ``None`` and leaves the original
    output object untouched. Otherwise it returns a clone with exactly one
    [batch, hidden] slice replaced. In both modes it captures the *original*
    slice before any replacement. After a successful context, ``capture`` gives
    a detached defensive clone; caller mutation cannot change the saved capture.
    Replacement input is detached and cloned at construction, so subsequent
    caller mutation cannot change the intervention either.

    Example::

        with PrefixResidualIntervention(block, site, replacement=donor) as hook:
            result = teacher_force(model, torch, prompt, target, q, "cpu")
        saved_vector = hook.capture

    Local forward hooks and pre-hooks must initially be absent. Concurrent use,
    gradient checkpointing, compiled/cached forwards and unexpected hook changes
    are unsupported. The caller separately audits other modules/global hooks.
    A failed forward exposes no usable capture and this object cannot be reused.
    """

    def __init__(self, block, site, *, replacement=None):
        import torch

        _require(isinstance(block, torch.nn.Module), "block_type")
        _require(type(site) is PrefixResidualSite, "site_type")
        self._torch, self._block, self._site = torch, block, site
        self._replacement = None
        if replacement is not None:
            self._validate_slice(replacement, "replacement")
            self._replacement = replacement.detach().clone()
        self._capture = None
        self._handle = None
        self._state = "new"
        self._invocations = 0
        self._replacements = 0

    @property
    def site(self):
        return self._site

    @property
    def invocation_count(self):
        return self._invocations

    @property
    def replacement_count(self):
        return self._replacements

    @property
    def completed(self):
        return self._state == "completed"

    @property
    def capture(self):
        _require(self.completed and self._capture is not None, "capture_unavailable")
        return self._capture.detach().clone()

    def _validate_slice(self, value, kind):
        torch = self._torch
        _require(isinstance(value, torch.Tensor), kind + "_tensor")
        _require(value.device.type == "cpu" and value.dtype == torch.float32
                 and value.layout == torch.strided, kind + "_cpu_fp32")
        _require(tuple(value.shape) == (1, self.site.hidden_width), kind + "_shape")
        _require(bool(torch.isfinite(value).all().item()), kind + "_finite")

    def _check_hooks(self, *, owned):
        expected = {self._handle.id} if owned else set()
        _require(not self._block._forward_pre_hooks, "unexpected_pre_hooks")
        _require(set(self._block._forward_hooks) == expected, "unexpected_forward_hooks")

    def _hook(self, module, _args, output):
        _require(self._state == "active" and module is self._block, "hook_lifecycle")
        self._invocations += 1
        _require(self._invocations == 1, "hook_invocation_count")
        self._check_hooks(owned=True)
        torch = self._torch
        _require(isinstance(output, torch.Tensor), "output_tensor")
        _require(output.device.type == "cpu" and output.dtype == torch.float32
                 and output.layout == torch.strided, "output_cpu_fp32")
        _require(tuple(output.shape) == (1, self.site.sequence_length, self.site.hidden_width),
                 "output_shape")
        source = output[:, self.site.absolute_position, :]
        self._validate_slice(source, "source")
        self._capture = source.detach().clone()
        if self._replacement is None:
            return None
        result = output.clone()
        result[:, self.site.absolute_position, :] = self._replacement
        self._replacements += 1
        return result

    def __enter__(self):
        _require(self._state == "new", "single_use")
        self._state = "failed"
        self._check_hooks(owned=False)
        self._handle = self._block.register_forward_hook(self._hook)
        try:
            self._check_hooks(owned=True)
        except BaseException:
            self._handle.remove()
            self._handle = None
            raise
        self._state = "active"
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._state = "failed"
        try:
            self._handle.remove()
        finally:
            self._handle = None
        try:
            self._check_hooks(owned=False)
            _require(self._invocations == 1, "hook_invocation_count")
            _require(self._replacements == int(self._replacement is not None), "replacement_count")
            _require(self._capture is not None, "capture_unavailable")
        except BaseException:
            self._capture = None
            if exc_type is None:
                raise
            # Preserve the original forward exception after removing our hook.
            # The object remains failed and cannot expose or reuse a capture.
            return False
        if exc_type is not None:
            self._capture = None
            return False
        self._state = "completed"
        return False
