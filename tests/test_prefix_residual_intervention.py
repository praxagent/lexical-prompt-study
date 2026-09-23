"""Public analytical and untrained-model qualification; no pretrained inputs."""
import dataclasses
import math
import types

import pytest
import torch

from lexical_prompt_study import instruction_selection_answer_path as scoring
from lexical_prompt_study.prefix_residual_intervention import (
    PrefixResidualIntervention, PrefixResidualSite, ResidualInterventionError,
)


@pytest.fixture(autouse=True, scope="module")
def deterministic_cpu():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)


def site(width=4):
    return PrefixResidualSite(prompt_token_count=2, target_token_count=5,
                              prefix_token_count=3, hidden_width=width, absolute_position=4)


def no_hooks(block):
    assert not block._forward_hooks and not block._forward_pre_hooks


def tensor():
    return torch.arange(28, dtype=torch.float32).reshape(1, 7, 4)


def test_site_is_frozen_and_matches_supplied_prefix_not_complete_input_end():
    value = site()
    assert value.absolute_position == 4 and value.sequence_length == 7
    assert value.absolute_position != value.sequence_length - 1
    assert value.absolute_position != value.sequence_length - 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        value.absolute_position = 5


@pytest.mark.parametrize("field,value", [
    ("prompt_token_count", True), ("target_token_count", 5.),
    ("prefix_token_count", False), ("hidden_width", 4.), ("absolute_position", True),
    ("prompt_token_count", 0), ("target_token_count", 3), ("target_token_count", -1),
    ("prefix_token_count", 0), ("prefix_token_count", 5), ("hidden_width", 0),
    ("absolute_position", -1), ("absolute_position", 3),
    ("absolute_position", 5), ("absolute_position", 6),
])
def test_invalid_site_rejected(field, value):
    args = dataclasses.asdict(site())
    args[field] = value
    with pytest.raises(ResidualInterventionError):
        PrefixResidualSite(**args)


def test_capture_is_identity_detached_and_defensively_immutable():
    block = torch.nn.Identity()
    source = tensor().requires_grad_()
    expected = source[:, 4, :].detach().clone()
    hook = PrefixResidualIntervention(block, site())
    with pytest.raises(ResidualInterventionError, match="capture_unavailable"):
        _ = hook.capture
    with hook:
        output = block(source)
        assert output is source
        with pytest.raises(ResidualInterventionError, match="capture_unavailable"):
            _ = hook.capture
    assert hook.completed and hook.invocation_count == 1 and hook.replacement_count == 0
    assert not hook.capture.requires_grad and hook.capture.grad_fn is None
    obtained = hook.capture
    obtained.add_(100)
    with torch.no_grad():
        source.fill_(-9)
    assert torch.equal(hook.capture, expected)
    assert hook.capture.data_ptr() != expected.data_ptr()
    no_hooks(block)
    with pytest.raises(ResidualInterventionError, match="single_use"):
        with hook:
            pass


def test_replacement_clones_single_site_and_freezes_caller_source():
    block = torch.nn.Identity()
    source = tensor()
    original = source.clone()
    donor = torch.tensor([[91., 92., 93., 94.]], requires_grad=True)
    fixed_donor = donor.detach().clone()
    hook = PrefixResidualIntervention(block, site(), replacement=donor)
    with torch.no_grad():
        donor.fill_(-1)
    with hook:
        output = block(source)
    assert output is not source and output.data_ptr() != source.data_ptr()
    assert torch.equal(source, original)
    assert torch.equal(output[:, 4, :], fixed_donor)
    assert torch.equal(output[:, :4], original[:, :4])
    assert torch.equal(output[:, 5:], original[:, 5:])
    assert torch.equal(hook.capture, original[:, 4, :])
    assert hook.invocation_count == hook.replacement_count == 1
    output.fill_(0)
    assert torch.equal(hook.capture, original[:, 4, :])
    no_hooks(block)


@pytest.mark.parametrize("kind", ["tuple", "list", "float64", "bfloat16", "int", "meta", "sparse",
                                  "batch", "short", "long", "width", "rank", "nan", "inf", "negative_inf"])
def test_output_contract_failures_remove_hook_and_invalidate_capture(kind):
    value = tensor()
    if kind == "tuple":
        value = (value,)
    elif kind == "list":
        value = [value]
    elif kind in ("float64", "bfloat16", "int"):
        value = value.to({"float64": torch.float64, "bfloat16": torch.bfloat16, "int": torch.int64}[kind])
    elif kind == "meta":
        value = torch.empty((1, 7, 4), device="meta")
    elif kind == "sparse":
        value = value.to_sparse()
    elif kind == "batch":
        value = value.expand(2, -1, -1)
    elif kind == "short":
        value = value[:, :-1]
    elif kind == "long":
        value = torch.zeros(1, 8, 4)
    elif kind == "width":
        value = value[:, :, :-1]
    elif kind == "rank":
        value = value[0]
    else:
        value[0, 4, 1] = {"nan": float("nan"), "inf": float("inf"), "negative_inf": -float("inf")}[kind]
    block = torch.nn.Identity()
    hook = PrefixResidualIntervention(block, site())
    with pytest.raises(ResidualInterventionError):
        with hook:
            block(value)
    assert not hook.completed and hook.invocation_count == 1
    with pytest.raises(ResidualInterventionError, match="capture_unavailable"):
        _ = hook.capture
    no_hooks(block)


@pytest.mark.parametrize("kind", ["tuple", "rank", "batch", "width", "float64", "bfloat16", "meta", "nan", "inf"])
def test_replacement_contract_rejected_before_registration(kind):
    value = torch.ones(1, 4)
    if kind == "tuple":
        value = (value,)
    elif kind == "rank":
        value = value[0]
    elif kind == "batch":
        value = value.expand(2, -1)
    elif kind == "width":
        value = value[:, :-1]
    elif kind in ("float64", "bfloat16"):
        value = value.to(getattr(torch, kind))
    elif kind == "meta":
        value = torch.empty(1, 4, device="meta")
    else:
        value[0, 0] = float(kind)
    block = torch.nn.Identity()
    with pytest.raises(ResidualInterventionError):
        PrefixResidualIntervention(block, site(), replacement=value)
    no_hooks(block)


@pytest.mark.parametrize("prehook", [False, True])
def test_existing_local_hooks_rejected_without_removing_other_owners(prehook):
    block = torch.nn.Identity()
    foreign = (block.register_forward_pre_hook(lambda *args: None) if prehook
               else block.register_forward_hook(lambda *args: None))
    try:
        hook = PrefixResidualIntervention(block, site())
        with pytest.raises(ResidualInterventionError, match="unexpected_"):
            with hook:
                pytest.fail("preexisting hook should block entry")
        assert foreign.id in (block._forward_pre_hooks if prehook else block._forward_hooks)
        assert not hook.completed and hook.invocation_count == 0
    finally:
        foreign.remove()
    no_hooks(block)


def test_no_invocation_including_direct_forward_bypass_is_rejected():
    block = torch.nn.Identity()
    hook = PrefixResidualIntervention(block, site())
    with pytest.raises(ResidualInterventionError, match="hook_invocation_count"):
        with hook:
            block.forward(tensor())
    no_hooks(block)


def test_repeated_invocation_is_never_a_silent_second_noop():
    block = torch.nn.Identity()
    hook = PrefixResidualIntervention(block, site(), replacement=torch.ones(1, 4))
    with pytest.raises(ResidualInterventionError, match="hook_invocation_count"):
        with hook:
            block(tensor())
            # Even if a caller swallows the second hook error, exit must fail.
            with pytest.raises(ResidualInterventionError, match="hook_invocation_count"):
                block(tensor())
    assert hook.invocation_count == 2 and hook.replacement_count == 1 and not hook.completed
    with pytest.raises(ResidualInterventionError, match="capture_unavailable"):
        _ = hook.capture
    no_hooks(block)


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("after_hook", [False, True])
def test_exception_cleanup_preserves_original_exception_and_invalidates_capture(exception, after_hook):
    block = torch.nn.Identity()
    hook = PrefixResidualIntervention(block, site())
    error = exception("public synthetic failure")
    with pytest.raises(exception) as caught:
        with hook:
            if after_hook:
                block(tensor())
            raise error
    assert caught.value is error and not hook.completed
    assert hook.invocation_count == int(after_hook)
    no_hooks(block)
    with pytest.raises(ResidualInterventionError, match="capture_unavailable"):
        _ = hook.capture
    with PrefixResidualIntervention(block, site()) as fresh:
        block(tensor())
    assert fresh.completed
    no_hooks(block)


def test_foreign_hook_added_during_context_is_detected_and_own_hook_is_removed():
    block = torch.nn.Identity()
    hook = PrefixResidualIntervention(block, site())
    foreign = None
    try:
        with pytest.raises(ResidualInterventionError, match="unexpected_forward_hooks"):
            with hook:
                block(tensor())
                foreign = block.register_forward_hook(lambda *args: None)
        assert set(block._forward_hooks) == {foreign.id}
        assert not hook.completed
    finally:
        if foreign is not None:
            foreign.remove()
    no_hooks(block)


def test_invalid_block_and_site_types():
    with pytest.raises(ResidualInterventionError, match="block_type"):
        PrefixResidualIntervention(object(), site())
    with pytest.raises(ResidualInterventionError, match="site_type"):
        PrefixResidualIntervention(torch.nn.Identity(), dataclasses.asdict(site()))


class AnalyticalCausal(torch.nn.Module):
    """Transparent cumulative-sum causal graph, with one hookable post-block."""

    def __init__(self, *, noncausal=False):
        super().__init__()
        self.block = torch.nn.Identity()
        self.noncausal = noncausal

    def forward(self, *, input_ids, attention_mask, use_cache, logits_to_keep):
        assert not use_cache and bool((attention_mask == 1).all())
        embeddings = torch.nn.functional.one_hot(input_ids, num_classes=16).float()
        hidden = embeddings.cumsum(dim=1)
        if self.noncausal:
            hidden = hidden + embeddings.sum(dim=1, keepdim=True)
        hidden = self.block(hidden)
        logits = hidden.cumsum(dim=1) / 8
        # This row predicts beyond the input EOS and must never be scored.
        logits[:, -1, :] = float("nan")
        return types.SimpleNamespace(logits=logits[:, -logits_to_keep:])


def test_analytical_causal_alignment_suffix_effect_and_discarded_eos_next_row():
    model = AnalyticalCausal().eval()
    prompt, target = [10, 11], [3, 4, 5, 6, 1]
    location = site(16)
    baseline = scoring.teacher_force(model, torch, prompt, target, 3)
    with PrefixResidualIntervention(model.block, location) as captured:
        capture_only = scoring.teacher_force(model, torch, prompt, target, 3)
    with PrefixResidualIntervention(model.block, location, replacement=captured.capture):
        self_patch = scoring.teacher_force(model, torch, prompt, target, 3)
    assert baseline == capture_only == self_patch
    donor = captured.capture
    donor[0, 6] += 8
    with PrefixResidualIntervention(model.block, location, replacement=donor) as patched:
        result = scoring.teacher_force(model, torch, prompt, target, 3)
    assert torch.equal(patched.capture, captured.capture)
    assert result["token_logprobs"][:3] == baseline["token_logprobs"][:3]
    assert result["chosen_logits"][:3] == baseline["chosen_logits"][:3]
    assert result["log_normalizers"][:3] == baseline["log_normalizers"][:3]
    assert result["chosen_logits"][3] - baseline["chosen_logits"][3] == 1.
    assert result["chosen_logits"][4] == baseline["chosen_logits"][4]
    assert result["log_normalizers"][4] != baseline["log_normalizers"][4]
    assert result["scores"]["eos_logprob"] == result["token_logprobs"][-1]
    assert result["scores"]["answer_continuation_logprob"] == math.fsum(result["token_logprobs"][3:])
    assert all(math.isfinite(x) for x in result["token_logprobs"])
    assert result["scores"]["joint_tokens"] == len(target)
    no_hooks(model.block)


def test_candidate_invariance_control_detects_a_noncausal_model():
    prompt = [10, 11]
    targets = ([3, 4, 5, 6, 1], [3, 4, 5, 7, 1])
    for noncausal in (False, True):
        model = AnalyticalCausal(noncausal=noncausal).eval()
        captures = []
        for target in targets:
            with PrefixResidualIntervention(model.block, site(16)) as hook:
                scoring.teacher_force(model, torch, prompt, target, 3)
            captures.append(hook.capture)
        assert torch.equal(*captures) is (not noncausal)
        no_hooks(model.block)


@pytest.fixture
def tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    with torch.random.fork_rng():
        torch.manual_seed(20260923)
        config = LlamaConfig(vocab_size=32, hidden_size=24, intermediate_size=40,
                            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=64, attention_dropout=0., use_cache=False,
                            bos_token_id=0, eos_token_id=1, pad_token_id=2)
        config._attn_implementation = "sdpa"
        model = LlamaForCausalLM(config).to(device="cpu", dtype=torch.float32).eval()
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters())
    return model


def explicit_layer_logits(model, token_ids, block_index, position, replacement=None):
    """Independent full-logit reference with explicit additive causal mask.

    No production intervention helper or hooks are used. Direct layer.forward
    calls bypass hook dispatch; a concatenation inserts the replacement slice.
    """
    ids = torch.tensor([token_ids], dtype=torch.long)
    core = model.model
    count = ids.shape[1]
    positions = torch.arange(count).unsqueeze(0)
    mask = torch.full((count, count), -float("inf"), dtype=torch.float32).triu(1)[None, None]
    with torch.inference_mode():
        hidden = core.embed_tokens(ids)
        rotary = core.rotary_emb(hidden, position_ids=positions)
        for index, layer in enumerate(core.layers):
            no_hooks(layer)
            hidden = layer.forward(hidden, attention_mask=mask, position_ids=positions,
                                   position_embeddings=rotary, use_cache=False)
            assert type(hidden) is torch.Tensor
            if index == block_index and replacement is not None:
                hidden = torch.cat((hidden[:, :position], replacement[:, None, :], hidden[:, position + 1:]), dim=1)
        return model.lm_head(core.norm(hidden)).float()


def test_tiny_llama_capture_future_candidate_invariance_and_exact_identity_controls(tiny_llama):
    model = tiny_llama
    prompt = [10, 11, 12]
    targets = ([3, 4, 5, 13, 14, 1], [3, 4, 5, 15, 16, 1],
               [3, 4, 5, 17, 18, 1], [3, 4, 5, 19, 20, 1])
    location = PrefixResidualSite(3, 6, 3, 24, 5)
    block = model.model.layers[1]
    captures, scores = [], []
    for target in targets:
        baseline = scoring.teacher_force(model, torch, prompt, target, 3)
        for _ in range(2):
            with PrefixResidualIntervention(block, location) as hook:
                measured = scoring.teacher_force(model, torch, prompt, target, 3)
            assert measured == baseline
            captures.append(hook.capture)
        with PrefixResidualIntervention(block, location, replacement=captures[0]) as self_hook:
            self_score = scoring.teacher_force(model, torch, prompt, target, 3)
        assert self_score == baseline and self_hook.replacement_count == 1
        scores.append(baseline)
    assert all(torch.equal(captures[0], value) for value in captures)
    assert all(value["token_logprobs"][:3] == scores[0]["token_logprobs"][:3] for value in scores)
    for layer in model.model.layers:
        no_hooks(layer)


def test_tiny_llama_patch_matches_explicit_layer_reference_and_causal_prefix(tiny_llama):
    model = tiny_llama
    prompt, target = [10, 11, 12], [3, 4, 5, 13, 14, 1]
    location = PrefixResidualSite(3, 6, 3, 24, 5)
    block_index = 1
    block = model.model.layers[block_index]
    baseline = scoring.teacher_force(model, torch, prompt, target, 3)
    with PrefixResidualIntervention(block, location) as capture:
        scoring.teacher_force(model, torch, prompt, target, 3)
    replacement = capture.capture + torch.linspace(-.3, .3, 24)[None, :]
    with PrefixResidualIntervention(block, location, replacement=replacement):
        patched = scoring.teacher_force(model, torch, prompt, target, 3)
    manual_base = explicit_layer_logits(model, prompt + target, block_index, location.absolute_position)
    manual_patch = explicit_layer_logits(model, prompt + target, block_index, location.absolute_position, replacement)
    assert torch.equal(manual_base[:, :location.absolute_position], manual_patch[:, :location.absolute_position])
    assert not torch.equal(manual_base[:, location.absolute_position], manual_patch[:, location.absolute_position])
    assert patched["token_logprobs"][:3] == baseline["token_logprobs"][:3]
    assert patched["chosen_logits"][:3] == baseline["chosen_logits"][:3]
    assert patched["log_normalizers"][:3] == baseline["log_normalizers"][:3]
    assert patched["scores"]["answer_continuation_logprob"] != baseline["scores"]["answer_continuation_logprob"]
    for actual, full_logits in ((baseline, manual_base), (patched, manual_patch)):
        used = full_logits[0, len(prompt) - 1:len(prompt) + len(target) - 1].double()
        independent = torch.log_softmax(used, dim=-1)
        expected = [independent[index, token].item() for index, token in enumerate(target)]
        assert actual["token_logprobs"] == pytest.approx(expected, abs=2e-7, rel=0)
        assert actual["scores"]["json_prefix_logprob"] == pytest.approx(math.fsum(expected[:3]), abs=6e-7, rel=0)
        assert actual["scores"]["answer_continuation_logprob"] == pytest.approx(math.fsum(expected[3:]), abs=6e-7, rel=0)
        assert actual["scores"]["eos_logprob"] == pytest.approx(expected[-1], abs=2e-7, rel=0)
    with PrefixResidualIntervention(block, location) as fresh:
        after = scoring.teacher_force(model, torch, prompt, target, 3)
    assert after == baseline and torch.equal(fresh.capture, capture.capture)
    for layer in model.model.layers:
        no_hooks(layer)
