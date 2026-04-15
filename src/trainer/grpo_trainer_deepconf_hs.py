# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union

import torch

# ===== DeepConf-GRPO additions =====
from dataclasses import dataclass, field
from typing import Optional, List
import torch.nn.functional as F
import math
from collections import deque

@dataclass
class LatentProbeConfig:
    """Config for logging hidden-state trajectory during generation."""
    enabled: bool = False
    sample_index: int = 0       # which sample in batch to log
    max_print_steps: int = 64   # how many steps to print in cos-sim trace


def _log_latent_trajectory(
    gen_outputs,
    prompt_completion_ids,
    prompt_length,
    think_start_id,
    think_end_id,
    cfg: LatentProbeConfig,
    tag: str = "",
) -> None:
    """Side-effect-only helper: prints Δh and cos(h_t, h_final) stats for one sample."""
    if (not cfg.enabled) or (gen_outputs is None):
        return

    hidden_states = getattr(gen_outputs, "hidden_states", None)
    if hidden_states is None or len(hidden_states) == 0:
        print("[LatentProbe] hidden_states is empty, nothing to log.")
        return

    # hidden_states is typically: tuple(num_steps) of tuple(num_layers+1) of (B, T, H)
    step_tensors = []
    first_step = hidden_states[0]
    if isinstance(first_step, (tuple, list)):
        for step_hidden in hidden_states:
            last_layer = step_hidden[-1]          # (B, T, H)
            step_tensors.append(last_layer)
    else:
        # Fallback: assume already (B, T, H) per step
        for step_hidden in hidden_states:
            step_tensors.append(step_hidden)

    try:
        last_layer_traj = torch.stack(
            [h[:, -1, :] for h in step_tensors], dim=0
        )  # (S, B, H)
    except Exception as e:
        print(f"[LatentProbe] error stacking hidden states: {e}")
        return

    S, B, H = last_layer_traj.shape
    if B == 0 or S == 0:
        return

    s = cfg.sample_index
    if s >= B:
        print(f"[LatentProbe] sample_index {s} out of range (batch size {B}), skipping.")
        return

    traj = last_layer_traj[:, s, :]  # (S, H)
    if traj.size(0) < 2:
        return

    seq = prompt_completion_ids[s].tolist()
    # Locate think span if tokens exist
    think_start_pos = None
    think_end_pos = None
    if think_start_id is not None and think_end_id is not None:
        try:
            think_start_pos = seq.index(think_start_id)
            think_end_pos = seq.index(think_end_id)
        except ValueError:
            pass

    # Map from generated step t (0..S-1) to full-seq index idx = prompt_length + t
    think_step_indices = []
    if think_start_pos is not None and think_end_pos is not None and think_end_pos > think_start_pos:
        for t in range(S):
            idx = prompt_length + t
            if think_start_pos < idx < think_end_pos:
                think_step_indices.append(t)

    # L2 deltas
    diffs = traj[1:] - traj[:-1]        # (S-1, H)
    deltas = diffs.norm(dim=-1)         # (S-1,)

    think_mask = torch.zeros_like(deltas, dtype=torch.bool)
    for t in think_step_indices:
        if 0 < t < traj.size(0):
            think_mask[t - 1] = True

    think_deltas = deltas[think_mask]
    nonthink_deltas = deltas[~think_mask]

    # Cosine similarity to final state
    h_final = traj[-1]  # (H,)
    cos_to_final = F.cosine_similarity(
        traj, h_final.unsqueeze(0), dim=-1
    )  # (S,)

    k = min(cfg.max_print_steps, S)
    step_indices = list(range(k))
    cos_values = [float(x) for x in cos_to_final[:k]]

    print(f"[LatentProbe] {tag} B={B} S={S}")
    print(f"  prompt_length={prompt_length}")
    if think_start_pos is not None and think_end_pos is not None:
        print(f"  think token span: [{think_start_pos}, {think_end_pos}]")
        print(f"  generated steps in think span: {think_step_indices}")
    else:
        print("  no <IMPLICIT_THINK> span detected in this sample.")

    if think_deltas.numel() > 0:
        print(f"  mean Δh (think span):    {float(think_deltas.mean()):.6f}")
    else:
        print("  mean Δh (think span):    N/A")

    if nonthink_deltas.numel() > 0:
        print(f"  mean Δh (non-think):     {float(nonthink_deltas.mean()):.6f}")
    else:
        print("  mean Δh (non-think):     N/A")

    print(f"  cos(h_t, h_final) for first {k} steps:")
    print("    indices:", step_indices)
    print("    values: ", [round(v, 4) for v in cos_values])

    if think_step_indices:
        # Print cos-sim only at think steps (clipped)
        clipped = [t for t in think_step_indices if t < k]
        print("  cos(h_t, h_final) at think steps (clipped):")
        print("    steps:", clipped)
        print("    values:", [round(float(cos_to_final[t]), 4) for t in clipped])

@dataclass
class DeepConfConfig:
    enabled: bool = True
    window_size: int = 2048   # sliding window length over completion tokens
    stride: int = 256         # sliding step
    top_p_keep: float = 0.25  # keep top p fraction (per prompt group) by lowest-group-confidence
    weight_clip_min: float = 0.5
    weight_clip_max: float = 1.5
    standardize_over: str = "batch"  # "batch" or "group"
    # (Optional) pseudo early-stop by truncating low-confidence tails post-hoc
    truncate_below_tau: bool = False
    tau_quantile: float = 0.90
    # running buffer to estimate tau across steps (used only if truncate_below_tau=True)
    _running_lgc: deque = field(default_factory=lambda: deque(maxlen=8192), init=False, repr=False)

def _exp_logprob_confidence(logps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # logps: (B, T), mask: (B, T) in {0,1}
    probs = torch.exp(torch.clamp(logps, min=-40.0, max=0.0))
    # avoid div by zero
    denom = mask.sum(dim=1).clamp_min(1)
    return (probs * mask).sum(dim=1) / denom  # (B,)

def _lowest_group_confidence(logps: torch.Tensor, mask: torch.Tensor, win: int, stride: int) -> torch.Tensor:
    # compute per-trajectory lowest sliding-window mean(exp(logp(selected_token)))
    B, T = logps.size()
    device = logps.device
    # if very short, fall back to global mean confidence
    if T < win:
        return _exp_logprob_confidence(logps, mask)
    # build sliding windows
    vals = []
    for start in range(0, T - win + 1, stride):
        end = start + win
        wmask = mask[:, start:end]
        wlogp = logps[:, start:end]
        vals.append(_exp_logprob_confidence(wlogp, wmask))  # (B,)
    stack = torch.stack(vals, dim=1)  # (B, W)
    return stack.min(dim=1).values    # (B,)

def _standardize(x: torch.Tensor, by: str, group_ids: Optional[torch.Tensor]=None):
    # x: (N,)
    if by == "group" and group_ids is not None:
        # z-score within each group id (size G)
        out = torch.empty_like(x)
        for gid in group_ids.unique():
            idx = (group_ids == gid)
            mu = x[idx].mean()
            sd = x[idx].std().clamp_min(1e-6)
            out[idx] = (x[idx] - mu) / sd
        return out
    else:
        mu = x.mean()
        sd = x.std().clamp_min(1e-6)
        return (x - mu) / sd

def _deepconf_weights(lgc: torch.Tensor, by: str, group_ids: Optional[torch.Tensor], clip_min: float, clip_max: float) -> torch.Tensor:
    z = _standardize(lgc, by, group_ids)
    w = torch.sigmoid(z)  # map to (0,1)
    # rescale to roughly (0.5,1.5)
    w = 0.5 + w  # (0.5,1.5)
    return torch.clamp(w, min=clip_min, max=clip_max)

def _groupwise_top_p_mask2(lgc: torch.Tensor, group_size: int, top_p_keep: float) -> torch.Tensor:
    # lgc length = B*G, group_size=G
    assert lgc.numel() % group_size == 0
    B = lgc.numel() // group_size
    keep_k = max(1, int(math.ceil(group_size * top_p_keep)))
    mask = torch.zeros_like(lgc, dtype=torch.bool)
    for b in range(B):
        s = b * group_size
        e = s + group_size
        vals = lgc[s:e]
        topk = torch.topk(vals, k=keep_k, largest=True).indices
        m = torch.zeros(group_size, dtype=torch.bool, device=lgc.device)
        m[topk] = True
        mask[s:e] = m
    return mask
    
def _groupwise_top_p_mask(conf: torch.Tensor, group_size: int, top_p: float) -> torch.Tensor:
    """
    在每个组内保留 top_p 置信度的样本（布尔mask）。
    """
    assert conf.numel() % group_size == 0
    G = conf.numel() // group_size
    groups = conf.view(G, group_size)
    k = (group_size * top_p)
    k = max(1, int(round(k)))
    # 取组内 top-k 的阈值
    thresh = torch.topk(groups, k=k, dim=1, largest=True).values[:, -1].unsqueeze(1)  # (G,1)
    mask = (groups >= thresh).view_as(conf)
    return mask
# ===== End DeepConf additions =====

# ===== Pause (latent thinking) additions =====
from dataclasses import dataclass
from collections import deque
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

@dataclass
class PauseConfig:
    enabled: bool = True
    tau_pause_quantile: float = 0.50   # entropy quantile to trigger pause (higher entropy => lower confidence)
    tau_abort_quantile: float = 0.10   # very high entropy -> abort (not used to force eos here to keep simple)
    max_pauses: int = 5
    max_think_tokens: int = 128
    recovery_bonus: float = 0.05
    leak_penalty: float = 1.0
    # --- Acoustic triggers to bias <PAUSE> when text mentions non-textual, speech-only cues ---
    acoustic_keywords: list[str] = field(default_factory=lambda: [
        "tone", "intonation", "pitch", "volume", "background music", "bgm", "noise", "emotion in voice"
    ])
    acoustic_tail_tokens: int = 64
    pause_bias_on_acoustics: float = 5.0  # strong positive bias on <PAUSE> logit when acoustic keywords are detected

    # --- Repetition / ellipsis based abort triggers ---
    repetition_abort_enabled: bool = True
    rep_window_tokens: int = 64
    rep_rate_threshold: float = 0.6       # fraction of tokens in window that are repeats
    max_ident_run: int = 6                # maximum allowed identical-token run length
    ellipses_run_chars: int = 6           # e.g., '......' or '……' length to trigger
    ellipses_count_threshold: int = 2     # number of ellipsis groups (like '...' or '…') in tail to trigger
    codeblock_repeat_threshold: int = 3   # repeated tripled backticks in tail to trigger


class PauseManager:
    def __init__(self, batch_size: int, pause_token_id: int, think_start_id: int, think_end_id: int, cfg: PauseConfig):
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_think = [False]*batch_size
        self.first_think_token = [True]*batch_size
        self.pauses_used = [0]*batch_size
        self.think_left = [0]*batch_size
        self.pause_token_id = pause_token_id
        self.think_start_id = think_start_id
        self.think_end_id = think_end_id
        self.cfg = cfg
        self.ent_hist = deque(maxlen=1024)  # running entropy bank for quantiles

    def start_think(self, i: int):
        self.in_think[i] = True
        self.first_think_token[i] = True
        self.think_left[i] = self.cfg.max_think_tokens

    def end_think(self, i: int):
        self.in_think[i] = False
        self.first_think_token[i] = False
        self.think_left[i] = 0

class PauseLogitsProcessor(LogitsProcessor):
    def __init__(self, manager: PauseManager, processing_class):
        self.mgr = manager
        self.processor = processing_class
        # try to obtain eos id and a decode function
        self.eos_token_id = getattr(processing_class, 'eos_token_id', None)
        if self.eos_token_id is None and hasattr(processing_class, 'tokenizer'):
            self.eos_token_id = getattr(processing_class.tokenizer, 'eos_token_id', None)
        # define a decoder
        if hasattr(processing_class, 'decode'):
            self._decode = processing_class.decode
        elif hasattr(processing_class, 'tokenizer') and hasattr(processing_class.tokenizer, 'decode'):
            self._decode = processing_class.tokenizer.decode
        else:
            self._decode = lambda ids, **kw: ''
        # precompile acoustic keyword regex
        import re as _re
        kws = [k.lower() for k in self.mgr.cfg.acoustic_keywords]
        pat = r'(' + '|'.join(_re.escape(k) for k in kws) + r')'
        self._acoustics_re = _re.compile(pat, flags=_re.I)

    def __call__(self, input_ids, scores):
        import torch
        # scores: (B, V), input_ids: (B, T)
        B, V = scores.size()
        probs = torch.softmax(scores, dim=-1)
        entropy = -(probs * torch.clamp(torch.log(probs + 1e-12), min=-40)).sum(dim=-1)  # (B,)
        # update global entropy history
        for e in entropy.detach().cpu().tolist():
            self.mgr.ent_hist.append(e)

        # ===== Acoustic keyword-triggered <PAUSE> bias =====
        # Decode a short tail of tokens per row and, if mentions of speech-only cues appear,
        # strongly bias the <PAUSE> token to enter latent reasoning.
        try:
            tail_n = max(8, int(self.mgr.cfg.acoustic_tail_tokens))
        except Exception:
            tail_n = 64
        # We will collect per-row flags and apply bias further below in the per-row loop.
        acoustic_flags = [False] * B
        tail_texts = [None] * B
        for i in range(B):
            try:
                tail_ids = input_ids[i].tolist()[-tail_n:]
                tail_text = self._decode(tail_ids, skip_special_tokens=False)
                tail_texts[i] = tail_text
                if tail_text and self._acoustics_re.search(tail_text) is not None:
                    acoustic_flags[i] = True
            except Exception:
                acoustic_flags[i] = False
                tail_texts[i] = None
        # compute quantile thresholds (fallback defaults)
        ent_list = list(self.mgr.ent_hist)
        if len(ent_list) >= 32:
            import numpy as np
            ent_pause = float(np.quantile(ent_list, self.mgr.cfg.tau_pause_quantile))
        else:
            ent_pause = float(entropy.mean().item())

        # Now modify scores per row
        for i in range(B):
            # ----- Repetition/Ellipsis based abort triggers -----
            # These triggers complement tau_abort_quantile-based abort.
            # If any is True, we force EOS on this row.
            force_abort = False
            try:
                if self.mgr.cfg.repetition_abort_enabled:
                    W = max(16, int(self.mgr.cfg.rep_window_tokens))
                    ids = input_ids[i].tolist()[-W:]
                    if len(ids) > 0:
                        # longest identical run
                        max_run, cur_run = 1, 1
                        for a, b in zip(ids, ids[1:]):
                            if a == b:
                                cur_run += 1
                                if cur_run > max_run:
                                    max_run = cur_run
                            else:
                                cur_run = 1
                        if max_run >= int(self.mgr.cfg.max_ident_run):
                            force_abort = True
                        # repetition rate
                        uniq = len(set(ids))
                        rep_rate = 1.0 - (uniq / max(1, len(ids)))
                        if rep_rate >= float(self.mgr.cfg.rep_rate_threshold):
                            force_abort = True
                    # Ellipses / code fences in decoded tail
                    tail_text = tail_texts[i] if tail_texts[i] is not None else ""
                    if tail_text:
                        import re as _re
                        # '...' runs or many '…' chars
                        dot_runs = [m.group(0) for m in _re.finditer(r'\.{3,}', tail_text)]
                        ellip_char_runs = [m.group(0) for m in _re.finditer(r'…+', tail_text)]
                        if any(len(s) >= int(self.mgr.cfg.ellipses_run_chars) for s in dot_runs + ellip_char_runs):
                            force_abort = True
                        # multiple groups of ellipses
                        if (len(dot_runs) + len(ellip_char_runs)) >= int(self.mgr.cfg.ellipses_count_threshold):
                            force_abort = True
                        # repeated code fences
                        if tail_text.count('```') >= int(self.mgr.cfg.codeblock_repeat_threshold):
                            force_abort = True
            except Exception:
                pass

            if force_abort and self.eos_token_id is not None:
                import torch
                mask = torch.full((V,), float("-inf"), device=scores.device)
                mask[self.eos_token_id] = 0.0
                scores[i] = mask
                continue

            # ----- Acoustic keyword bias toward <PAUSE> -----
            if acoustic_flags[i]:
                try:
                    bias = float(self.mgr.cfg.pause_bias_on_acoustics)
                except Exception:
                    bias = 5.0
                scores[i, self.mgr.pause_token_id] = scores[i, self.mgr.pause_token_id] + bias
            # If in thinking mode: restrict to THINK vocab, auto open with <THINK> on first step
            if self.mgr.in_think[i]:
                mask = torch.full((V,), float("-inf"), device=scores.device)
                # allow think start on first token, otherwise allow any token except forcing THINK end when budget used up
                if self.mgr.first_think_token[i]:
                    mask[self.mgr.think_start_id] = 0.0
                else:
                    # allow all tokens, but prefer not to output visible tokens accidentally
                    # we can't perfectly mask; we just heavily downweight pause and normal eos here
                    mask[:] = 0.0
                # if budget exhausted, force </THINK>
                if self.mgr.think_left[i] <= 0:
                    mask = torch.full((V,), float("-inf"), device=scores.device)
                    mask[self.mgr.think_end_id] = 0.0
                scores[i] = scores[i] + mask
                # book-keeping for next step happens outside (via callback not available), so we approximate:
                # We'll decrement think_left when the model does NOT choose <THINK> again (i.e., after first token)
                # This will be handled by counting tokens in post-hoc; here we cannot see sampled token yet.
                continue

            # Not in think mode: maybe trigger a pause
            trigger_pause = (
                (entropy[i].item() >= ent_pause)
                and (self.mgr.cfg.enabled)
                and (self.mgr.pauses_used[i] < self.mgr.cfg.max_pauses)
                and (self.mgr.pause_token_id is not None)
            )
            if trigger_pause:
                bias = torch.zeros_like(scores[i])
                bias[self.mgr.pause_token_id] = 50.0  # strong push to emit <pause>
                scores[i] = scores[i] + bias
                # Mark that on next step we'll enter think mode (we can't modify input_ids here)
                self.mgr.pauses_used[i] += 1
                self.mgr.start_think(i)

        return scores
# ===== End Pause additions =====


import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, selective_log_softmax
from trl.trainer.callbacks import SyncRefModelCallback

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


# Based on R1-V code base, https://github.com/Deep-Agent/R1-V/blob/main/src/r1-v/src/open_r1/trainer/grpo_trainer.py
class GRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        deepconf_config ([`DeepConfConfig`], *optional*, defaults to `None`):
            DeepConf configuration for confidence-aware selection and weighting. If `None`, default configuration is used.
        pause_config ([`PauseConfig`], *optional*, defaults to `None`):
            Pause configuration for latent thinking. If `None`, default configuration is used.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        attn_implementation: str = "flash_attention_2",
        deepconf_config: Optional[DeepConfConfig] = None,
        pause_config: Optional[PauseConfig] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            if "Qwen2-Audio" in model_id:
                model = Qwen2AudioForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2-Audio" in model_id:
                 self.ref_model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif not is_peft_model(model):
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-Audio" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                processing_class.pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                print(f"Initialized Qwen2-Audio processor: {type(processing_class)}")
                print(f"Processor attributes: {dir(processing_class)}")
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        self.deepconf = deepconf_config or DeepConfConfig()
        self.pause_cfg = pause_config or PauseConfig()
        # Hidden-state trajectory logging (off by default)
        self.latent_probe_cfg = LatentProbeConfig()
        set_seed(args.seed, device_specific=True)

        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=args.temperature,
            num_return_sequences=self.num_generations,
            pad_token_id=processing_class.pad_token_id,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]
    
    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, features_values, features_masks):
        if features_values is not None and features_masks is not None:
            # Audio model
            logits = model(input_ids, attention_mask=attention_mask, input_features=features_values, feature_attention_mask=features_masks).logits  # (B, L, V)
        else:
            # Text-only model
            logits = model(input_ids, attention_mask=attention_mask).logits  # (B, L, V)
        
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
     
        # print("*"*20)
        # print(inputs[0])
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        
        # Check if this is an audio model by looking for audio data in inputs
        has_audio = "audio" in inputs[0] if inputs else False
        
        if has_audio:
            audios = [x["audio"] for x in inputs]
            # Check audio length and add debug info
            for i, audio in enumerate(audios):
                if len(audio) < 100:  # Less than ~6ms at 16kHz
                    print(f"Warning: Audio {i} is very short ({len(audio)} samples, {len(audio)/16000:.3f}s)")
            
            # Try different ways to process audio
            # print(f"Processing {len(audios)} audio samples...")
            # print(f"Audio sample shapes: {[audio.shape if hasattr(audio, 'shape') else len(audio) for audio in audios]}")
            
            # Method 1: Try with audios parameter
            try:
                prompt_inputs = self.processing_class(
                    text=prompts_text,
                    audio=audios,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                )
                # print(f"Method 1 (audios) result keys: {list(prompt_inputs.keys())}")
            except Exception as e:
                print(f"Method 1 failed: {e}")
                # Method 2: Try with audio parameter (singular)
                try:
                    prompt_inputs = self.processing_class(
                        text=prompts_text,
                        audio=audios[0] if len(audios) == 1 else audios,
                        sampling_rate=16000,
                        return_tensors="pt",
                        padding=True
                    )
                    print(f"Method 2 (audio) result keys: {list(prompt_inputs.keys())}")
                except Exception as e2:
                    print(f"Method 2 failed: {e2}")
                    # Method 3: Try with raw audio data
                    prompt_inputs = self.processing_class(
                        text=prompts_text,
                        return_tensors="pt",
                        padding=True
                    )
                    # Add raw audio data manually
                    prompt_inputs["audio"] = audios
                    print(f"Method 3 (manual) result keys: {list(prompt_inputs.keys())}")
            
            # Debug: print what keys we got from the processor
            # print(f"Final audio processing result keys: {list(prompt_inputs.keys())}")
            # if "input_features" in prompt_inputs:
            #     print(f"Audio features shape: {prompt_inputs['input_features'].shape}")
            # if "feature_attention_mask" in prompt_inputs:
            #     print(f"Audio attention mask shape: {prompt_inputs['feature_attention_mask'].shape}")
            # if "audio" in prompt_inputs:
            #     print(f"Raw audio data shape: {[a.shape if hasattr(a, 'shape') else len(a) for a in prompt_inputs['audio']]}")
        else:
            # For text-only models, use tokenizer directly
            prompt_inputs = self.processing_class(
                prompts_text,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
        
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        
        # Handle audio features only if they exist
        # Check for different possible key names for audio features
        audio_feature_keys = ["input_features", "audio_features", "features"]
        audio_mask_keys = ["feature_attention_mask", "audio_attention_mask", "attention_mask_audio"]
        
        features_values = None
        features_masks = None
        
        if has_audio:
            for feature_key in audio_feature_keys:
                if feature_key in prompt_inputs:
                    features_values = prompt_inputs[feature_key]
                    # print(f"Found audio features with key '{feature_key}': {features_values.shape}")
                    break
            
            for mask_key in audio_mask_keys:
                if mask_key in prompt_inputs:
                    features_masks = prompt_inputs[mask_key]
                    # print(f"Found audio attention mask with key '{mask_key}': {features_masks.shape}")
                    break
            
            # If we still don't have features, print available keys for debugging
            if features_values is None:
                print(f"Warning: No audio features found. Available keys: {list(prompt_inputs.keys())}")
                print(f"Expected audio feature keys: {audio_feature_keys}")
                print(f"Expected audio mask keys: {audio_mask_keys}")
                # Try to use the raw audio data as a fallback
                print("Attempting to use raw audio data...")
                if "audio" in prompt_inputs:
                    features_values = prompt_inputs["audio"]
                    # print(f"Using raw audio data: {[a.shape if hasattr(a, 'shape') else len(a) for a in features_values]}")
                    # Create a dummy attention mask for raw audio
                    features_masks = torch.ones(len(features_values), max(len(a) if hasattr(a, '__len__') else 1 for a in features_values), dtype=torch.long)
                    # print(f"Created dummy audio attention mask: {features_masks.shape}")
                else:
                    print("No raw audio data available either!")
            # else:
                # print(f"Audio features successfully extracted: {features_values.shape}")
        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions

                # ==== Build logits processors (DeepConf pause) ====
        logits_processor = None
        if self.pause_cfg.enabled:
            tok = self.processing_class
            try:
                pause_id = tok.convert_tokens_to_ids("<PAUSE>")
                think_start_id = tok.convert_tokens_to_ids("<IMPLICIT_THINK>")
                think_end_id = tok.convert_tokens_to_ids("</IMPLICIT_THINK>")
            except Exception:
                pause_id = 154931
                think_start_id = think_end_id = None
            if pause_id is not None and think_start_id is not None and think_end_id is not None:
                # batch size after prompt expansion (B*G)
                batch_size = prompt_ids.size(0)
                mgr = PauseManager(batch_size, pause_id, think_start_id, think_end_id, self.pause_cfg)
                from transformers.generation.logits_process import LogitsProcessorList
                logits_processor = LogitsProcessorList([PauseLogitsProcessor(mgr, self.processing_class)])
        else:
            think_start_id = None
            think_end_id = None

        gen_outputs = None
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            # Prepare generation inputs with audio features if available
            generation_inputs = prompt_inputs.copy()
            if has_audio and features_values is not None and features_masks is not None:
                # Try different ways to pass audio data to the model
                if isinstance(features_values, list):
                    # Raw audio data
                    generation_inputs["audio"] = features_values
                    generation_inputs["audio_attention_mask"] = features_masks
                    # print(f"Generation with raw audio data: {[a.shape if hasattr(a, 'shape') else len(a) for a in features_values]}")
                else:
                    # Processed audio features
                    generation_inputs["input_features"] = features_values
                    generation_inputs["feature_attention_mask"] = features_masks
                    # print(f"Generation with audio features: {features_values.shape}")
            elif has_audio:
                print("Warning: Audio detected but no features available for generation")
            
            # --- Generation with optional hidden-state logging ---
            try:
                gen_outputs = unwrapped_model.generate(
                    **generation_inputs,
                    generation_config=self.generation_config,
                    logits_processor=logits_processor,
                    return_dict_in_generate=True,
                    output_hidden_states=self.latent_probe_cfg.enabled,
                )
                prompt_completion_ids = gen_outputs.sequences
            except TypeError:
                # Backwards compatibility: generate() without return_dict_in_generate
                print("[LatentProbe] generate() does not support return_dict_in_generate, falling back without hidden states.")
                gen_outputs = None
                prompt_completion_ids = unwrapped_model.generate(
                    **generation_inputs,
                    generation_config=self.generation_config,
                    logits_processor=logits_processor,
                )

        # ----- Hidden-state trajectory probe -----
        prompt_length = prompt_ids.size(1)
        if gen_outputs is not None and self.latent_probe_cfg.enabled:
            _log_latent_trajectory(
                gen_outputs,
                prompt_completion_ids,
                prompt_length,
                think_start_id,
                think_end_id,
                self.latent_probe_cfg,
                tag="compute_loss",
            )

        # ----- Back to original logic -----
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]
        prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
        

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        
        # Handle audio features only if they exist
        if features_values is not None and features_masks is not None:
            features_values = features_values.repeat(self.num_generations, 1, 1)
            features_masks = features_masks.repeat_interleave(self.num_generations, dim=0)

        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, features_values, features_masks)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]
        # === DeepConf: LGC over completion tokens ===
        if self.deepconf.enabled:
            comp_logps = per_token_logps  # shape (B*G, C) after removing prompt
            comp_mask = completion_mask.float()  # (B*G, C)
            lgc = _lowest_group_confidence(comp_logps, comp_mask, self.deepconf.window_size, self.deepconf.stride)  # (B*G,)
        

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask, features_values, features_masks)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, features_values, features_masks)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)
        # ==== Recovery bonus & leak penalty (Pause) ====
        if self.pause_cfg.enabled:
            # Compute crude recovery: last-window mean exp(logp) minus first-window mean exp(logp) on completions
            def _mean_conf(x):  # x: (N, T)
                import torch
                return torch.exp(torch.clamp(x, min=-40.0, max=0.0)).mean(dim=1)
            comp_conf_first = _mean_conf(per_token_logps[:, : min(64, per_token_logps.size(1))])
            comp_conf_last  = _mean_conf(per_token_logps[:, -min(64, per_token_logps.size(1)) :])
            recovery = (comp_conf_last - comp_conf_first).clamp(min=0.0)  # (B*G,)
            rewards = rewards + self.pause_cfg.recovery_bonus * recovery

            # Leak penalty: if think tokens appear (approx)
            tok = self.processing_class
            try:
                think_start_id = tok.convert_tokens_to_ids("<IMPLICIT_THINK>")
                think_end_id = tok.convert_tokens_to_ids("</IMPLICIT_THINK>")
            except Exception:
                think_start_id = think_end_id = None
            if think_start_id is not None and think_end_id is not None:
                leak = ((completion_ids == think_start_id) | (completion_ids == think_end_id)).any(dim=1).float()
                rewards = rewards - self.pause_cfg.leak_penalty * leak
        

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # === DeepConf selection & weighting ===
        if self.deepconf.enabled:
            keep_mask = _groupwise_top_p_mask(lgc, self.num_generations, self.deepconf.top_p_keep)  # (B*G,)
            # Sample weights from standardized LGC
            Bprompts = rewards.numel() // self.num_generations
            group_ids = torch.arange(Bprompts, device=rewards.device).repeat_interleave(self.num_generations)
            weights = _deepconf_weights(lgc, self.deepconf.standardize_over, group_ids, self.deepconf.weight_clip_min, self.deepconf.weight_clip_max)
            weights = weights * keep_mask.float()
        else:
            keep_mask = torch.ones_like(rewards, dtype=torch.bool)
            weights = torch.ones_like(rewards, dtype=torch.float32)


        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss_per_seq = (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        loss_per_seq = loss_per_seq * (weights > 0).float()
        denom = (weights > 0).float().sum().clamp_min(1.0)
        loss = loss_per_seq.sum() / denom

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
