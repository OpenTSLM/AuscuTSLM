# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
AudioFlamingo: Flamingo-based audio language model.

This class is designed specifically for audio inputs, properly handling
custom audio encoders without the SimpleNamespace wrapper bug in OpenTSLMFlamingo.
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from opentslm.model.llm.TimeSeriesFlamingoWithTrainableEncoder import (
    TimeSeriesFlamingoWithTrainableEncoder,
)
from open_flamingo.src.flamingo_lm import FlamingoLMMixin
from open_flamingo.src.utils import extend_instance

from opentslm.model_config import ENCODER_OUTPUT_DIM
from opentslm.model.llm.TimeSeriesLLM import TimeSeriesLLM
from opentslm.prompt.full_prompt import FullPrompt
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)


# ─── LoRA wrapper for Linear layers in cross-attention ─────────────────────────

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a low-rank adapter.

    The original weight W is frozen and a trainable low-rank delta
    ΔW = A @ B (with scaling) is added:
        output = x @ W^T + (x @ A @ B) * (alpha / rank)

    This drastically reduces trainable params for large cross-attention
    layers while preserving full model capacity at init.
    """

    def __init__(self, original_linear: nn.Linear, rank: int = 16, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Keep original weight frozen
        self.weight = original_linear.weight
        self.weight.requires_grad_(False)
        self.bias = original_linear.bias
        if self.bias is not None:
            self.bias.requires_grad_(False)

        # Low-rank adapter
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Kaiming init for A, zero init for B → starts as identity
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B is already zeros → ΔW = 0 at init

    def forward(self, x):
        # Original forward
        result = F.linear(x, self.weight, self.bias)
        # LoRA delta
        x_drop = self.lora_dropout(x)
        lora_out = (x_drop @ self.lora_A.T) @ self.lora_B.T
        return result + lora_out * self.scaling


def apply_lora_to_cross_attention(model: nn.Module, rank: int = 16, alpha: float = 16.0, dropout: float = 0.0):
    """Replace Linear layers in gated_cross_attn_layers with LoRA variants.

    Only the low-rank A/B matrices are trainable; the original weights
    are frozen.  This reduces cross-attention trainable params by ~95%.
    """
    replaced = 0
    for name, module in model.lang_encoder.gated_cross_attn_layers.named_modules():
        # Target: to_q, to_kv, to_out in MaskedCrossAttention,
        #         and the two Linears in the FeedForward sub-module
        if isinstance(module, nn.Linear):
            # Get parent module and attribute name
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = dict(model.lang_encoder.gated_cross_attn_layers.named_modules())[parent_name]
            else:
                attr_name = parts[0]
                parent = model.lang_encoder.gated_cross_attn_layers

            lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent, attr_name, lora_layer)
            replaced += 1

    return replaced

# Monkey-patch FlamingoLayer to add attention_type property for compatibility with newer transformers
from open_flamingo.src.flamingo_lm import FlamingoLayer


def _attention_type_property(self):
    """Proxy the attention_type attribute from the underlying decoder layer."""
    return getattr(self.decoder_layer, "attention_type", None)


# Add the attention_type property to FlamingoLayer
FlamingoLayer.attention_type = property(_attention_type_property)  # type: ignore


class AudioFlamingo(TimeSeriesLLM):
    """
    Flamingo-based model for audio language modeling.
    
    Unlike OpenTSLMFlamingo, this class:
    1. Accepts a custom audio encoder directly (no SimpleNamespace wrapper)
    2. Properly handles projector for dimension mapping
    3. Passes encoder directly to TimeSeriesFlamingoWithTrainableEncoder
    
    Architecture:
        Audio → Encoder → Projector → Perceiver → Cross-Attention with LLM → Output
    
    Args:
        encoder: Audio encoder (e.g., RawAudioTokenizer, MelSpectrogramTokenizer, Wav2Vec2Encoder)
        projector: Optional MLP to map encoder output to vis_dim
        device: Device to run on ('cuda' or 'cpu')
        llm_id: HuggingFace model ID for the language model
        vis_dim: Visual dimension for Flamingo perceiver (default: ENCODER_OUTPUT_DIM)
        cross_attn_every_n_layers: Insert cross-attention every N layers
        freeze_lm_embeddings: Whether to freeze LLM input embeddings (default: True to reduce overfitting)
        freeze_encoder: Whether to freeze the audio encoder
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        projector: Optional[nn.Module] = None,
        device: str = "cuda",
        llm_id: str = "meta-llama/Llama-3.2-1B",
        vis_dim: int = ENCODER_OUTPUT_DIM,
        cross_attn_every_n_layers: int = 4,
        freeze_lm_embeddings: bool = True,
        freeze_encoder: bool = False,
        num_latents: int = 64,
        gradient_checkpointing: bool = False,
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__(device)
        print(f"AudioFlamingo using device: {self.device}")
        
        # Store encoder and projector
        self.audio_encoder = encoder.to(device)
        self.projector = projector.to(device) if projector is not None else None
        
        # Get encoder output dimension
        if hasattr(encoder, 'get_output_dim'):
            encoder_output_dim = encoder.get_output_dim()
        elif hasattr(encoder, 'output_dim'):
            encoder_output_dim = encoder.output_dim
        elif hasattr(encoder, '_output_dim'):
            encoder_output_dim = encoder._output_dim
        else:
            encoder_output_dim = vis_dim
        print(f"Encoder output dimension: {encoder_output_dim}")
        
        # If projector exists, the final dimension going to Flamingo is vis_dim
        # Otherwise, encoder output goes directly to Flamingo
        if projector is not None:
            flamingo_vis_dim = vis_dim
            print(f"Using projector: {encoder_output_dim} → {vis_dim}")
        else:
            flamingo_vis_dim = encoder_output_dim
            print(f"No projector, using encoder output directly: {encoder_output_dim}")
        
        # Create wrapped encoder that applies projector if provided.
        # The encoder stays in float32 (some ops like cuFFT don't support bfloat16)
        # and the wrapper casts its output to the LLM's dtype at the boundary.
        if projector is not None:
            class EncoderWithProjector(nn.Module):
                def __init__(self, enc, proj, output_dtype=None):
                    super().__init__()
                    self.encoder = enc
                    self.projector = proj
                    self.output_dtype = output_dtype  # set later once LLM is loaded
                
                def forward(self, x):
                    # Run encoder+projector in float32 (cuFFT requires it)
                    x_f32 = x.float()
                    enc_out = self.encoder(x_f32)  # [B, N, D_enc]
                    b, n, d = enc_out.shape
                    proj_out = self.projector(enc_out.reshape(b * n, d))  # [B*N, D_vis]
                    result = proj_out.reshape(b, n, -1)  # [B, N, D_vis]
                    # Cast output to LLM dtype (e.g. bfloat16) at the boundary
                    if self.output_dtype is not None:
                        result = result.to(self.output_dtype)
                    return result
            
            vision_encoder = EncoderWithProjector(self.audio_encoder, self.projector).to(device)
        else:
            vision_encoder = self.audio_encoder
        
        # Load text tokenizer
        text_tokenizer = AutoTokenizer.from_pretrained(
            llm_id,
            local_files_only=False,
            trust_remote_code=True,
            cache_dir=None,
        )
        
        # Load language model
        print(f"Loading LLM: {llm_id}")
        lang_encoder = AutoModelForCausalLM.from_pretrained(
            llm_id,
            local_files_only=False,
            trust_remote_code=True,
            cache_dir=None,
            device_map={"": device},
            attn_implementation="eager",
        )
        
        # Add Flamingo special tokens
        text_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
        )
        if text_tokenizer.pad_token is None:
            text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            text_tokenizer.pad_token = "<PAD>"
        
        # Convert LM to FlamingoLM
        extend_instance(lang_encoder, FlamingoLMMixin)
        
        # Infer decoder layers attribute name
        decoder_layers_attr_name = self._infer_decoder_layers_attr_name(lang_encoder)
        lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
        lang_encoder.resize_token_embeddings(len(text_tokenizer))
        
        # Fix compatibility for Gemma3Config
        if hasattr(lang_encoder.config, "text_config") and hasattr(
            lang_encoder.config.text_config, "hidden_size"
        ):
            if not hasattr(lang_encoder.config, "hidden_size"):
                lang_encoder.config.hidden_size = (
                    lang_encoder.config.text_config.hidden_size
                )
        
        # Enable gradient checkpointing on the LLM BEFORE wrapping with Flamingo.
        # This tells HuggingFace to checkpoint each transformer layer, reducing
        # activation memory by ~60% at the cost of ~30% extra compute.
        if gradient_checkpointing:
            if hasattr(lang_encoder, 'gradient_checkpointing_enable'):
                lang_encoder.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                print("✓ Gradient checkpointing ENABLED on LLM")
            else:
                print("⚠ LLM does not support gradient checkpointing")

        # Create Flamingo model - pass encoder directly (NOT wrapped in SimpleNamespace!)
        model = TimeSeriesFlamingoWithTrainableEncoder(
            vision_encoder,  # ← Direct nn.Module, not SimpleNamespace
            lang_encoder,
            text_tokenizer.encode("<|endofchunk|>")[-1],
            text_tokenizer.encode("<image>")[-1],
            vis_dim=flamingo_vis_dim,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            num_latents=num_latents,
        )
        
        print(f"Perceiver latents: {num_latents}")
        print(f"Cross-attention: every {cross_attn_every_n_layers} layers")
        
        # Count how many cross-attention blocks were created
        n_xattn = sum(1 for layer in model.lang_encoder.gated_cross_attn_layers if layer is not None)
        print(f"Cross-attention blocks: {n_xattn}")
        
        # ── Apply LoRA to cross-attention layers (before freezing) ──────────
        if lora_rank > 0:
            n_replaced = apply_lora_to_cross_attention(
                model, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout,
            )
            print(f"LoRA applied: rank={lora_rank}, alpha={lora_alpha}, "
                  f"dropout={lora_dropout}, layers replaced={n_replaced}")
        
        # Freeze all parameters first
        model.requires_grad_(False)
        
        # Unfreeze trainable components
        model.perceiver.requires_grad_(True)
        
        if lora_rank > 0:
            # LoRA mode: only unfreeze the LoRA adapters (A, B) and gates
            # The original cross-attention weights stay frozen
            for name, param in model.lang_encoder.gated_cross_attn_layers.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    param.requires_grad_(True)
                elif "gate" in name:
                    # attn_gate and ff_gate must be trainable
                    param.requires_grad_(True)
                elif "lora_dropout" in name:
                    pass  # dropout has no parameters
                # Original weights (weight, bias) stay frozen
        else:
            # Full fine-tuning mode: unfreeze all cross-attention params
            model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
        
        if not freeze_lm_embeddings:
            model.lang_encoder.get_input_embeddings().requires_grad_(True)
        
        # Unfreeze encoder (for non-pretrained encoders like mel/tokenizer)
        if not freeze_encoder:
            model.vision_encoder.requires_grad_(True)
        
        # Always unfreeze MLP projector — it's randomly initialized and must be
        # trained even when the encoder is frozen (e.g. pretrained wav2vec2/whisper).
        if hasattr(model.vision_encoder, 'projector'):
            model.vision_encoder.projector.requires_grad_(True)
        
        # Log freeze/unfreeze status
        xattn_mode = f"✅ LoRA (rank={lora_rank})" if lora_rank > 0 else "✅ trainable (full)"
        print(f"Freeze status:")
        print(f"  LLM backbone:      ❄️ frozen")
        print(f"  LLM embeddings:    {'❄️ frozen' if freeze_lm_embeddings else '✅ trainable'}")
        print(f"  Audio encoder:     {'❄️ frozen' if freeze_encoder else '✅ trainable'}")
        print(f"  MLP projector:     ✅ trainable (always)")
        print(f"  Perceiver:         ✅ trainable")
        print(f"  Cross-attention:   {xattn_mode}")
        
        # Unify dtypes for the Flamingo-side components.
        # The audio encoder + projector STAY in float32 (cuFFT doesn't support bfloat16).
        # The EncoderWithProjector wrapper casts its output at the boundary.
        # Perceiver and cross-attention layers are cast to match the LLM.
        lm_dtype = next(lang_encoder.parameters()).dtype
        if lm_dtype != torch.float32:
            print(f"LLM dtype is {lm_dtype} — casting perceiver/cross-attn to match")
            print(f"  Encoder/projector stay float32 (cuFFT requirement)")
            model.perceiver.to(lm_dtype)
            model.lang_encoder.gated_cross_attn_layers.to(lm_dtype)
            # Tell the encoder wrapper to cast its output to the LLM dtype
            if hasattr(model.vision_encoder, 'output_dtype'):
                model.vision_encoder.output_dtype = lm_dtype
        
        self.model = model
        self.llm = model
        self.text_tokenizer = text_tokenizer
        self.tokenizer = text_tokenizer  # Alias for compatibility
        self.encoder = self.audio_encoder  # Alias for compatibility
        
        # Print trainable parameter count
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    def _infer_decoder_layers_attr_name(self, model):
        """Infer the attribute name for decoder layers based on model architecture."""
        __KNOWN_DECODER_LAYERS_ATTR_NAMES = {
            "opt": "model.decoder.layers",
            "gptj": "transformer.h",
            "gpt-j": "transformer.h",
            "pythia": "gpt_neox.layers",
            "llama": "model.layers",
            "gptneoxforcausallm": "gpt_neox.layers",
            "mpt": "transformer.blocks",
            "mosaicgpt": "transformer.blocks",
            "gemma": "model.layers",
            "gemma2": "model.layers",
            "gemma3": "model.layers",
            "medgemma": "model.layers",
        }
        
        model_class_name = model.__class__.__name__
        if "gemma3" in model_class_name.lower():
            if "ConditionalGeneration" in model_class_name:
                return "language_model.layers"
            else:
                return "model.layers"
        
        for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
            if k.lower() in model.__class__.__name__.lower():
                return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]
        
        raise ValueError(
            f"Unknown model architecture: {model.__class__.__name__}. "
            "Please supply decoder_layers_attr_name manually."
        )
    
    def pad_and_apply_batch(
        self,
        batch: List[Dict[str, any]],
        include_labels: bool,
        return_answer_tail_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Process a batch of audio-text pairs for training or inference."""
        
        MAX_NUM_SERIES = 8  # Cap to prevent GPU OOM from huge padded tensors

        def pad_time_series(batch, max_length=None):
            """Pad time series to the same length AND same number of series.

            Each item's time_series has shape [num_series, length].
            Different patients may have different num_series (e.g. CirCor has
            4 valve recordings, SPRSound can have dozens of segments).
            We pad both dimensions so all items can be stacked into
            [B, max_num_series, max_length].

            To prevent GPU OOM on outlier samples with dozens of series,
            num_series is capped at MAX_NUM_SERIES.  Excess series are
            **randomly sampled** (not truncated) so that repeated calls
            cover different clips over time.
            """
            time_series = [item["time_series"] for item in batch]

            # Randomly sample if more series than the cap (rather than
            # always taking the first k, which would never see later clips).
            sampled = []
            for ts in time_series:
                if ts.shape[0] > MAX_NUM_SERIES:
                    indices = sorted(random.sample(range(ts.shape[0]), MAX_NUM_SERIES))
                    ts = ts[indices]
                sampled.append(ts)
            time_series = sampled
            
            if max_length is None:
                max_length = max(ts.shape[1] for ts in time_series)
            max_num_series = max(ts.shape[0] for ts in time_series)
            
            padded_series = []
            for ts in time_series:
                # Pad length dimension (dim 1)
                current_length = ts.shape[1]
                if current_length < max_length:
                    padding_shape = list(ts.shape)
                    padding_shape[1] = max_length - current_length
                    padding = torch.zeros(
                        padding_shape, device=ts.device, dtype=ts.dtype
                    )
                    padded = torch.cat([ts, padding], dim=1)
                else:
                    padded = ts[:, :max_length]
                
                # Pad num_series dimension (dim 0)
                current_num = padded.shape[0]
                if current_num < max_num_series:
                    series_padding = torch.zeros(
                        max_num_series - current_num, padded.shape[1],
                        device=padded.device, dtype=padded.dtype,
                    )
                    padded = torch.cat([padded, series_padding], dim=0)
                
                padded_series.append(padded)
            
            return torch.stack(padded_series)
        
        tokenizer = self.text_tokenizer
        media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
        endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]
        
        # Process audio data — keep float32 (encoder handles dtype casting internally)
        images = pad_time_series(batch).to(self.device, non_blocking=True)
        images = images.unsqueeze(1)  # Add time dimension [B, 1, F, L]
        
        # Process text
        text_inputs = []
        prompt_lengths = []
        answer_tail_start_tokens = []
        
        for item in batch:
            prompt_text = item["pre_prompt"]
            for ts_text in item["time_series_text"]:
                prompt_text += f" {tokenizer.decode([media_token_id])} {ts_text} {tokenizer.decode([endofchunk_token_id])}"
            if item["post_prompt"]:
                prompt_text += f" {item['post_prompt']}"
            
            if include_labels:
                text_inputs.append(prompt_text)
                continue
            
            prompt_tokens = tokenizer(prompt_text, add_special_tokens=True).input_ids
            prompt_lengths.append(len(prompt_tokens))
            
            answer_text = str(item["answer"])
            full_text = prompt_text + f" {answer_text}"
            text_inputs.append(full_text)

            # Track where the final answer span starts (after last "Answer:")
            # so loss can upweight that tail relative to long rationale tokens.
            tail_start = None
            answer_lower = answer_text.lower()
            marker_idx = answer_lower.rfind("answer:")
            if marker_idx != -1:
                prefix_before_tail = prompt_text + f" {answer_text[:marker_idx]}"
                prefix_tokens = tokenizer(prefix_before_tail, add_special_tokens=True).input_ids
                tail_start = len(prefix_tokens)
            answer_tail_start_tokens.append(tail_start)
        
        tokenized = tokenizer(text_inputs, padding="longest", return_tensors="pt")
        input_ids = tokenized.input_ids.to(self.device, non_blocking=True)
        attention_mask = tokenized.attention_mask.to(self.device, non_blocking=True)
        
        if include_labels:
            return input_ids, images, attention_mask, None, None
        
        # Create labels (-100 for prompt tokens, actual tokens for answer)
        labels = torch.full_like(input_ids, -100)
        answer_tail_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        for i, prompt_length in enumerate(prompt_lengths):
            non_padding_indices = torch.where(input_ids[i] != tokenizer.pad_token_id)[0]
            answer_indices = non_padding_indices[non_padding_indices >= prompt_length]
            if len(answer_indices) > 0:
                labels[i, answer_indices] = input_ids[i, answer_indices]
                tail_start = answer_tail_start_tokens[i]
                if tail_start is not None:
                    tail_indices = answer_indices[answer_indices >= tail_start]
                    if len(tail_indices) > 0:
                        answer_tail_mask[i, tail_indices] = 1.0

        if return_answer_tail_mask:
            return input_ids, images, attention_mask, labels, answer_tail_mask
        return input_ids, images, attention_mask, labels, None
    
    def generate(
        self, batch: List[Dict[str, any]], max_new_tokens: int = 50, **generate_kwargs
    ) -> List[str]:
        """Generate text responses for audio inputs."""
        original_disable = torch._dynamo.config.disable
        torch._dynamo.config.disable = True
        
        try:
            with torch.inference_mode():
                input_ids, images, attention_mask, _, _ = self.pad_and_apply_batch(
                    batch, include_labels=True
                )
                
                # Prepare generation kwargs
                # Note: eos_token_id and pad_token_id are not passed to Flamingo.generate()
                # as the installed open-flamingo version doesn't accept them in kwargs.
                # The underlying language model will use its default eos_token_id.
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    **generate_kwargs,
                }

                gen_ids = self.llm.generate(
                    vision_x=images,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )
                
                answer_only_ids = gen_ids[:, input_ids.shape[1]:]
                return self.text_tokenizer.batch_decode(
                    answer_only_ids, skip_special_tokens=True
                )
        finally:
            torch._dynamo.config.disable = original_disable
    
    def compute_loss(
        self,
        batch: List[Dict[str, any]],
        sample_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        answer_tail_weight: float = 1.0,
    ) -> torch.Tensor:
        """Compute cross-entropy loss for a batch.

        Args:
            batch: List of sample dicts produced by
                ``CareSoundDataset.__getitem__``.
            sample_weights: Optional per-sample weight tensor of shape
                ``(B,)``.  When provided the loss is computed per-sample
                and weighted by these values (useful for dataset-aware
                loss weighting to counteract class/source imbalance).
                When ``None`` the standard mean-reduction CE is used
                (fast path, no extra overhead).
            label_smoothing: Label smoothing factor (0.0 = no smoothing,
                0.1 = recommended for CoT).  Softens one-hot targets to
                prevent overconfident predictions on rationale tokens.
            answer_tail_weight: Multiplicative loss weight applied to tokens
                in the final answer span (tokens after the last "Answer:"
                marker in target text). 1.0 disables weighting.

        Returns:
            Scalar loss tensor.
        """
        input_ids, images, attention_mask, labels, answer_tail_mask = self.pad_and_apply_batch(
            batch,
            include_labels=False,
            return_answer_tail_mask=(answer_tail_weight > 1.0),
        )

        use_custom_loss = (
            (sample_weights is not None)
            or (label_smoothing > 0.0)
            or (answer_tail_weight > 1.0)
        )

        if not use_custom_loss:
            # ── Fast path: standard mean CE (no per-sample overhead) ──
            output = self.model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            return output[0]

        # ── Custom loss path: label smoothing and/or sample weights ───
        # Forward WITHOUT labels so the HF LLM returns logits only
        output = self.model(
            vision_x=images,
            lang_x=input_ids,
            attention_mask=attention_mask,
            labels=None,  # skip internal loss computation
        )
        logits = output.logits if hasattr(output, "logits") else output[0]

        # Causal-LM shift: predict token t+1 from token t
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Per-token CE (reduction='none' → shape [B, T])
        loss_fct = torch.nn.CrossEntropyLoss(
            reduction="none",
            ignore_index=-100,
            label_smoothing=label_smoothing,
        )
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        if answer_tail_weight > 1.0 and answer_tail_mask is not None:
            shift_tail_mask = answer_tail_mask[..., 1:].contiguous()
            token_weights = torch.ones_like(per_token_loss)
            token_weights = token_weights + (answer_tail_weight - 1.0) * shift_tail_mask
            per_token_loss = per_token_loss * token_weights

        # Average over valid tokens per sample → shape [B]
        valid_mask = (shift_labels != -100).float()
        per_sample_loss = (
            (per_token_loss * valid_mask).sum(dim=1)
            / valid_mask.sum(dim=1).clamp(min=1)
        )

        if sample_weights is not None:
            # Weighted mean
            w = sample_weights.to(per_sample_loss.device)
            return (per_sample_loss * w).sum() / w.sum()
        else:
            # Simple mean (label smoothing only, no sample weights)
            return per_sample_loss.mean()
    
    # ── Stage 2 (CoT) reconfiguration ─────────────────────────────────────────

    def reconfigure_for_cot(self, cot_dropout: float = 0.3):
        """Reconfigure the model for Stage 2 CoT fine-tuning.

        Keeps the same components trainable as Stage 1 (encoder [if lightweight],
        perceiver, projector, cross-attention).  Only the dropout rates are
        bumped for stronger regularisation during CoT training.

        Call this **after** loading the best Stage 1 checkpoint and
        **before** creating the Stage 2 optimizer.

        Args:
            cot_dropout: Dropout rate applied to cross-attention / LoRA layers
                         during CoT training (higher than Stage 1 to prevent
                         memorisation).
        """
        model = self.model

        # Bump dropout for stronger regularisation during CoT.
        # requires_grad flags are left unchanged (same trainable set as Stage 1).
        has_lora = any(
            "lora_A" in name or "lora_B" in name
            for name, _ in model.lang_encoder.gated_cross_attn_layers.named_parameters()
        )
        if has_lora:
            for module in model.lang_encoder.gated_cross_attn_layers.modules():
                if isinstance(module, LoRALinear) and hasattr(module, "lora_dropout"):
                    if isinstance(module.lora_dropout, nn.Dropout):
                        module.lora_dropout.p = cot_dropout
        else:
            for module in model.lang_encoder.gated_cross_attn_layers.modules():
                if isinstance(module, nn.Dropout):
                    module.p = cot_dropout

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        # In self.model (Flamingo), encoder params are under "vision_encoder.encoder.*"
        # and projector params are under "vision_encoder.projector.*".
        def _frozen(prefix):
            params = [p for n, p in model.named_parameters() if n.startswith(prefix)]
            if not params:
                return None  # prefix not found — can't determine
            return not any(p.requires_grad for p in params)

        def _status(frozen):
            if frozen is None:
                return "❓ (not found)"
            return "❄️ frozen" if frozen else "🔥 trainable"

        print(f"\n{'='*60}")
        print(f"RECONFIGURED FOR COT (Stage 2) — same trainable set as Stage 1")
        print(f"{'='*60}")
        print(f"  Encoder:      {_status(_frozen('vision_encoder.encoder'))}")
        print(f"  Projector:    {_status(_frozen('vision_encoder.projector'))}")
        print(f"  Perceiver:    {_status(_frozen('perceiver'))}")
        print(f"  Cross-attn:   🔥 trainable")
        print(f"  LLM backbone: ❄️ frozen")
        print(f"  CoT dropout:  {cot_dropout}")
        print(f"  Total trainable: {trainable:,} / {total:,} "
              f"({100*trainable/total:.3f}%)")
        print(f"{'='*60}\n")

        return trainable

    def get_eos_token(self) -> str:
        return self.text_tokenizer.eos_token
    
    def store_to_file(self, path: str = "best_audio_model.pt"):
        """Save model checkpoint."""
        state_dict = {
            "llm": self.llm.state_dict(),
        }
        torch.save(state_dict, path)
        print(f"Model saved to {path}")
    
    def load_from_file(self, path: str = "best_audio_model.pt"):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if "llm" in checkpoint:
            model_state = checkpoint["llm"]
        elif "model_state" in checkpoint:
            model_state = checkpoint["model_state"]
        else:
            raise RuntimeError("No recognized model state key in checkpoint.")
        
        if hasattr(self, "module"):
            model_state = {f"module.{k}": v for k, v in model_state.items()}
        
        # Remove prefixes ('model.' or 'llm.') if present in checkpoint keys
        # The checkpoint may have been saved with keys prefixed with 'model.' or 'llm.'
        # but the actual Flamingo model expects keys without these prefixes
        new_model_state = {}
        prefix_removed_count = 0
        for k, v in model_state.items():
            new_key = k
            # Remove 'llm.' prefix if present (common when saved as checkpoint["llm"])
            if new_key.startswith("llm."):
                new_key = new_key[4:]  # Remove "llm." (4 characters)
                prefix_removed_count += 1
            # Remove 'model.' prefix if present
            elif new_key.startswith("model."):
                new_key = new_key[6:]  # Remove "model." (6 characters)
                prefix_removed_count += 1
            new_model_state[new_key] = v
        model_state = new_model_state

        if prefix_removed_count > 0:
            print(f"ℹ️  Removed prefix from {prefix_removed_count} checkpoint keys")

        # The checkpoint contains state for self.model (the Flamingo model), not self
        missing_keys, unexpected_keys = self.model.load_state_dict(model_state, strict=False)
        if missing_keys:
            print(f"⚠️  Warning: Missing keys: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"⚠️  Warning: Unexpected keys: {unexpected_keys[:5]}...")
        self.to(self.device)
    
    def eval_prompt(
        self, prompt: FullPrompt, max_new_tokens: int = 1000, normalize: bool = False
    ) -> str:
        """Evaluate a single prompt."""
        original_disable = torch._dynamo.config.disable
        torch._dynamo.config.disable = True
        try:
            batch = [prompt.to_dict()]
            self.eval()
            batch = extend_time_series_to_match_patch_size_and_aggregate(
                batch, normalize=normalize
            )
            output = self.generate(batch, max_new_tokens=max_new_tokens)
            return output[0]
        finally:
            torch._dynamo.config.disable = original_disable
