#!/usr/bin/env python3
"""
VietOCR Optimizations - Collection of performance improvements
"""

import gc
import time
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


class OptimizedTrainer:
    """
    Optimized version of Trainer with performance improvements
    """

    def __init__(self, config, pretrained=True):
        self.config = config
        self.device = config["device"]

        # Mixed precision training
        self.scaler = GradScaler()
        self.use_amp = True

        # Memory optimization
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)

        # Early stopping
        self.early_stopping_patience = config.get("early_stopping_patience", 10)
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Initialize model and other components
        self._initialize_model(pretrained)

    def _initialize_model(self, pretrained):
        """Initialize model with optimizations"""
        from vietocr.tool.translate import build_model

        self.model, self.vocab = build_model(self.config)

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model.transformer, "gradient_checkpointing_enable"):
            self.model.transformer.gradient_checkpointing_enable()

        # Move to device
        self.model = self.model.to(self.device)

        # Initialize optimizer with better settings
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["optimizer"]["max_lr"],
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=0.01,
        )

        # Better learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            total_steps=self.config["trainer"]["iters"],
            **self.config["optimizer"],
        )

    def training_step(self, batch):
        """Optimized training step with mixed precision"""
        self.model.train()

        # Move batch to device
        batch = self._batch_to_device(batch)

        # Mixed precision forward pass
        with autocast(enabled=self.use_amp):
            outputs = self.model(
                batch["img"],
                batch["tgt_input"],
                tgt_key_padding_mask=batch["tgt_padding_mask"],
            )

            outputs = outputs.view(-1, outputs.size(2))
            tgt_output = batch["tgt_output"].view(-1)

            loss = F.cross_entropy(outputs, tgt_output, ignore_index=0)
            loss = loss / self.gradient_accumulation_steps

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()

        # Gradient accumulation
        if (self.current_step + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()

        return loss.item() * self.gradient_accumulation_steps

    def validation_step(self, batch):
        """Optimized validation step"""
        self.model.eval()

        with torch.no_grad():
            batch = self._batch_to_device(batch)

            # Check for NaN in inputs
            if torch.isnan(batch["img"]).any():
                return None

            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    batch["img"],
                    batch["tgt_input"],
                    tgt_key_padding_mask=batch["tgt_padding_mask"],
                )

                outputs = outputs.view(-1, outputs.size(2))
                tgt_output = batch["tgt_output"].view(-1)

                loss = F.cross_entropy(outputs, tgt_output, ignore_index=0)

            return loss.item()

    def _batch_to_device(self, batch):
        """Optimized batch transfer to device"""
        return {
            "img": batch["img"].to(self.device, non_blocking=True),
            "tgt_input": batch["tgt_input"].to(self.device, non_blocking=True),
            "tgt_output": batch["tgt_output"].to(self.device, non_blocking=True),
            "tgt_padding_mask": batch["tgt_padding_mask"].to(
                self.device, non_blocking=True
            ),
            "filenames": batch["filenames"],
        }


class MemoryEfficientAttention(nn.Module):
    """
    Memory efficient attention implementation
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, chunk_size=512):
        batch_size, seq_len, _ = q.shape

        # Project queries, keys, values
        q = (
            self.q_proj(q)
            .view(batch_size, seq_len, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(k)
            .view(batch_size, seq_len, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(v)
            .view(batch_size, seq_len, self.nhead, self.head_dim)
            .transpose(1, 2)
        )

        # Chunked attention for memory efficiency
        output = []
        for i in range(0, seq_len, chunk_size):
            end = min(i + chunk_size, seq_len)
            q_chunk = q[:, :, i:end, :]

            # Compute attention for chunk
            attn_output = self._attention_chunk(q_chunk, k, v, mask)
            output.append(attn_output)

        # Concatenate chunks
        output = torch.cat(output, dim=2)
        output = (
            output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )

        return self.out_proj(output)

    def _attention_chunk(self, q, k, v, mask=None):
        """Compute attention for a chunk of queries"""
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, v)


class OptimizedBeamSearch:
    """
    Optimized beam search with caching and early termination
    """

    def __init__(self, beam_size=4, cache_size=1000, early_termination=True):
        self.beam_size = beam_size
        self.cache_size = cache_size
        self.early_termination = early_termination
        self.cache = {}

    def search(self, memory, model, max_length=128, sos_token=1, eos_token=2):
        """Optimized beam search with caching"""
        # Create cache key
        cache_key = self._create_cache_key(memory, max_length)

        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Perform beam search
        result = self._beam_search(memory, model, max_length, sos_token, eos_token)

        # Cache result
        if len(self.cache) < self.cache_size:
            self.cache[cache_key] = result

        return result

    def _create_cache_key(self, memory, max_length):
        """Create cache key from memory"""
        # Use hash of memory tensor for caching
        return hash(memory.cpu().numpy().tobytes()) + max_length

    def _beam_search(self, memory, model, max_length, sos_token, eos_token):
        """Actual beam search implementation"""
        from vietocr.model.beam import Beam

        beam = Beam(
            beam_size=self.beam_size,
            min_length=0,
            n_top=1,
            start_token_id=sos_token,
            end_token_id=eos_token,
        )

        device = memory.device

        with torch.no_grad():
            memory = model.transformer.expand_memory(memory, self.beam_size)

            for step in range(max_length):
                tgt_inp = beam.get_current_state().transpose(0, 1).to(device)
                decoder_outputs, memory = model.transformer.forward_decoder(
                    tgt_inp, memory
                )

                log_prob = F.log_softmax(decoder_outputs[:, -1, :].squeeze(0), dim=-1)
                beam.advance(log_prob.cpu())

                # Early termination
                if self.early_termination and beam.done():
                    break

            scores, ks = beam.sort_finished(minimum=1)
            hypothesis = beam.get_hypothesis(ks[0][0][0], ks[0][0][1])

        return [sos_token] + [int(i) for i in hypothesis[:-1]]


class DynamicBatching:
    """
    Dynamic batching for optimal GPU utilization
    """

    def __init__(self, max_batch_size=64, max_tokens=4096):
        self.max_batch_size = max_batch_size
        self.max_tokens = max_tokens

    def create_batches(self, samples):
        """Create optimal batches based on sequence lengths"""
        # Sort samples by sequence length
        samples_with_lengths = [(s, len(s["word"])) for s in samples]
        samples_with_lengths.sort(key=lambda x: x[1])

        batches = []
        current_batch = []
        current_tokens = 0

        for sample, length in samples_with_lengths:
            # Check if adding this sample would exceed limits
            if (
                len(current_batch) >= self.max_batch_size
                or current_tokens + length > self.max_tokens
            ):

                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0

            current_batch.append(sample)
            current_tokens += length

        # Add remaining batch
        if current_batch:
            batches.append(current_batch)

        return batches


class AdaptiveAugmentation:
    """
    Adaptive augmentation based on training progress
    """

    def __init__(self, initial_strength=0.1, max_strength=0.5):
        self.initial_strength = initial_strength
        self.max_strength = max_strength
        self.current_epoch = 0

    def __call__(self, img, epoch=None):
        """Apply adaptive augmentation"""
        if epoch is not None:
            self.current_epoch = epoch

        # Calculate current strength
        strength = min(
            self.max_strength, self.initial_strength + self.current_epoch * 0.01
        )

        # Apply augmentation with current strength
        return self._apply_augmentation(img, strength)

    def _apply_augmentation(self, img, strength):
        """Apply augmentation with given strength"""
        import albumentations as A

        # Create augmentation pipeline based on strength
        transform = A.Compose(
            [
                A.ColorJitter(brightness=strength, contrast=strength, p=0.5),
                A.MotionBlur(blur_limit=int(3 * strength), p=0.3),
                A.RandomBrightnessContrast(brightness_limit=strength, p=0.3),
                A.Perspective(scale=(0.01, 0.05 * strength), p=0.2),
            ]
        )

        img_array = np.array(img)
        transformed = transform(image=img_array)
        return Image.fromarray(transformed["image"])


class PerformanceMonitor:
    """
    Monitor training performance and resource usage
    """

    def __init__(self):
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "gpu_memory": [],
            "training_time": [],
            "validation_time": [],
        }

    def log_metrics(self, **kwargs):
        """Log various metrics"""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)

    def get_gpu_memory_usage(self):
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3  # GB
        return 0

    def print_summary(self):
        """Print performance summary"""
        print("=== Performance Summary ===")
        for key, values in self.metrics.items():
            if values:
                print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

        print(f"GPU Memory: {self.get_gpu_memory_usage():.2f} GB")


def auto_configure_model(config, gpu_memory_gb=None):
    """
    Auto-configure model based on available GPU memory
    """
    if gpu_memory_gb is None and torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    # Adjust configuration based on GPU memory
    if gpu_memory_gb < 8:
        # Low memory GPU (4-8GB)
        config["trainer"]["batch_size"] = 16
        config["transformer"]["d_model"] = 128
        config["transformer"]["num_encoder_layers"] = 4
        config["transformer"]["num_decoder_layers"] = 4
        config["use_gradient_checkpointing"] = True
        config["use_mixed_precision"] = True

    elif gpu_memory_gb < 16:
        # Medium memory GPU (8-16GB)
        config["trainer"]["batch_size"] = 32
        config["transformer"]["d_model"] = 256
        config["transformer"]["num_encoder_layers"] = 6
        config["transformer"]["num_decoder_layers"] = 6
        config["use_gradient_checkpointing"] = False
        config["use_mixed_precision"] = True

    else:
        # High memory GPU (16GB+)
        config["trainer"]["batch_size"] = 64
        config["transformer"]["d_model"] = 512
        config["transformer"]["num_encoder_layers"] = 8
        config["transformer"]["num_decoder_layers"] = 8
        config["use_gradient_checkpointing"] = False
        config["use_mixed_precision"] = False

    return config


# Utility functions
def optimize_memory():
    """Optimize memory usage"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def benchmark_model(model, input_size=(1, 3, 32, 512), num_runs=100):
    """Benchmark model performance"""
    device = next(model.parameters()).device
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(input_size).to(device)
    dummy_tgt = torch.randint(0, 100, (input_size[0], 50)).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input, dummy_tgt)

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input, dummy_tgt)

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    fps = 1.0 / avg_time

    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")

    return avg_time, fps
