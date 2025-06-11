#!/usr/bin/env python3
"""
Demo script để test các tối ưu hóa VietOCR
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time

import numpy as np
import torch

from vietocr.tool.config import Cfg
from vietocr_optimizations import (
    AdaptiveAugmentation,
    DynamicBatching,
    MemoryEfficientAttention,
    OptimizedBeamSearch,
    OptimizedTrainer,
    PerformanceMonitor,
    auto_configure_model,
    benchmark_model,
    optimize_memory,
)


def test_auto_configuration():
    """Test auto-configuration based on GPU memory"""
    print("=== Testing Auto Configuration ===")

    config = Cfg.load_config_from_name("vgg_transformer")
    print(f"Original batch size: {config['trainer']['batch_size']}")
    print(f"Original d_model: {config['transformer']['d_model']}")

    # Auto-configure
    optimized_config = auto_configure_model(config)
    print(f"Optimized batch size: {optimized_config['trainer']['batch_size']}")
    print(f"Optimized d_model: {optimized_config['transformer']['d_model']}")

    return optimized_config


def test_memory_efficient_attention():
    """Test memory efficient attention"""
    print("\n=== Testing Memory Efficient Attention ===")

    d_model = 256
    nhead = 8
    batch_size = 4
    seq_len = 128

    # Create attention module
    attention = MemoryEfficientAttention(d_model, nhead)

    # Create dummy inputs
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)

    # Test forward pass
    start_time = time.time()
    output = attention(q, k, v)
    end_time = time.time()

    print(f"Memory efficient attention time: {(end_time - start_time)*1000:.2f} ms")
    print(f"Output shape: {output.shape}")

    return output


def test_optimized_beam_search():
    """Test optimized beam search"""
    print("\n=== Testing Optimized Beam Search ===")

    # Load model
    config = Cfg.load_config_from_name("vgg_transformer")
    from vietocr.tool.translate import build_model

    model, vocab = build_model(config)

    # Create dummy memory
    memory = torch.randn(64, 1, 256)  # (seq_len, batch_size, d_model)

    # Test beam search
    beam_search = OptimizedBeamSearch(beam_size=4, cache_size=100)

    start_time = time.time()
    result = beam_search.search(memory, model, max_length=50)
    end_time = time.time()

    print(f"Optimized beam search time: {(end_time - start_time)*1000:.2f} ms")
    print(f"Result length: {len(result)}")

    return result


def test_dynamic_batching():
    """Test dynamic batching"""
    print("\n=== Testing Dynamic Batching ===")

    # Create dummy samples
    samples = []
    for i in range(100):
        samples.append(
            {
                "word": [1, 2, 3, 4, 5] * (i % 10 + 1),  # Variable length
                "img": torch.randn(3, 32, 100 + i * 10),
                "img_path": f"sample_{i}.jpg",
            }
        )

    # Test dynamic batching
    batcher = DynamicBatching(max_batch_size=16, max_tokens=1024)
    batches = batcher.create_batches(samples)

    print(f"Created {len(batches)} batches")
    for i, batch in enumerate(batches[:3]):  # Show first 3 batches
        print(f"Batch {i}: {len(batch)} samples")

    return batches


def test_adaptive_augmentation():
    """Test adaptive augmentation"""
    print("\n=== Testing Adaptive Augmentation ===")

    # Create dummy image
    from PIL import Image

    img = Image.new("RGB", (100, 32), color="white")

    # Test adaptive augmentation
    aug = AdaptiveAugmentation(initial_strength=0.1, max_strength=0.5)

    start_time = time.time()
    augmented_img = aug(img, epoch=50)
    end_time = time.time()

    print(f"Adaptive augmentation time: {(end_time - start_time)*1000:.2f} ms")
    print(f"Augmented image size: {augmented_img.size}")

    return augmented_img


def test_performance_monitor():
    """Test performance monitoring"""
    print("\n=== Testing Performance Monitor ===")

    monitor = PerformanceMonitor()

    # Simulate some metrics
    for i in range(10):
        monitor.log_metrics(
            train_loss=1.0 - i * 0.1,
            val_loss=1.2 - i * 0.08,
            learning_rate=0.001 * (0.9**i),
            gpu_memory=(
                torch.cuda.memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0
            ),
            training_time=0.5,
            validation_time=0.1,
        )

    monitor.print_summary()
    return monitor


def test_model_benchmark():
    """Test model benchmarking"""
    print("\n=== Testing Model Benchmark ===")

    # Load model
    config = Cfg.load_config_from_name("vgg_transformer")
    from vietocr.tool.translate import build_model

    model, vocab = build_model(config)

    if torch.cuda.is_available():
        model = model.cuda()

    # Benchmark model
    avg_time, fps = benchmark_model(model, input_size=(1, 3, 32, 512), num_runs=50)

    print(f"Benchmark results:")
    print(f"  Average time: {avg_time*1000:.2f} ms")
    print(f"  Throughput: {fps:.2f} FPS")

    return avg_time, fps


def test_memory_optimization():
    """Test memory optimization"""
    print("\n=== Testing Memory Optimization ===")

    if torch.cuda.is_available():
        print(
            f"GPU memory before optimization: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )
        optimize_memory()
        print(
            f"GPU memory after optimization: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )
    else:
        print("CUDA not available, skipping memory optimization test")


def compare_performance():
    """Compare performance between original and optimized versions"""
    print("\n=== Performance Comparison ===")

    # Load original config
    original_config = Cfg.load_config_from_name("vgg_transformer")

    # Load optimized config
    optimized_config = auto_configure_model(original_config)

    print("Configuration comparison:")
    print(
        f"  Batch size: {original_config['trainer']['batch_size']} -> {optimized_config['trainer']['batch_size']}"
    )
    print(
        f"  d_model: {original_config['transformer']['d_model']} -> {optimized_config['transformer']['d_model']}"
    )
    print(f"  Mixed precision: {optimized_config.get('use_mixed_precision', False)}")
    print(
        f"  Gradient checkpointing: {optimized_config.get('use_gradient_checkpointing', False)}"
    )

    # Estimate performance improvements
    batch_size_improvement = (
        optimized_config["trainer"]["batch_size"]
        / original_config["trainer"]["batch_size"]
    )
    d_model_improvement = (
        optimized_config["transformer"]["d_model"]
        / original_config["transformer"]["d_model"]
    )

    print(f"\nEstimated improvements:")
    print(f"  Batch size improvement: {batch_size_improvement:.1f}x")
    print(f"  Model capacity: {d_model_improvement:.1f}x")
    print(
        f"  Expected training speedup: {batch_size_improvement * 0.7:.1f}x (with mixed precision)"
    )
    print(
        f"  Expected memory reduction: {1/batch_size_improvement * 0.8:.1f}x (with optimizations)"
    )


def main():
    """Run all optimization tests"""
    print("🚀 VietOCR Optimization Tests")
    print("=" * 50)

    try:
        # Test auto-configuration
        optimized_config = test_auto_configuration()

        # Test memory efficient attention
        test_memory_efficient_attention()

        # Test optimized beam search
        test_optimized_beam_search()

        # Test dynamic batching
        test_dynamic_batching()

        # Test adaptive augmentation
        test_adaptive_augmentation()

        # Test performance monitoring
        test_performance_monitor()

        # Test model benchmarking
        test_model_benchmark()

        # Test memory optimization
        test_memory_optimization()

        # Compare performance
        compare_performance()

        print("\n✅ All optimization tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
