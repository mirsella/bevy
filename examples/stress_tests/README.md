# Stress Tests

These examples are used to stress test Bevy's performance in various ways. These
should be run with the "stress-test" profile to accurately represent performance
in production, otherwise they will run in cargo's default "dev" profile which is
very slow.

## Example Command

```bash
cargo run --profile stress-test --example <EXAMPLE>
```

## Benchmarking Tools

The stress test examples now include a frame benchmarking system for automated
performance measurement. This allows precise A/B testing of performance changes
and regression detection.

### Frame Benchmark Utilities (`frame_bench_util.rs`)

A reusable benchmarking plugin that collects frame timing statistics and exits
with a structured report.

#### Usage

Add the frame benchmark plugin to any stress test example:

```rust
mod frame_bench_util;

fn main() {
    let mut app = App::new();
    // ... setup plugins ...
    
    app.add_plugins(frame_bench_util::FrameBenchPlugin {
        config: frame_bench_util::FrameBenchConfig {
            warmup_frames: 120,  // Frames to skip before measuring
            sample_frames: 600,  // Frames to measure
            print_each_frame: false,
        },
    });
    
    app.run();
}
```

#### Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `warmup_frames` | Number of initial frames to discard (allows GPU warmup) | 120 |
| `sample_frames` | Number of frames to measure | 600 |
| `print_each_frame` | Print per-frame timing to stdout | false |

#### Output Format

When benchmarking completes, the plugin prints a single `FRAME_BENCH_RESULT` line:

```
FRAME_BENCH_RESULT frames=600 warmup=120 mean_ms=2.086198 median_ms=2.069466 stddev_ms=0.171368 min_ms=1.620614 p25_ms=1.969946 p75_ms=2.178507 p90_ms=2.320207 p95_ms=2.395197 p99_ms=2.599387 max_ms=2.789829 mean_fps=479.341 median_fps=483.216
```

Statistics provided:
- **Mean/Median**: Central tendency of frame times
- **Stddev**: Variability in frame times
- **Min/Max**: Range extremes
- **Percentiles (p25, p75, p90, p95, p99)**: Distribution analysis
- **Mean/Median FPS**: Frames per second calculated from frame times

#### Running Benchmarks from Command Line

Examples that support benchmarking accept these arguments:

```bash
# Enable benchmark mode
cargo run --release --example many_sprites -- --bench

# Customize warmup and sample frames
cargo run --release --example many_sprites -- --bench --bench-warmup 240 --bench-frames 600

# Print per-frame timings (useful for debugging)
cargo run --release --example many_sprites -- --bench --bench-print-each-frame
```

### UI Clip Benchmark (`ui_clip_bench.rs`)

A specialized benchmark for testing UI overflow clipping performance. This creates
a grid of containers with `Overflow::clip()` to stress-test the scissor-based
clipping system.

#### Usage

```bash
# Run in benchmark mode with default settings
cargo run --release --example ui_clip_bench -- --bench

# Stress test with many unique clip rects
cargo run --release --example ui_clip_bench -- \
  --bench \
  --containers-x 80 \
  --containers-y 45 \
  --images-per-container 1 \
  --texture-count 1

# Test without clipping (baseline comparison)
cargo run --release --example ui_clip_bench -- \
  --bench \
  --containers-x 80 \
  --containers-y 45 \
  --images-per-container 1 \
  --texture-count 1 \
  --no-clip

# Stress test with many textures per clip rect
cargo run --release --example ui_clip_bench -- \
  --bench \
  --containers-x 40 \
  --containers-y 22 \
  --images-per-container 8 \
  --texture-count 8
```

#### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--containers-x` | Number of container columns | 40 |
| `--containers-y` | Number of container rows | 20 |
| `--images-per-container` | Images inside each container | 1 |
| `--texture-count` | Unique textures to create | 1 |
| `--image-scale` | Image size relative to container | 1.5 |
| `--no-clip` | Disable clipping (baseline mode) | false |

#### Benchmarking Workflow for A/B Testing

To compare performance across commits:

```bash
# Build the benchmark
cargo build --release --example ui_clip_bench

# Run at current HEAD
RUST_LOG=error target/release/examples/ui_clip_bench \
  --bench --bench-warmup 240 --bench-frames 600 \
  --containers-x 80 --containers-y 45 \
  --images-per-container 1 --texture-count 1

# Checkout comparison commit
git checkout <COMMIT_SHA>
cargo build --release --example ui_clip_bench

# Run again with identical parameters
RUST_LOG=error target/release/examples/ui_clip_bench \
  --bench --bench-warmup 240 --bench-frames 600 \
  --containers-x 80 --containers-y 45 \
  --images-per-container 1 --texture-count 1

# Parse the FRAME_BENCH_RESULT lines to compare
```

### Available Stress Tests

- **`many_sprites`** - Large numbers of sprites with optional color variation
- **`many_buttons`** - UI button stress test with various layouts
- **`ui_clip_bench`** - UI overflow clipping performance test

### Best Practices

1. **Use Release Profile**: Always run benchmarks with `--release` or `--profile stress-test`
2. **Disable VSync**: Benchmarks use `PresentMode::AutoNoVsync` to measure raw throughput
3. **Warmup Period**: Allow at least 120 warmup frames for GPU shader compilation and cache warmup
4. **Multiple Runs**: Run each benchmark 3-5 times and use the median result for accuracy
5. **Consistent Environment**: Close other applications and maintain consistent system state
6. **Log Level**: Use `RUST_LOG=error` to suppress non-essential output during benchmarks

### Interpreting Results

- **Mean vs Median**: Median is more robust to outliers (occasional stutters)
- **Stddev**: Lower is better; high stddev indicates inconsistent performance
- **Percentiles**: p95 and p99 show worst-case frame times that affect perceived smoothness
- **FPS**: Higher is better, but consistency (low stddev) is often more important than peak FPS
