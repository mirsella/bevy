use bevy::{app::AppExit, prelude::*};

#[derive(Resource, Clone, Copy)]
pub struct FrameBenchConfig {
    pub warmup_frames: u32,
    pub sample_frames: u32,
    pub print_each_frame: bool,
}

impl FrameBenchConfig {
    pub const fn new(warmup_frames: u32, sample_frames: u32) -> Self {
        Self {
            warmup_frames,
            sample_frames,
            print_each_frame: false,
        }
    }
}

pub struct FrameBenchPlugin {
    pub config: FrameBenchConfig,
}

impl Plugin for FrameBenchPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.config)
            .init_resource::<FrameBenchState>()
            .add_systems(Update, frame_bench_system);
    }
}

#[derive(Resource, Default)]
struct FrameBenchState {
    frame: u32,
    samples_ms: Vec<f64>,
}

fn frame_bench_system(
    time: Res<Time>,
    config: Res<FrameBenchConfig>,
    mut state: ResMut<FrameBenchState>,
    mut app_exit: MessageWriter<AppExit>,
) {
    state.frame = state.frame.saturating_add(1);

    // We measure wall-clock frame time via Time's delta.
    let dt_ms = time.delta().as_secs_f64() * 1000.0;

    if state.frame <= config.warmup_frames {
        if state.frame == config.warmup_frames {
            state.samples_ms.reserve(config.sample_frames as usize);
        }
        return;
    }

    let sample_index = state.frame - config.warmup_frames;
    if sample_index > config.sample_frames {
        return;
    }

    state.samples_ms.push(dt_ms);

    if config.print_each_frame {
        println!("FRAME_BENCH_FRAME {sample_index} {dt_ms:.6}");
    }

    if sample_index == config.sample_frames {
        if let Some(stats) = FrameBenchStats::from_samples_ms(&state.samples_ms) {
            println!(
                "FRAME_BENCH_RESULT frames={} warmup={} mean_ms={:.6} median_ms={:.6} stddev_ms={:.6} min_ms={:.6} p25_ms={:.6} p75_ms={:.6} p90_ms={:.6} p95_ms={:.6} p99_ms={:.6} max_ms={:.6} mean_fps={:.3} median_fps={:.3}",
                config.sample_frames,
                config.warmup_frames,
                stats.mean_ms,
                stats.median_ms,
                stats.stddev_ms,
                stats.min_ms,
                stats.p25_ms,
                stats.p75_ms,
                stats.p90_ms,
                stats.p95_ms,
                stats.p99_ms,
                stats.max_ms,
                stats.mean_fps,
                stats.median_fps,
            );
        } else {
            println!(
                "FRAME_BENCH_RESULT frames=0 warmup={} mean_ms=nan median_ms=nan stddev_ms=nan min_ms=nan p25_ms=nan p75_ms=nan p90_ms=nan p95_ms=nan p99_ms=nan max_ms=nan mean_fps=nan median_fps=nan",
                config.warmup_frames
            );
        }

        app_exit.write(AppExit::Success);
    }
}

#[derive(Debug, Clone, Copy)]
struct FrameBenchStats {
    mean_ms: f64,
    median_ms: f64,
    stddev_ms: f64,
    min_ms: f64,
    max_ms: f64,
    p25_ms: f64,
    p75_ms: f64,
    p90_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    mean_fps: f64,
    median_fps: f64,
}

impl FrameBenchStats {
    fn from_samples_ms(samples_ms: &[f64]) -> Option<Self> {
        let n = samples_ms.len();
        if n == 0 {
            return None;
        }

        let mut sorted = samples_ms.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));

        let sum: f64 = samples_ms.iter().sum();
        let mean_ms = sum / n as f64;

        let median_ms = if n % 2 == 1 {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) * 0.5
        };

        let min_ms = *sorted.first().unwrap();
        let max_ms = *sorted.last().unwrap();

        let stddev_ms = if n > 1 {
            let var = samples_ms
                .iter()
                .map(|x| {
                    let d = x - mean_ms;
                    d * d
                })
                .sum::<f64>()
                / (n as f64 - 1.0);
            var.sqrt()
        } else {
            0.0
        };

        let p25_ms = percentile_nearest(&sorted, 25.0);
        let p75_ms = percentile_nearest(&sorted, 75.0);
        let p90_ms = percentile_nearest(&sorted, 90.0);
        let p95_ms = percentile_nearest(&sorted, 95.0);
        let p99_ms = percentile_nearest(&sorted, 99.0);

        let mean_fps = if mean_ms > 0.0 { 1000.0 / mean_ms } else { 0.0 };
        let median_fps = if median_ms > 0.0 {
            1000.0 / median_ms
        } else {
            0.0
        };

        Some(Self {
            mean_ms,
            median_ms,
            stddev_ms,
            min_ms,
            max_ms,
            p25_ms,
            p75_ms,
            p90_ms,
            p95_ms,
            p99_ms,
            mean_fps,
            median_fps,
        })
    }
}

fn percentile_nearest(sorted: &[f64], p: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    debug_assert!((0.0..=100.0).contains(&p));

    if sorted.len() == 1 {
        return sorted[0];
    }

    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
