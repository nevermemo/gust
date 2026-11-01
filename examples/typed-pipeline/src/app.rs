use std::{path::PathBuf, time::Instant};

#[cfg(test)]
use gust::GpuPod;
use gust::{KernelDescriptor, gpu};
use gust_wgpu::{
    BufferDispatch, GpuBufferAccess, HeadlessDevice, PipelineCacheStats, render_wgpu_source,
};

#[gpu]
mod sensor_pipeline {
    #[derive(Debug, PartialEq)]
    pub struct SensorReading {
        pub raw_temperature: float,
        pub raw_humidity: float,
        pub sensor_id: uint,
        pub status: int,
    }

    #[derive(Debug, PartialEq)]
    pub struct Calibration {
        pub temperature_scale: float,
        pub temperature_bias: float,
        pub humidity_scale: float,
        pub humidity_bias: float,
    }

    #[derive(Debug, PartialEq)]
    pub struct CalibratedSample {
        pub temperature: float,
        pub humidity: float,
        pub quality: float,
        pub sensor_id: uint,
    }

    #[derive(Debug, PartialEq)]
    pub struct AlertThresholds {
        pub target_temperature: float,
        pub target_humidity: float,
        pub temperature_weight: float,
        pub humidity_weight: float,
        pub warning: float,
        pub critical: float,
    }

    #[derive(Debug, PartialEq)]
    pub struct Alert {
        pub score: float,
        pub quality: float,
        pub sensor_id: uint,
        pub severity: int,
    }

    #[derive(Debug, PartialEq)]
    pub struct Report {
        pub adjusted_score: float,
        pub confidence: float,
        pub sensor_id: uint,
        pub action: int,
    }

    fn clamp_value(value: float, low: float, high: float) -> float {
        let mut result = value;
        if result < low {
            result = low;
        }
        if result > high {
            result = high;
        }
        result
    }

    fn sample_quality(humidity: float) -> float {
        let centered = humidity - 0.5;
        clamp_value(1.0 - centered * centered, 0.0, 1.0)
    }

    fn classification_score(sample: CalibratedSample, thresholds: AlertThresholds) -> float {
        let temperature_delta = sample.temperature - thresholds.target_temperature;
        let humidity_delta = sample.humidity - thresholds.target_humidity;
        let temperature_score =
            temperature_delta * temperature_delta * thresholds.temperature_weight;
        let humidity_score = humidity_delta * humidity_delta * thresholds.humidity_weight;
        temperature_score + humidity_score
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn calibrate(
        id: SV_DispatchThreadID,
        readings: StructuredBuffer<SensorReading>,
        calibration: StructuredBuffer<Calibration>,
        mut out: RWStructuredBuffer<CalibratedSample>,
    ) {
        let i = id.x;
        if i < out.len() {
            let reading = readings[i];
            let parameters = calibration[i];
            let temperature = reading.raw_temperature * parameters.temperature_scale
                + parameters.temperature_bias;
            let humidity = clamp_value(
                reading.raw_humidity * parameters.humidity_scale + parameters.humidity_bias,
                0.0,
                1.0,
            );
            out[i].temperature = temperature;
            out[i].humidity = humidity;
            out[i].quality = sample_quality(humidity);
            out[i].sensor_id = reading.sensor_id;
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn classify(
        id: SV_DispatchThreadID,
        samples: StructuredBuffer<CalibratedSample>,
        thresholds: StructuredBuffer<AlertThresholds>,
        mut out: RWStructuredBuffer<Alert>,
    ) {
        let i = id.x;
        if i < out.len() {
            let sample = samples[i];
            let limits = thresholds[i];
            let score = classification_score(sample, limits);
            out[i].score = score;
            out[i].quality = sample.quality;
            out[i].sensor_id = sample.sensor_id;
            if score > limits.critical {
                out[i].severity = 2;
            } else if score > limits.warning {
                out[i].severity = 1;
            } else {
                out[i].severity = 0;
            }
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn finalize(
        id: SV_DispatchThreadID,
        alerts: StructuredBuffer<Alert>,
        mut out: RWStructuredBuffer<Report>,
    ) {
        let i = id.x;
        if i < out.len() {
            let alert = alerts[i];
            let quality_penalty = (1.0 - alert.quality) * 0.5;
            out[i].adjusted_score = alert.score + quality_penalty;
            out[i].confidence = alert.quality;
            out[i].sensor_id = alert.sensor_id;
            if alert.severity > 0 && alert.quality < 0.25 {
                out[i].action = -1;
            } else {
                out[i].action = alert.severity;
            }
        }
    }
}

struct Inputs {
    readings: Vec<sensor_pipeline::SensorReading>,
    calibration: Vec<sensor_pipeline::Calibration>,
    thresholds: Vec<sensor_pipeline::AlertThresholds>,
}

struct PipelineOutput {
    samples: Vec<sensor_pipeline::CalibratedSample>,
    alerts: Vec<sensor_pipeline::Alert>,
    reports: Vec<sensor_pipeline::Report>,
    cache: PipelineCacheStats,
}

fn make_inputs(count: usize) -> Inputs {
    let indices = 0..count;
    Inputs {
        readings: indices
            .clone()
            .map(|index| sensor_pipeline::SensorReading {
                raw_temperature: 18.0 + (index % 29) as f32 * 0.35,
                raw_humidity: 0.2 + (index % 41) as f32 * 0.015,
                sensor_id: 10_000 + index as u32,
                status: (index % 5) as i32 - 2,
            })
            .collect(),
        calibration: indices
            .clone()
            .map(|index| sensor_pipeline::Calibration {
                temperature_scale: 0.98 + (index % 3) as f32 * 0.01,
                temperature_bias: (index % 7) as f32 * 0.05 - 0.15,
                humidity_scale: 0.95 + (index % 5) as f32 * 0.02,
                humidity_bias: (index % 11) as f32 * 0.005 - 0.025,
            })
            .collect(),
        thresholds: indices
            .map(|index| sensor_pipeline::AlertThresholds {
                target_temperature: 22.0,
                target_humidity: 0.5,
                temperature_weight: 0.08 + (index % 2) as f32 * 0.01,
                humidity_weight: 2.0,
                warning: 0.8,
                critical: 2.4,
            })
            .collect(),
    }
}

fn clamp_value(value: f32, low: f32, high: f32) -> f32 {
    value.max(low).min(high)
}

fn run_cpu(inputs: &Inputs) -> PipelineOutput {
    let mut samples = Vec::with_capacity(inputs.readings.len());
    let mut alerts = Vec::with_capacity(inputs.readings.len());
    let mut reports = Vec::with_capacity(inputs.readings.len());

    for ((reading, calibration), thresholds) in inputs
        .readings
        .iter()
        .zip(&inputs.calibration)
        .zip(&inputs.thresholds)
    {
        let temperature =
            reading.raw_temperature * calibration.temperature_scale + calibration.temperature_bias;
        let humidity = clamp_value(
            reading.raw_humidity * calibration.humidity_scale + calibration.humidity_bias,
            0.0,
            1.0,
        );
        let centered = humidity - 0.5;
        let quality = clamp_value(1.0 - centered * centered, 0.0, 1.0);
        let sample = sensor_pipeline::CalibratedSample {
            temperature,
            humidity,
            quality,
            sensor_id: reading.sensor_id,
        };

        let temperature_delta = sample.temperature - thresholds.target_temperature;
        let humidity_delta = sample.humidity - thresholds.target_humidity;
        let score = temperature_delta * temperature_delta * thresholds.temperature_weight
            + humidity_delta * humidity_delta * thresholds.humidity_weight;
        let severity = if score > thresholds.critical {
            2
        } else if score > thresholds.warning {
            1
        } else {
            0
        };
        let alert = sensor_pipeline::Alert {
            score,
            quality: sample.quality,
            sensor_id: sample.sensor_id,
            severity,
        };
        let report = sensor_pipeline::Report {
            adjusted_score: alert.score + (1.0 - alert.quality) * 0.5,
            confidence: alert.quality,
            sensor_id: alert.sensor_id,
            action: if alert.severity > 0 && alert.quality < 0.25 {
                -1
            } else {
                alert.severity
            },
        };
        samples.push(sample);
        alerts.push(alert);
        reports.push(report);
    }

    PipelineOutput {
        samples,
        alerts,
        reports,
        cache: PipelineCacheStats::default(),
    }
}

fn run_gpu(device: &HeadlessDevice, inputs: &Inputs) -> Result<PipelineOutput, gust_wgpu::Error> {
    let count = inputs.readings.len();
    let readings = device.create_typed_buffer(
        "sensor readings",
        &inputs.readings,
        GpuBufferAccess::ReadOnly,
    );
    let calibration = device.create_typed_buffer(
        "sensor calibration",
        &inputs.calibration,
        GpuBufferAccess::ReadOnly,
    );
    let thresholds = device.create_typed_buffer(
        "alert thresholds",
        &inputs.thresholds,
        GpuBufferAccess::ReadOnly,
    );
    let sample_seed = vec![empty_sample(); count];
    let alert_seed = vec![empty_alert(); count];
    let report_seed = vec![empty_report(); count];
    let samples = device.create_typed_buffer(
        "calibrated samples",
        &sample_seed,
        GpuBufferAccess::ReadWrite,
    );
    let alerts = device.create_typed_buffer("alerts", &alert_seed, GpuBufferAccess::ReadWrite);
    let reports = device.create_typed_buffer("reports", &report_seed, GpuBufferAccess::ReadWrite);

    // Typed calls are locals: the generated `call` checks every buffer's element type and device at compile
    // time, and the batch entries borrow the calls.
    let count = count as u32;
    let calibrate = sensor_pipeline::calibrate::call(count, &readings, &calibration, &samples);
    let classify = sensor_pipeline::classify::call(count, &samples, &thresholds, &alerts);
    let finalize = sensor_pipeline::finalize::call(count, &alerts, &reports);
    let dispatches = [
        BufferDispatch::from(&calibrate),
        BufferDispatch::from(&classify),
        BufferDispatch::from(&finalize),
    ];
    device.dispatch_batch(&dispatches)?;

    Ok(PipelineOutput {
        samples: device.read_typed_buffer(&samples)?,
        alerts: device.read_typed_buffer(&alerts)?,
        reports: device.read_typed_buffer(&reports)?,
        cache: device.pipeline_cache_stats(),
    })
}

fn empty_sample() -> sensor_pipeline::CalibratedSample {
    sensor_pipeline::CalibratedSample {
        temperature: 0.0,
        humidity: 0.0,
        quality: 0.0,
        sensor_id: 0,
    }
}

fn empty_alert() -> sensor_pipeline::Alert {
    sensor_pipeline::Alert {
        score: 0.0,
        quality: 0.0,
        sensor_id: 0,
        severity: 0,
    }
}

fn empty_report() -> sensor_pipeline::Report {
    sensor_pipeline::Report {
        adjusted_score: 0.0,
        confidence: 0.0,
        sensor_id: 0,
        action: 0,
    }
}

fn assert_close(actual: &PipelineOutput, expected: &PipelineOutput) {
    assert_eq!(actual.samples.len(), expected.samples.len());
    assert_eq!(actual.alerts.len(), expected.alerts.len());
    assert_eq!(actual.reports.len(), expected.reports.len());
    for (actual, expected) in actual.samples.iter().zip(&expected.samples) {
        assert_float(actual.temperature, expected.temperature);
        assert_float(actual.humidity, expected.humidity);
        assert_float(actual.quality, expected.quality);
        assert_eq!(actual.sensor_id, expected.sensor_id);
    }
    for (actual, expected) in actual.alerts.iter().zip(&expected.alerts) {
        assert_float(actual.score, expected.score);
        assert_float(actual.quality, expected.quality);
        assert_eq!(actual.sensor_id, expected.sensor_id);
        assert_eq!(actual.severity, expected.severity);
    }
    for (actual, expected) in actual.reports.iter().zip(&expected.reports) {
        assert_float(actual.adjusted_score, expected.adjusted_score);
        assert_float(actual.confidence, expected.confidence);
        assert_eq!(actual.sensor_id, expected.sensor_id);
        assert_eq!(actual.action, expected.action);
    }
}

fn assert_float(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= 2.0e-5 * expected.abs().max(1.0),
        "GPU value {actual} differs from CPU value {expected}"
    );
}

fn descriptors() -> [&'static KernelDescriptor; 3] {
    [
        &sensor_pipeline::calibrate::DESCRIPTOR,
        &sensor_pipeline::classify::DESCRIPTOR,
        &sensor_pipeline::finalize::DESCRIPTOR,
    ]
}

fn export_artifacts() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../generated-wgpu");
    std::fs::create_dir_all(&directory)?;
    for descriptor in descriptors() {
        let stem = format!("{}__{}", descriptor.module, descriptor.name);
        std::fs::write(
            directory.join(format!("{stem}.slang")),
            descriptor.slang_source,
        )?;
        std::fs::write(
            directory.join(format!("{stem}.wgsl")),
            gust::slang::compile_wgsl(descriptor)?,
        )?;
        std::fs::write(
            directory.join(format!("{stem}.spv")),
            gust::spirv::words_as_le_bytes(&gust::slang::compile_spirv(descriptor)?),
        )?;
        std::fs::write(
            directory.join(format!("{stem}.rs")),
            render_wgpu_source(descriptor)?,
        )?;
    }
    Ok(directory.canonicalize().unwrap_or(directory))
}

/// Element count of a debug run.
const DEBUG_COUNT: usize = 1 << 18;
/// Element count of a release run. Every stage dispatches `count / workgroup_size`
/// groups, which must stay within the default per-dimension workgroup limit of
/// 65 535; 2^21 elements need 32 768 groups of 64.
const RELEASE_COUNT: usize = 1 << 21;

fn demo_count() -> usize {
    if cfg!(debug_assertions) {
        DEBUG_COUNT
    } else {
        RELEASE_COUNT
    }
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let count = demo_count();
    let inputs = make_inputs(count);
    let cpu_started = Instant::now();
    let expected = run_cpu(&inputs);
    let cpu_elapsed = cpu_started.elapsed();

    let device = HeadlessDevice::new()?;
    let gpu_started = Instant::now();
    let actual = run_gpu(&device, &inputs)?;
    let gpu_elapsed = gpu_started.elapsed();
    assert_close(&actual, &expected);
    let artifacts = export_artifacts()?;

    println!("Typed sensor pipeline: {count} elements");
    println!("adapter: {}", device.adapter_info().name);
    println!("shader entry points: {}", descriptors().len());
    println!("ordered dispatches per batch: 3");
    println!("distinct GPU struct types: 6");
    println!("CPU reference: {cpu_elapsed:?}");
    println!("GPU end-to-end: {gpu_elapsed:?}");
    println!("pipeline cache: {:?}", actual.cache);
    println!("generated artifacts: {}", artifacts.display());
    println!("first report: {:?}", actual.reports.first());
    Ok(())
}

#[cfg(test)]
mod tests {
    use gust::cpu::{CpuDevice, Error as CpuError};

    use super::*;

    /// `2^-24`.
    const U: f64 = 1.0 / 16_777_216.0;

    /// The twin-versus-GPU bounds of the float outputs, absolute, in units of `U`, one per output and kernel. Each
    /// kernel is compared on identical inputs (the twin is fed what the GPU read back from the stage before), so a
    /// bound covers that kernel's own roundings only.
    ///
    /// Derived from the kernel text (docs/SEMANTICS.md section 4, "Deriving a tolerance"), not from a measurement.
    /// With the domain asserted by `assert_derivation_domain` and `assert_stage_domains`, one rounding of at most
    /// `u` times the result's magnitude on the twin and `2u` on the GPU for `+ - *`, first-order propagation,
    /// magnitudes `M`, twin error `T` and GPU error `G` in units of `u`, the GPU side without fusion:
    ///
    /// `calibrate` (`raw_temperature <= 28`, `temperature_scale <= 1.01`, `abs(temperature_bias) <= 0.16`,
    /// `raw_humidity <= 0.81`, `humidity_scale <= 1.04`, `abs(humidity_bias) <= 0.026`; `clamp_value` never changes
    /// a value here and is continuous, so it adds no error):
    ///
    /// | step | M | T | G |
    /// | --- | --- | --- | --- |
    /// | `temperature` = `raw * scale + bias` | 28.44 | 56.72 | 113.44 |
    /// | `humidity` = `clamp(raw * scale + bias)` | 0.87 | 1.71 | 3.42 |
    /// | `centered` = `humidity - 0.5`, `abs <= 0.37` | 0.37 | 2.08 | 4.16 |
    /// | `centered * centered` | 0.14 | 1.68 | 3.35 |
    /// | `quality` = `1.0 - centered * centered` | 1.00 | 2.68 | 5.35 |
    ///
    /// Twin plus GPU: `temperature` 170.16u gives 172u, `humidity` 5.13u gives 7u, `quality` 8.03u gives 10u
    /// (rounded up to a whole `u` plus 1u for the second-order terms, below 0.001u here).
    ///
    /// `classify` (`17.4 <= temperature <= 28.5`, `0.16 <= humidity <= 0.87`, `0.0 <= temperature_weight <= 0.091`,
    /// `humidity_weight = 2`; `temperature - 22` is at most 6.5 in magnitude):
    ///
    /// | step | M | T | G |
    /// | --- | --- | --- | --- |
    /// | `temperature_delta * temperature_delta` | 42.25 | 126.75 | 253.50 |
    /// | `... * temperature_weight` | 3.84 | 15.38 | 30.76 |
    /// | `humidity_delta * humidity_delta * 2.0` (the 2.0 is exact) | 0.27 | 0.82 | 1.64 |
    /// | `score` = sum of the two | 4.12 | 20.32 | 40.64 |
    ///
    /// Twin plus GPU: `score` 60.96u gives 62u, plus 1u is 63u. A chain of products has the same relative error in
    /// any grouping; the sum has two operands.
    ///
    /// `finalize` (`score <= 4.12`, `quality >= 0.86`, so `1.0 - quality <= 0.14`): `(1.0 - quality) * 0.5` has
    /// M 0.07, T 0.07 and G 0.14, and `adjusted_score = score + that` has M 4.19, T 4.26 and G 8.52; twin plus GPU
    /// 12.78u gives 13u, plus 1u is 14u.
    ///
    /// The integer fields and `confidence` are copies and must be equal. `severity` and `action` are decisions on
    /// floats: `severity` compares `score` with 0.8 and 2.4 and is asserted equal only after the test has checked
    /// that no score lies within the `score` bound of either threshold (`assert_decisions_are_stable`); `action`
    /// compares the copied `quality` (at least 0.86) with 0.25 and the copied `severity`, so both targets decide on
    /// identical inputs.
    ///
    /// Observed maxima are printed separately by the ignored test and never become the tolerance.
    const TEMPERATURE_BOUND: f64 = 172.0 * U;
    const HUMIDITY_BOUND: f64 = 7.0 * U;
    const QUALITY_BOUND: f64 = 10.0 * U;
    const SCORE_BOUND: f64 = 63.0 * U;
    const ADJUSTED_SCORE_BOUND: f64 = 14.0 * U;

    /// The domain the bounds are derived for. `make_inputs` is periodic with periods of at most 41, so 4096 elements
    /// reach every extreme.
    fn assert_derivation_domain(inputs: &Inputs) {
        for reading in &inputs.readings {
            assert!(
                (18.0..=28.0).contains(&reading.raw_temperature),
                "raw_temperature"
            );
            assert!((0.0..=0.81).contains(&reading.raw_humidity), "raw_humidity");
        }
        for parameters in &inputs.calibration {
            assert!(
                (0.0..=1.01).contains(&parameters.temperature_scale),
                "temperature_scale"
            );
            assert!(
                parameters.temperature_bias.abs() <= 0.16,
                "temperature_bias"
            );
            assert!(
                (0.0..=1.04).contains(&parameters.humidity_scale),
                "humidity_scale"
            );
            assert!(parameters.humidity_bias.abs() <= 0.026, "humidity_bias");
        }
        for limits in &inputs.thresholds {
            assert!(
                (0.0..=0.091).contains(&limits.temperature_weight),
                "temperature_weight"
            );
            assert_eq!(limits.humidity_weight, 2.0);
            assert_eq!(limits.target_temperature, 22.0);
            assert_eq!(limits.target_humidity, 0.5);
        }
    }

    /// The domains of the values that cross from one stage to the next, which the bounds of `classify` and
    /// `finalize` assume. They hold for the twin's and the GPU's samples alike.
    fn assert_stage_domains(samples: &[sensor_pipeline::CalibratedSample], scores: &[f32]) {
        for sample in samples {
            assert!((17.4..=28.5).contains(&sample.temperature), "temperature");
            assert!((0.16..=0.87).contains(&sample.humidity), "humidity");
            assert!((0.86..=1.0).contains(&sample.quality), "quality");
        }
        for score in scores {
            assert!((0.0..=4.12).contains(score), "score");
        }
    }

    /// `severity` is a decision on a float: the targets agree on it unless a score lies within the `score` bound of a
    /// threshold. Checked on the twin's scores, so the GPU's (within the bound) fall on the same side.
    fn assert_decisions_are_stable(
        alerts: &[sensor_pipeline::Alert],
        thresholds: &[sensor_pipeline::AlertThresholds],
    ) {
        assert_eq!(alerts.len(), thresholds.len());
        for (alert, limits) in alerts.iter().zip(thresholds) {
            // The f32 values the kernel compares with, not the decimal literals they were written as.
            for threshold in [f64::from(limits.warning), f64::from(limits.critical)] {
                assert!(
                    (f64::from(alert.score) - threshold).abs() > SCORE_BOUND,
                    "score {} lies within the derived bound of the threshold {threshold}",
                    alert.score
                );
            }
        }
    }

    fn calibrate_on_cpu(
        readings: &[sensor_pipeline::SensorReading],
        calibration: &[sensor_pipeline::Calibration],
    ) -> Result<Vec<sensor_pipeline::CalibratedSample>, CpuError> {
        let device = CpuDevice::new();
        let readings_buffer = device.buffer(readings);
        let calibration_buffer = device.buffer(calibration);
        let out = device.buffer(&vec![empty_sample(); readings.len()]);
        sensor_pipeline::calibrate::dispatch(
            &device,
            readings.len() as u32,
            &readings_buffer,
            &calibration_buffer,
            &out,
        )?;
        Ok(out.read())
    }

    fn classify_on_cpu(
        samples: &[sensor_pipeline::CalibratedSample],
        thresholds: &[sensor_pipeline::AlertThresholds],
    ) -> Result<Vec<sensor_pipeline::Alert>, CpuError> {
        let device = CpuDevice::new();
        let samples_buffer = device.buffer(samples);
        let thresholds_buffer = device.buffer(thresholds);
        let out = device.buffer(&vec![empty_alert(); samples.len()]);
        sensor_pipeline::classify::dispatch(
            &device,
            samples.len() as u32,
            &samples_buffer,
            &thresholds_buffer,
            &out,
        )?;
        Ok(out.read())
    }

    fn finalize_on_cpu(
        alerts: &[sensor_pipeline::Alert],
    ) -> Result<Vec<sensor_pipeline::Report>, CpuError> {
        let device = CpuDevice::new();
        let alerts_buffer = device.buffer(alerts);
        let out = device.buffer(&vec![empty_report(); alerts.len()]);
        sensor_pipeline::finalize::dispatch(&device, alerts.len() as u32, &alerts_buffer, &out)?;
        Ok(out.read())
    }

    #[test]
    fn a_nan_in_a_compared_value_fails_the_bound_checks() {
        // `fold(0.0, f64::max)` returned 0 for a NaN difference, so NaN output passed a bound.
        let finite = [1.0f32, 2.0, 3.0];
        let nan_second = [1.0f32, f32::NAN, 3.0];
        let nan_first = [f32::NAN, 2.0, 3.0];
        assert_eq!(largest_abs(finite.into_iter(), finite.into_iter()), 0.0);
        assert_eq!(
            largest_abs(finite.into_iter(), [1.5f32, 2.0, 0.0].into_iter()),
            3.0
        );
        for (left, right) in [
            (finite, nan_second),
            (nan_second, finite),
            (nan_first, finite),
        ] {
            let largest = largest_abs(left.into_iter(), right.into_iter());
            assert!(largest.is_nan(), "{largest}");
            let passes = largest <= 1.0e9;
            assert!(!passes, "a NaN must not pass the bound");
        }
    }

    /// The largest absolute difference (computed in `f64`) between two float sequences.
    fn largest_abs(left: impl Iterator<Item = f32>, right: impl Iterator<Item = f32>) -> f64 {
        left.zip(right)
            .map(|(l, r)| (f64::from(l) - f64::from(r)).abs())
            .fold(0.0, |max: f64, diff| {
                // `f64::max` would return the other operand for a NaN and turn it into 0; a NaN stays NaN here,
                // so `largest <= bound` fails.
                if max.is_nan() || diff.is_nan() {
                    f64::NAN
                } else {
                    max.max(diff)
                }
            })
    }

    #[test]
    fn inputs_stay_inside_the_derivation_domain() {
        assert_derivation_domain(&make_inputs(4096));
        assert_derivation_domain(&make_inputs(DEBUG_COUNT));
    }

    #[test]
    fn twin_chain_matches_host_reference() {
        // The three twins run the shadow bodies, which perform the operations of `run_cpu` in the same order, and Rust
        // does not contract them, so floats agree to the last bit and integers are equal. 0 and 1 are the edges;
        // 63 to 65 straddle one workgroup of 64.
        for count in [0, 1, 63, 64, 65, 257, 4096] {
            let inputs = make_inputs(count);
            let expected = run_cpu(&inputs);
            let samples =
                calibrate_on_cpu(&inputs.readings, &inputs.calibration).expect("calibrate twin");
            assert_eq!(samples, expected.samples, "calibrate, {count} elements");
            let alerts = classify_on_cpu(&samples, &inputs.thresholds).expect("classify twin");
            assert_eq!(alerts, expected.alerts, "classify, {count} elements");
            let reports = finalize_on_cpu(&alerts).expect("finalize twin");
            assert_eq!(reports, expected.reports, "finalize, {count} elements");
        }
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn twin_matches_reference_and_gpu_within_the_derived_bounds() {
        let device = HeadlessDevice::new().expect("Vulkan adapter");
        for count in [0, 1, 63, 64, 65, 257, 1 << 15] {
            let inputs = make_inputs(count);
            assert_derivation_domain(&inputs);
            let host = run_cpu(&inputs);
            let gpu = run_gpu(&device, &inputs).unwrap();
            // The GPU against the host reference comes first so that a failing twin cannot hide it.
            assert_close(&gpu, &host);

            let samples =
                calibrate_on_cpu(&inputs.readings, &inputs.calibration).expect("calibrate twin");
            assert_eq!(samples, host.samples, "calibrate twin versus run_cpu");
            let temperature = largest_abs(
                samples.iter().map(|s| s.temperature),
                gpu.samples.iter().map(|s| s.temperature),
            );
            let humidity = largest_abs(
                samples.iter().map(|s| s.humidity),
                gpu.samples.iter().map(|s| s.humidity),
            );
            let quality = largest_abs(
                samples.iter().map(|s| s.quality),
                gpu.samples.iter().map(|s| s.quality),
            );
            println!(
                "S4 typed-pipeline calibrate, {count} elements: twin versus GPU max abs error temperature {temperature:e} (bound {TEMPERATURE_BOUND:e}), humidity {humidity:e} (bound {HUMIDITY_BOUND:e}), quality {quality:e} (bound {QUALITY_BOUND:e})"
            );
            assert!(
                temperature <= TEMPERATURE_BOUND,
                "temperature {temperature:e}"
            );
            assert!(humidity <= HUMIDITY_BOUND, "humidity {humidity:e}");
            assert!(quality <= QUALITY_BOUND, "quality {quality:e}");
            assert!(
                samples
                    .iter()
                    .zip(&gpu.samples)
                    .all(|(twin, gpu)| twin.sensor_id == gpu.sensor_id),
                "calibrate sensor_id"
            );

            // From here on each twin is fed what the GPU produced for the stage before: identical inputs.
            assert_stage_domains(&gpu.samples, &[]);
            let alerts = classify_on_cpu(&gpu.samples, &inputs.thresholds).expect("classify twin");
            let scores: Vec<f32> = alerts.iter().map(|alert| alert.score).collect();
            assert_stage_domains(&gpu.samples, &scores);
            let gpu_scores: Vec<f32> = gpu.alerts.iter().map(|alert| alert.score).collect();
            assert_stage_domains(&gpu.samples, &gpu_scores);
            assert_decisions_are_stable(&alerts, &inputs.thresholds);
            assert_eq!(
                alerts.len(),
                gpu.alerts.len(),
                "classify output length, {count} elements"
            );
            let score = largest_abs(
                alerts.iter().map(|a| a.score),
                gpu.alerts.iter().map(|a| a.score),
            );
            println!(
                "S4 typed-pipeline classify, {count} elements: twin versus GPU max abs error score {score:e} (bound {SCORE_BOUND:e})"
            );
            assert!(score <= SCORE_BOUND, "score {score:e}");
            for (twin, gpu) in alerts.iter().zip(&gpu.alerts) {
                assert_eq!(twin.quality, gpu.quality, "classify quality");
                assert_eq!(twin.sensor_id, gpu.sensor_id, "classify sensor_id");
                assert_eq!(twin.severity, gpu.severity, "classify severity");
            }

            let reports = finalize_on_cpu(&gpu.alerts).expect("finalize twin");
            let adjusted = largest_abs(
                reports.iter().map(|r| r.adjusted_score),
                gpu.reports.iter().map(|r| r.adjusted_score),
            );
            println!(
                "S4 typed-pipeline finalize, {count} elements: twin versus GPU max abs error adjusted_score {adjusted:e} (bound {ADJUSTED_SCORE_BOUND:e})"
            );
            assert!(
                adjusted <= ADJUSTED_SCORE_BOUND,
                "adjusted_score {adjusted:e}"
            );
            for (twin, gpu) in reports.iter().zip(&gpu.reports) {
                assert_eq!(twin.confidence, gpu.confidence, "finalize confidence");
                assert_eq!(twin.sensor_id, gpu.sensor_id, "finalize sensor_id");
                assert_eq!(twin.action, gpu.action, "finalize action");
            }
        }
    }

    #[test]
    fn demo_sizes_stay_within_the_default_workgroup_limit() {
        for count in [DEBUG_COUNT, RELEASE_COUNT] {
            for descriptor in descriptors() {
                let groups = u32::try_from(count)
                    .unwrap()
                    .div_ceil(descriptor.workgroup_size[0]);
                assert!(
                    groups <= gust_wgpu::DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
                    "{}: {count} elements need {groups} workgroups",
                    descriptor.entry_point
                );
            }
        }
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn descriptors_cover_distinct_types_helpers_and_bindings() {
        assert_eq!(sensor_pipeline::MODULE_DESCRIPTOR.kernels.len(), 3);
        let calibrate = sensor_pipeline::calibrate::DESCRIPTOR;
        assert!(calibrate.slang_source.contains("struct SensorReading"));
        assert!(calibrate.slang_source.contains("struct Calibration"));
        assert!(calibrate.slang_source.contains("struct CalibratedSample"));
        assert!(calibrate.slang_source.contains("float clamp_value("));
        assert!(calibrate.slang_source.contains("float sample_quality("));
        assert_eq!(calibrate.parameters[1].binding.unwrap().binding, 0);
        assert_eq!(calibrate.parameters[3].binding.unwrap().binding, 2);

        let classify = sensor_pipeline::classify::DESCRIPTOR;
        assert!(classify.slang_source.contains("struct AlertThresholds"));
        assert!(classify.slang_source.contains("struct Alert"));
        assert!(
            classify
                .slang_source
                .contains("float classification_score(")
        );

        let finalize = sensor_pipeline::finalize::DESCRIPTOR;
        assert!(finalize.slang_source.contains("struct Report"));
        assert_eq!(finalize.parameters[2].binding.unwrap().binding, 1);
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn every_stage_compiles_to_wgsl() {
        for descriptor in descriptors() {
            let wgsl = gust::slang::compile_wgsl(descriptor).unwrap();
            assert!(wgsl.contains("@compute"));
            assert!(wgsl.contains(descriptor.entry_point));
        }
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn dependent_multi_dispatch_pipeline_matches_cpu() {
        let device = HeadlessDevice::new().expect("Vulkan adapter");
        for count in [0, 1, 63, 64, 65, 257] {
            let inputs = make_inputs(count);
            let expected = run_cpu(&inputs);
            let actual = run_gpu(&device, &inputs).unwrap();
            assert_close(&actual, &expected);
        }
        let cache = device.pipeline_cache_stats();
        assert_eq!(cache.entries, 3);
        assert_eq!(cache.misses, 3);
        assert!(cache.hits >= 12);
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn struct_layouts_are_distinct_and_storage_compatible() {
        let layouts = [
            <sensor_pipeline::SensorReading as GpuPod>::LAYOUT,
            <sensor_pipeline::Calibration as GpuPod>::LAYOUT,
            <sensor_pipeline::CalibratedSample as GpuPod>::LAYOUT,
            <sensor_pipeline::AlertThresholds as GpuPod>::LAYOUT,
            <sensor_pipeline::Alert as GpuPod>::LAYOUT,
            <sensor_pipeline::Report as GpuPod>::LAYOUT,
        ];
        assert_eq!(layouts[0].size, 16);
        assert_eq!(layouts[3].size, 24);
        assert!(layouts.into_iter().all(|layout| layout.is_storage_v1()));
    }
}
