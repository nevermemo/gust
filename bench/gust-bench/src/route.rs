//! The route record of a cold start (plan R1.2-5): which path produced the shader of a kernel's first dispatch, the
//! precompiled artifact or a run-time compile, and what shows it. One record per cold-start sample, so the artifact
//! path is demonstrated by the run, not assumed from a label.
//!
//! The evidence, strongest first (`evidence` and `shown` in the record say which one a sample has):
//!
//! 1. `route-event`: a build with the `spirv-route` feature keeps a route event per pipeline
//!    (`HeadlessDevice::route_events`); a kernel built from an embedded artifact says "precompiled WGSL artifact".
//! 2. `strict-profile-build`: a build under `GUST_PROFILE=native` (or `web`) refuses the run-time compiler and requires an
//!    artifact for every kernel at compile time, so a dispatch that succeeded can only have used the artifact. The
//!    harness reads the build's own `gust::profile::STRICT`.
//! 3. `no-embedded-artifact`: the kernel's descriptor holds no artifact, so the run-time compiler is the only source.
//! 4. `descriptor-and-policy` (`shown: false`, an inference): the descriptor holds an artifact and the shader source
//!    policy (D45) prefers it, but this build has neither an event nor a build rule that proves it. Reported as such.

use serde_json::{Value, json};

/// The route event of one pipeline, as the runtime reports it (a copy of `gust_wgpu::RouteEvent`'s fields, so this
/// module builds without the GPU runtime).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event {
    /// `spirv` or `wgsl`.
    pub route: &'static str,
    /// Why the SPIR-V route was not taken, for a `wgsl` event of a build with the route.
    pub reason: Option<String>,
}

/// What a cold start knows about its kernel and its build.
pub struct Evidence<'a> {
    pub kernel: &'a str,
    /// Artifacts embedded in the kernel's descriptor.
    pub embedded_artifacts: usize,
    /// `gust::profile::STRICT` of this build.
    pub strict_profile: bool,
    /// The last route event of this kernel's pipeline, when the build keeps events.
    pub event: Option<Event>,
}

/// The route record: `path` is `artifact` or `run-time compile`.
pub fn record(e: &Evidence<'_>) -> Value {
    let (path, evidence, shown, route, reason) = match &e.event {
        Some(event) => {
            let from_artifact = event
                .reason
                .as_deref()
                .is_some_and(|r| r.contains("precompiled WGSL artifact"));
            (
                if from_artifact {
                    "artifact"
                } else {
                    "run-time compile"
                },
                "route-event",
                true,
                Some(event.route),
                event.reason.clone(),
            )
        }
        None if e.strict_profile => ("artifact", "strict-profile-build", true, None, None),
        None if e.embedded_artifacts == 0 => {
            ("run-time compile", "no-embedded-artifact", true, None, None)
        }
        None => ("artifact", "descriptor-and-policy", false, None, None),
    };
    json!({
        "kernel": e.kernel,
        "path": path,
        "evidence": evidence,
        "shown": shown,
        "embedded_artifacts": e.embedded_artifacts,
        "strict_profile": e.strict_profile,
        "route_event": route,
        "route_event_reason": reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(artifacts: usize, strict: bool, event: Option<Event>) -> Value {
        record(&Evidence {
            kernel: "k::f",
            embedded_artifacts: artifacts,
            strict_profile: strict,
            event,
        })
    }

    #[test]
    fn a_precompiled_strict_build_reports_the_artifact_route_as_shown() {
        let r = evidence(1, true, None);
        assert_eq!(r["path"], "artifact");
        assert_eq!(r["evidence"], "strict-profile-build");
        assert_eq!(r["shown"], true);
        assert_eq!(r["kernel"], "k::f");
    }

    #[test]
    fn a_route_event_decides_over_everything_else() {
        let artifact = Event {
            route: "wgsl",
            reason: Some("precompiled WGSL artifact: its bytes are the contract".to_owned()),
        };
        let r = evidence(1, false, Some(artifact));
        assert_eq!(
            (r["path"].as_str(), r["evidence"].as_str()),
            (Some("artifact"), Some("route-event"))
        );
        let compiled = Event {
            route: "spirv",
            reason: None,
        };
        let r = evidence(0, false, Some(compiled));
        assert_eq!(r["path"], "run-time compile");
        assert_eq!(r["route_event"], "spirv");
        // A WGSL fallback with another reason is a run-time compile too.
        let fallback = Event {
            route: "wgsl",
            reason: Some("device lacks PASSTHROUGH_SHADERS".to_owned()),
        };
        assert_eq!(
            evidence(0, false, Some(fallback))["path"],
            "run-time compile"
        );
    }

    #[test]
    fn a_kernel_without_an_artifact_compiles_and_an_unproven_artifact_is_marked_inferred() {
        let r = evidence(0, false, None);
        assert_eq!(
            (r["path"].as_str(), r["shown"].as_bool()),
            (Some("run-time compile"), Some(true))
        );
        let r = evidence(2, false, None);
        assert_eq!(r["path"], "artifact");
        assert_eq!(r["shown"], false, "an inference is never reported as shown");
        assert_eq!(r["evidence"], "descriptor-and-policy");
    }
}
