//! The portable limits floor (M8, D36, D84, D143).
//!
//! [`PORTABLE`] is the limits every WebGPU implementation must offer, which are `wgpu`'s `Limits::default()`: the
//! numbers a kernel can count in its own body. A crate opts into the floor with `limits = "portable"` in the
//! `[package.metadata.gust]` table of its `Cargo.toml`, and the `#[gpu]` macro then refuses a kernel that is over any of
//! them (GUST-E0746) when the crate is compiled. Without that setting nothing is checked at compile time and the
//! launch checks of `gust-compute-wgpu` (D36) stay the only ones.
//!
//! No GUST device opens with more than these values for the limits in the table: `LimitsPolicy::Adapter` raises only
//! `max_buffer_size`, `max_storage_buffer_binding_size` and `max_compute_workgroups_per_dimension`, which the table does
//! not hold. So the floor does not settle the two meanings of "default" that decision D84 recorded; those are about the
//! three raised limits. The table holds the limits a kernel's own source decides: the workgroup size on each axis and
//! their product, the numbers of storage and uniform buffers, sampled and storage textures and samplers, and the
//! workgroup memory. The sizes of buffers and bindings and the number of workgroups depend on how a kernel is launched,
//! so they are not in it.
//!
//! The macro cannot depend on this crate, so it keeps a copy of the table; unit tests of the macro, of this crate and
//! of `gust-compute-wgpu` (against `wgpu::Limits::default()` and against the limits an opened device reports) pin the
//! three to one list of values.

/// The limits of the portable floor, by `wgpu`'s names. Read them from [`PORTABLE`]; a later release may add a limit,
/// so the struct is `#[non_exhaustive]` and cannot be built outside this crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct PortableLimits {
    /// The most invocations of a workgroup along x.
    pub max_compute_workgroup_size_x: u32,
    /// The most invocations of a workgroup along y.
    pub max_compute_workgroup_size_y: u32,
    /// The most invocations of a workgroup along z.
    pub max_compute_workgroup_size_z: u32,
    /// The most invocations of a workgroup in all, the product of the three axes.
    pub max_compute_invocations_per_workgroup: u32,
    /// The most storage buffers (read-only or read-write) one kernel binds.
    pub max_storage_buffers_per_shader_stage: u32,
    /// The most uniform buffers one kernel binds.
    pub max_uniform_buffers_per_shader_stage: u32,
    /// The most sampled textures one kernel binds.
    pub max_sampled_textures_per_shader_stage: u32,
    /// The most storage textures (read-write or write-only) one kernel binds.
    pub max_storage_textures_per_shader_stage: u32,
    /// The most samplers one kernel binds.
    pub max_samplers_per_shader_stage: u32,
    /// The most workgroup memory of a kernel, in bytes.
    pub max_compute_workgroup_storage_size: u32,
}

impl PortableLimits {
    /// Every limit of the table as `(wgpu's name, value)`, in a fixed order: the three workgroup axes, the invocations,
    /// the storage buffers, the uniform buffers, the sampled textures, the storage textures, the samplers and the
    /// workgroup memory.
    pub const fn entries(&self) -> [(&'static str, u32); 10] {
        [
            (
                "max_compute_workgroup_size_x",
                self.max_compute_workgroup_size_x,
            ),
            (
                "max_compute_workgroup_size_y",
                self.max_compute_workgroup_size_y,
            ),
            (
                "max_compute_workgroup_size_z",
                self.max_compute_workgroup_size_z,
            ),
            (
                "max_compute_invocations_per_workgroup",
                self.max_compute_invocations_per_workgroup,
            ),
            (
                "max_storage_buffers_per_shader_stage",
                self.max_storage_buffers_per_shader_stage,
            ),
            (
                "max_uniform_buffers_per_shader_stage",
                self.max_uniform_buffers_per_shader_stage,
            ),
            (
                "max_sampled_textures_per_shader_stage",
                self.max_sampled_textures_per_shader_stage,
            ),
            (
                "max_storage_textures_per_shader_stage",
                self.max_storage_textures_per_shader_stage,
            ),
            (
                "max_samplers_per_shader_stage",
                self.max_samplers_per_shader_stage,
            ),
            (
                "max_compute_workgroup_storage_size",
                self.max_compute_workgroup_storage_size,
            ),
        ]
    }
}

/// The portable floor: WebGPU's default limits (a workgroup of 256 x 256 x 64 invocations and 256 in all, 8 storage
/// buffers, 12 uniform buffers, 16 sampled textures, 4 storage textures, 16 samplers and 16 KiB of workgroup memory).
/// `gust-compute-wgpu`'s tests assert it equal to `wgpu::Limits::default()` and to what every device it opens reports.
pub const PORTABLE: PortableLimits = PortableLimits {
    max_compute_workgroup_size_x: 256,
    max_compute_workgroup_size_y: 256,
    max_compute_workgroup_size_z: 64,
    max_compute_invocations_per_workgroup: 256,
    max_storage_buffers_per_shader_stage: 8,
    max_uniform_buffers_per_shader_stage: 12,
    max_sampled_textures_per_shader_stage: 16,
    max_storage_textures_per_shader_stage: 4,
    max_samplers_per_shader_stage: 16,
    max_compute_workgroup_storage_size: 16384,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// The values of the floor, spelled out. The macro's copy of the table (`validate/limits.rs`) pastes this list and
    /// `gust-compute-wgpu`'s tests compare the table with wgpu's defaults, so a changed value fails in three places.
    const GOLDEN: [(&str, u32); 10] = [
        ("max_compute_workgroup_size_x", 256),
        ("max_compute_workgroup_size_y", 256),
        ("max_compute_workgroup_size_z", 64),
        ("max_compute_invocations_per_workgroup", 256),
        ("max_storage_buffers_per_shader_stage", 8),
        ("max_uniform_buffers_per_shader_stage", 12),
        ("max_sampled_textures_per_shader_stage", 16),
        ("max_storage_textures_per_shader_stage", 4),
        ("max_samplers_per_shader_stage", 16),
        ("max_compute_workgroup_storage_size", 16384),
    ];

    #[test]
    fn the_floor_is_the_webgpu_default_table() {
        assert_eq!(PORTABLE.entries(), GOLDEN);
    }

    #[test]
    fn the_entries_are_the_fields_in_order() {
        let floor = PORTABLE;
        let fields = [
            floor.max_compute_workgroup_size_x,
            floor.max_compute_workgroup_size_y,
            floor.max_compute_workgroup_size_z,
            floor.max_compute_invocations_per_workgroup,
            floor.max_storage_buffers_per_shader_stage,
            floor.max_uniform_buffers_per_shader_stage,
            floor.max_sampled_textures_per_shader_stage,
            floor.max_storage_textures_per_shader_stage,
            floor.max_samplers_per_shader_stage,
            floor.max_compute_workgroup_storage_size,
        ];
        assert_eq!(fields, GOLDEN.map(|(_, value)| value));
    }

    #[test]
    fn the_workgroup_axes_fit_the_invocation_limit() {
        // The three axes can each be at their maximum only one at a time: the product limit is what binds.
        let axes = [
            PORTABLE.max_compute_workgroup_size_x,
            PORTABLE.max_compute_workgroup_size_y,
            PORTABLE.max_compute_workgroup_size_z,
        ];
        assert!(
            axes.iter()
                .all(|axis| *axis <= PORTABLE.max_compute_invocations_per_workgroup)
        );
    }
}
