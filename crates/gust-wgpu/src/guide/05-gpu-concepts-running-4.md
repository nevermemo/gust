```

A `+subgroups` build needs a Vulkan adapter with subgroup operations. In the browser the pinned wgpu cannot request the
feature, so any non-empty capability set is `Error::MissingCapability` there.
