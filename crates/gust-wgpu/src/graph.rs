//! Staged execution graph: upload, dispatch, and readback nodes with explicit or
//! derived dependencies, executed as one ordered submission.
//!
//! This is the first bounded proof from `docs/EXECUTION_GRAPH.md`. The explicit
//! methods (`upload`, `dispatch`, `dispatch_indirect`, `readback`) take edges the
//! host declares; the typed methods (`write`, `run`, `run_indirect`, `read`)
//! derive them from the buffers and access modes the node touches. Either way
//! execution rejects a graph whose edges do not cover the buffer hazards its nodes
//! actually create, and runs nodes in insertion order.
//! Uploads are encoded as buffer copies so they stay ordered relative to the
//! dispatches around them; `Queue::write_buffer` would run before the whole
//! submission. A uniform is updated the same way ([`StagedGraph::write_uniform`]): the copy node is a write of the
//! uniform, so a dispatch that reads it, and a later update of it, are ordered around it.

use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

use gust::{Access, Call, GpuPod, Kernel, TypeLayout, pod_slice_as_bytes};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::{
    BufferBinding, BufferDispatch, Error, GpuBuffer, GpuBufferAccess, GpuUniform, HeadlessDevice,
    error::INDIRECT_ARGUMENTS, pod_vec_from_bytes, same_buffer,
};

/// Handle to one node of a [`StagedGraph`], in insertion order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

impl NodeId {
    pub const fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferDirection {
    Upload,
    Readback,
}

/// One host/GPU transfer performed by an executed graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransferRecord {
    pub node: NodeId,
    pub direction: TransferDirection,
    pub bytes: u64,
}

/// What an execution moved across the host boundary and what stayed resident.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct GraphReport {
    pub transfers: Vec<TransferRecord>,
    pub upload_bytes: u64,
    pub readback_bytes: u64,
    /// Dispatch nodes that produced GPU work (zero-element dispatches are skipped).
    pub dispatches: usize,
    /// Dispatches whose workgroup count was read from a GPU buffer; the host
    /// never learned that count, so they are excluded from `workgroups`.
    pub indirect_dispatches: usize,
    pub workgroups: u64,
    /// Dependency edges the graph derived for typed nodes (`write`, `run`, `run_indirect`, `read`).
    pub derived_edges: usize,
    /// Distinct dependency edges the host declared on explicit nodes.
    pub declared_edges: usize,
    /// Bytes of buffers touched by dispatches but by no transfer in this execution.
    pub resident_bytes: u64,
    pub host_elapsed: Duration,
}

enum NodeKind<'a> {
    Upload {
        target: BufferBinding<'a>,
        bytes: &'a [u8],
        element_count: usize,
    },
    Dispatch(BufferDispatch<'a>),
    /// Workgroup counts come from `args` on the GPU; no host element count exists.
    DispatchIndirect {
        dispatch: BufferDispatch<'a>,
        args: BufferBinding<'a>,
    },
    Readback {
        source: BufferBinding<'a>,
        layout: TypeLayout,
    },
}

struct Node<'a> {
    kind: NodeKind<'a>,
    dependencies: Vec<NodeId>,
    /// The graph computed `dependencies` from the buffers the node touches.
    derived: bool,
}

fn upload_kind<'a, T: GpuPod>(buffer: &'a GpuBuffer<T>, values: &'a [T]) -> NodeKind<'a> {
    NodeKind::Upload {
        target: BufferBinding::new(buffer, Access::ReadWrite),
        bytes: pod_slice_as_bytes(values),
        element_count: values.len(),
    }
}

fn uniform_upload_kind<'a, T: GpuPod>(uniform: &'a GpuUniform<T>, value: &'a T) -> NodeKind<'a> {
    NodeKind::Upload {
        target: BufferBinding::new_uniform(uniform, Access::ReadWrite),
        bytes: pod_slice_as_bytes(std::slice::from_ref(value)),
        element_count: 1,
    }
}

fn indirect_kind<'a, T: GpuPod>(
    dispatch: BufferDispatch<'a>,
    args: &'a GpuBuffer<T>,
) -> NodeKind<'a> {
    NodeKind::DispatchIndirect {
        dispatch,
        args: BufferBinding::new(args, Access::ReadOnly),
    }
}

fn readback_kind<'a, T: GpuPod>(buffer: &'a GpuBuffer<T>) -> NodeKind<'a> {
    NodeKind::Readback {
        source: BufferBinding::new(buffer, Access::ReadOnly),
        layout: T::LAYOUT,
    }
}

/// An ordered set of execution nodes over resident typed buffers. Explicit nodes carry declared
/// edges; typed nodes carry edges the graph derives.
#[derive(Default)]
pub struct StagedGraph<'a> {
    nodes: Vec<Node<'a>>,
}

impl<'a> StagedGraph<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Replace the complete contents of a resident buffer from host memory.
    pub fn upload<T: GpuPod>(
        &mut self,
        buffer: &'a GpuBuffer<T>,
        values: &'a [T],
        dependencies: &[NodeId],
    ) -> NodeId {
        self.push(upload_kind(buffer, values), dependencies)
    }

    /// Run one kernel over resident buffers.
    pub fn dispatch(&mut self, dispatch: BufferDispatch<'a>, dependencies: &[NodeId]) -> NodeId {
        self.push(NodeKind::Dispatch(dispatch), dependencies)
    }

    /// Run one kernel with workgroup counts read from `args` on the GPU.
    ///
    /// `dispatch.element_count` is ignored; every binding must opt in with
    /// `independent_length()` and be non-empty, and `args` must come from
    /// `HeadlessDevice::create_indirect_buffer`. The runtime cannot bound the
    /// count: with wgpu's default instance flags (`InstanceFlags::VALIDATION_INDIRECT_CALL`) wgpu
    /// silently zeroes a dispatch that exceeds the device's per-dimension workgroup limit on Vulkan
    /// (wgpu-core also needs `INDIRECT_EXECUTION` and `COMPUTE_SHADERS` downlevel support, which
    /// Vulkan adapters have), and without them the count is unvalidated, so the kernel that writes `args` must clamp.
    pub fn dispatch_indirect<T: GpuPod>(
        &mut self,
        dispatch: BufferDispatch<'a>,
        args: &'a GpuBuffer<T>,
        dependencies: &[NodeId],
    ) -> NodeId {
        self.push(indirect_kind(dispatch, args), dependencies)
    }

    /// Copy a resident read-write buffer back to host memory after execution.
    pub fn readback<T: GpuPod>(
        &mut self,
        buffer: &'a GpuBuffer<T>,
        dependencies: &[NodeId],
    ) -> NodeId {
        self.push(readback_kind(buffer), dependencies)
    }

    /// Typed twin of [`upload`](Self::upload): the edges are derived, not declared.
    pub fn write<T: GpuPod>(&mut self, buffer: &'a GpuBuffer<T>, values: &'a [T]) -> NodeId {
        self.push_derived(upload_kind(buffer, values))
    }

    /// Replace the value of a uniform, ordered with the dispatches around it, with derived edges.
    ///
    /// The value is copied from a staging buffer by a command in the graph's one command buffer, so a dispatch
    /// added before this node reads the old value and one added after it reads the new one. `Queue::write_buffer`
    /// would run before the whole submission and make every dispatch see the last value. The node counts as a write
    /// of the uniform for the edges: it waits for the earlier dispatches that read it and the later ones wait for it.
    /// `value` must outlive the graph, which records its bytes.
    pub fn write_uniform<T: GpuPod>(&mut self, uniform: &'a GpuUniform<T>, value: &'a T) -> NodeId {
        self.push_derived(uniform_upload_kind(uniform, value))
    }

    /// Typed twin of [`dispatch`](Self::dispatch): run a typed [`Call`], with derived edges. The call
    /// must outlive the graph, which records its bindings.
    pub fn run<'r: 'a, K: Kernel, const N: usize>(
        &mut self,
        call: &'a Call<'r, HeadlessDevice, K, N>,
    ) -> NodeId {
        self.push_derived(NodeKind::Dispatch(BufferDispatch::from(call)))
    }

    /// Typed twin of [`dispatch_indirect`](Self::dispatch_indirect): workgroup counts come from `args` on
    /// the GPU and the call's element count is ignored. Every argument of the call must be bound
    /// `Independent`; `args` is read, never written, by the dispatch.
    ///
    /// The runtime cannot bound the count: with wgpu's default instance flags
    /// (`InstanceFlags::VALIDATION_INDIRECT_CALL`) wgpu silently zeroes a dispatch that exceeds the
    /// per-dimension workgroup limit on Vulkan (wgpu-core also needs `INDIRECT_EXECUTION` and
    /// `COMPUTE_SHADERS` downlevel support, which Vulkan adapters have), and without them the count
    /// is unvalidated, so the kernel that writes `args` must clamp. wgpu checks the limits of the
    /// real device, so a limit lowered with `HeadlessDevice::with_limits` never applies to indirect
    /// counts; every other limit GUST checks still does.
    pub fn run_indirect<'r: 'a, K: Kernel, const N: usize, A: GpuPod>(
        &mut self,
        call: &'a Call<'r, HeadlessDevice, K, N>,
        args: &'a GpuBuffer<A>,
    ) -> NodeId {
        self.push_derived(indirect_kind(BufferDispatch::from(call), args))
    }

    /// Typed twin of [`readback`](Self::readback), with derived edges.
    pub fn read<T: GpuPod>(&mut self, buffer: &'a GpuBuffer<T>) -> NodeId {
        self.push_derived(readback_kind(buffer))
    }

    /// The edges of `node`: declared for explicit nodes, derived for typed ones. Ascending for
    /// derived nodes; in declaration order for explicit ones.
    ///
    /// # Panics
    /// If `node` is not a node of this graph.
    pub fn dependencies(&self, node: NodeId) -> &[NodeId] {
        &self.nodes[node.0].dependencies
    }

    /// The edges of `node` as node indices (`NodeId::index`), in the order of
    /// [`dependencies`](Self::dependencies).
    ///
    /// # Panics
    /// If `node` is not a node of this graph.
    pub fn dependency_indices(&self, node: NodeId) -> Vec<usize> {
        self.dependencies(node)
            .iter()
            .map(|dependency| dependency.index())
            .collect()
    }

    /// Whether the graph derived the edges of `node` (a typed method built it).
    ///
    /// # Panics
    /// If `node` is not a node of this graph.
    pub fn is_derived(&self, node: NodeId) -> bool {
        self.nodes[node.0].derived
    }

    fn push(&mut self, kind: NodeKind<'a>, dependencies: &[NodeId]) -> NodeId {
        self.nodes.push(Node {
            kind,
            dependencies: dependencies.to_vec(),
            derived: false,
        });
        NodeId(self.nodes.len() - 1)
    }

    fn push_derived(&mut self, kind: NodeKind<'a>) -> NodeId {
        let dependencies = self.derive_dependencies(&kind);
        self.nodes.push(Node {
            kind,
            dependencies,
            derived: true,
        });
        NodeId(self.nodes.len() - 1)
    }

    /// The conservative edge set of a node about to be appended (not a transitive reduction): for
    /// every buffer it touches, the latest earlier writer; and, for a buffer it writes, every earlier
    /// node that touched the buffer since that writer (the readers). Sorted and deduplicated.
    ///
    /// Every earlier conflicting access is then a direct or transitive dependency, because the latest
    /// writer already depends on everything before it that touches the buffer. `validate` still runs
    /// on the whole graph, so an explicit node with a missing edge is reported, never repaired.
    fn derive_dependencies(&self, kind: &NodeKind<'a>) -> Vec<NodeId> {
        let mut dependencies = Vec::new();
        for (buffer, access) in Self::kind_accesses(kind) {
            let latest_writer = self.nodes.iter().rposition(|node| {
                Self::accesses(node)
                    .into_iter()
                    .any(|(other, other_access)| {
                        same_buffer(other, buffer) && other_access == Access::ReadWrite
                    })
            });
            if let Some(writer) = latest_writer {
                dependencies.push(NodeId(writer));
            }
            if access == Access::ReadWrite {
                let first_reader = latest_writer.map_or(0, |writer| writer + 1);
                for (index, node) in self.nodes.iter().enumerate().skip(first_reader) {
                    if Self::accesses(node)
                        .into_iter()
                        .any(|(other, _)| same_buffer(other, buffer))
                    {
                        dependencies.push(NodeId(index));
                    }
                }
            }
        }
        dependencies.sort_by_key(|dependency| dependency.0);
        dependencies.dedup();
        dependencies
    }

    /// Buffers a node touches, with the access the hazard check must assume.
    fn accesses(node: &Node<'a>) -> Vec<(&'a wgpu::Buffer, Access)> {
        Self::kind_accesses(&node.kind)
    }

    fn kind_accesses(kind: &NodeKind<'a>) -> Vec<(&'a wgpu::Buffer, Access)> {
        match kind {
            NodeKind::Upload { target, .. } => vec![(target.raw, Access::ReadWrite)],
            NodeKind::Dispatch(dispatch) => dispatch
                .bindings
                .iter()
                .map(|binding| (binding.raw, binding.access))
                .collect(),
            NodeKind::DispatchIndirect { dispatch, args } => dispatch
                .bindings
                .iter()
                .map(|binding| (binding.raw, binding.access))
                .chain([(args.raw, Access::ReadOnly)])
                .collect(),
            NodeKind::Readback { source, .. } => vec![(source.raw, Access::ReadOnly)],
        }
    }

    /// Check every node and every edge before any GPU object is created: bindings, launch limits and
    /// aliasing of each node, then the hazards between nodes. Nothing is compiled or cached when this
    /// fails, however late in the graph the offending node is.
    fn validate(&self, device: &HeadlessDevice) -> Result<(), Error> {
        let device_id = device.id;
        let mut ancestors: Vec<Vec<bool>> = Vec::with_capacity(self.nodes.len());
        for (index, node) in self.nodes.iter().enumerate() {
            let mut reachable = vec![false; index];
            for dependency in &node.dependencies {
                if dependency.0 >= index {
                    return Err(Error::GraphDependencyOrder {
                        node: index,
                        dependency: dependency.0,
                    });
                }
                reachable[dependency.0] = true;
                for (earlier, &covered) in ancestors[dependency.0].iter().enumerate() {
                    reachable[earlier] |= covered;
                }
            }

            match &node.kind {
                NodeKind::Upload {
                    target,
                    element_count,
                    ..
                } => {
                    if target.device_id != device_id {
                        return Err(Error::ForeignPersistentBuffer);
                    }
                    if target.len != *element_count {
                        return Err(Error::PersistentBufferLength {
                            expected: target.len,
                            actual: *element_count,
                        });
                    }
                }
                NodeKind::Dispatch(dispatch) => {
                    let resources = crate::validate_persistent_dispatch(
                        device_id,
                        dispatch.kernel,
                        dispatch.element_count,
                        dispatch.bindings,
                    )?;
                    device.check_launch(
                        dispatch.kernel,
                        dispatch.element_count,
                        &resources,
                        dispatch.bindings.iter().map(BufferBinding::limit_bytes),
                    )?;
                }
                NodeKind::DispatchIndirect { dispatch, args } => {
                    if args.device_id != device_id {
                        return Err(Error::ForeignPersistentBuffer);
                    }
                    if !args.indirect
                        || !crate::is_indirect_args_layout(args.layout)
                        || args.len == 0
                    {
                        return Err(Error::IndirectArgsLayout(args.layout.name));
                    }
                    let resources = crate::validate_persistent_dispatch(
                        device_id,
                        dispatch.kernel,
                        // Placeholder count: every binding must be independent anyway.
                        1,
                        dispatch.bindings,
                    )?;
                    for (parameter, binding) in resources.iter().zip(dispatch.bindings) {
                        if !binding.independent_length || binding.len == 0 {
                            return Err(Error::IndirectBindingLength {
                                parameter: parameter.name.to_owned(),
                            });
                        }
                    }
                    // The GPU reads `args` as indirect arguments while the dispatch runs; wgpu rejects a
                    // usage scope that also writes it (an uncaptured-error panic, measured in m2.e).
                    // Read-only storage use of the same buffer is compatible and stays accepted.
                    for (parameter, binding) in resources.iter().zip(dispatch.bindings) {
                        if binding.access == Access::ReadWrite && same_buffer(binding.raw, args.raw)
                        {
                            return Err(Error::AliasedBinding {
                                kernel: dispatch.kernel.qualified_name().to_string(),
                                first: parameter.name.to_owned(),
                                second: INDIRECT_ARGUMENTS.to_owned(),
                            });
                        }
                    }
                    // The host never sees the count (D16): only the kernel shape, the binding indices
                    // and the binding sizes are checked, never the workgroup count.
                    device.check_indirect_launch(
                        dispatch.kernel,
                        &resources,
                        dispatch.bindings.iter().map(BufferBinding::limit_bytes),
                    )?;
                }
                NodeKind::Readback { source, .. } => {
                    if source.device_id != device_id {
                        return Err(Error::ForeignPersistentBuffer);
                    }
                    if source.buffer_access != GpuBufferAccess::ReadWrite {
                        return Err(Error::PersistentBufferNotReadable);
                    }
                }
            }

            // Every earlier conflicting access to the same buffer must be an ancestor.
            for (buffer, access) in Self::accesses(node) {
                for (earlier, previous) in self.nodes[..index].iter().enumerate() {
                    let conflict =
                        Self::accesses(previous)
                            .into_iter()
                            .any(|(other, other_access)| {
                                same_buffer(other, buffer)
                                    && (access == Access::ReadWrite
                                        || other_access == Access::ReadWrite)
                            });
                    if conflict && !reachable[earlier] {
                        return Err(Error::GraphMissingDependency {
                            node: index,
                            producer: earlier,
                        });
                    }
                }
            }
            ancestors.push(reachable);
        }
        Ok(())
    }

    /// Dependency edges recorded on derived (`true`) or explicit (`false`) nodes.
    /// Duplicate declarations of one edge count once.
    fn edge_count(&self, derived: bool) -> usize {
        self.nodes
            .iter()
            .filter(|node| node.derived == derived)
            .map(|node| {
                let mut unique: Vec<usize> = node.dependencies.iter().map(|id| id.0).collect();
                unique.sort_unstable();
                unique.dedup();
                unique.len()
            })
            .sum()
    }

    fn resident_bytes(&self) -> u64 {
        let transferred: Vec<&wgpu::Buffer> = self
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                NodeKind::Upload { target, .. } => Some(target.raw),
                NodeKind::Readback { source, .. } => Some(source.raw),
                NodeKind::Dispatch(_) | NodeKind::DispatchIndirect { .. } => None,
            })
            .collect();
        let mut seen: Vec<&wgpu::Buffer> = Vec::new();
        let mut bytes = 0;
        for node in &self.nodes {
            let bindings = match &node.kind {
                NodeKind::Dispatch(dispatch) | NodeKind::DispatchIndirect { dispatch, .. } => {
                    dispatch.bindings
                }
                _ => continue,
            };
            for binding in bindings {
                if transferred.contains(&binding.raw) || seen.contains(&binding.raw) {
                    continue;
                }
                seen.push(binding.raw);
                bytes += binding.byte_len();
            }
        }
        bytes
    }
}

struct PendingReadback {
    node: NodeId,
    layout: TypeLayout,
    staging: Option<wgpu::Buffer>,
}

/// Readbacks and the transfer report of one executed graph.
pub struct GraphOutput {
    pub report: GraphReport,
    readbacks: Vec<(NodeId, TypeLayout, Vec<u8>)>,
}

impl GraphOutput {
    /// Decode the readback produced by `node`; the element type must match the
    /// buffer the node was created from.
    pub fn readback<T: GpuPod>(&self, node: NodeId) -> Result<Vec<T>, Error> {
        let (_, layout, bytes) = self
            .readbacks
            .iter()
            .find(|(id, ..)| *id == node)
            .ok_or(Error::GraphUnknownReadback(node.0))?;
        if *layout != T::LAYOUT {
            return Err(Error::GraphReadbackType {
                expected: layout.name,
                actual: T::LAYOUT.name,
            });
        }
        pod_vec_from_bytes(bytes)
    }
}

impl HeadlessDevice {
    /// Validate, encode, and run a graph, waiting for its readbacks.
    ///
    /// Nodes execute in insertion order inside one command buffer, so a node's
    /// dependencies, declared or derived, are a contract that is checked, not a
    /// scheduling hint. Intermediate buffers that no node uploads or reads back never leave
    /// the GPU; the report counts their bytes as resident.
    pub fn execute_graph(&self, graph: &StagedGraph<'_>) -> Result<GraphOutput, Error> {
        let started = Instant::now();
        graph.validate(self)?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GUST staged graph encoder"),
            });
        let mut staging_uploads = Vec::new();
        let mut pending = Vec::new();
        let mut transfers = Vec::new();
        let mut dispatches = 0;
        let mut indirect_dispatches = 0;
        let mut workgroups = 0_u64;

        for (index, node) in graph.nodes.iter().enumerate() {
            match &node.kind {
                NodeKind::Upload { target, bytes, .. } => {
                    if !bytes.is_empty() {
                        let staging = self.device.create_buffer_init(&BufferInitDescriptor {
                            label: Some("GUST staged graph upload"),
                            contents: bytes,
                            usage: wgpu::BufferUsages::COPY_SRC,
                        });
                        encoder.copy_buffer_to_buffer(
                            &staging,
                            0,
                            target.raw,
                            0,
                            bytes.len() as u64,
                        );
                        staging_uploads.push(staging);
                    }
                    transfers.push(TransferRecord {
                        node: NodeId(index),
                        direction: TransferDirection::Upload,
                        bytes: bytes.len() as u64,
                    });
                }
                NodeKind::Dispatch(dispatch) => {
                    for prepared in self.prepare_batch(std::slice::from_ref(dispatch))? {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("GUST staged graph compute pass"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&prepared.cached_kernel.pipeline);
                        pass.set_bind_group(0, &prepared.bind_group, &[]);
                        pass.dispatch_workgroups(prepared.workgroup_count, 1, 1);
                        dispatches += 1;
                        workgroups += u64::from(prepared.workgroup_count);
                    }
                }
                NodeKind::DispatchIndirect { dispatch, args } => {
                    // Validation already fixed every binding length; the count is
                    // irrelevant to pipeline and bind-group preparation, and the workgroup count limit
                    // does not apply to it.
                    let placeholder = BufferDispatch::new(dispatch.kernel, 1, dispatch.bindings);
                    for prepared in self.prepare_indirect(&placeholder)? {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("GUST staged graph indirect compute pass"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&prepared.cached_kernel.pipeline);
                        pass.set_bind_group(0, &prepared.bind_group, &[]);
                        pass.dispatch_workgroups_indirect(args.raw, 0);
                        dispatches += 1;
                        indirect_dispatches += 1;
                    }
                }
                NodeKind::Readback { source, layout } => {
                    let byte_size = source.byte_len();
                    let staging = (byte_size > 0).then(|| {
                        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("GUST staged graph readback"),
                            size: byte_size,
                            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                            mapped_at_creation: false,
                        });
                        encoder.copy_buffer_to_buffer(source.raw, 0, &staging, 0, byte_size);
                        staging
                    });
                    pending.push(PendingReadback {
                        node: NodeId(index),
                        layout: *layout,
                        staging,
                    });
                    transfers.push(TransferRecord {
                        node: NodeId(index),
                        direction: TransferDirection::Readback,
                        bytes: byte_size,
                    });
                }
            }
        }

        let submission = self.queue.submit([encoder.finish()]);
        let receivers: Vec<_> = pending
            .iter()
            .map(|readback| {
                readback.staging.as_ref().map(|staging| {
                    let (sender, receiver) = mpsc::channel();
                    staging.map_async(wgpu::MapMode::Read, .., move |result| {
                        let _ = sender.send(result.map_err(|error| error.to_string()));
                    });
                    receiver
                })
            })
            .collect();
        self.wait_for_submission(submission)?;
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| Error::Poll(error.to_string()))?;
        drop(staging_uploads);

        let mut readbacks = Vec::with_capacity(pending.len());
        for (readback, receiver) in pending.into_iter().zip(receivers) {
            let bytes = match (readback.staging, receiver) {
                (Some(staging), Some(receiver)) => {
                    receiver
                        .recv()
                        .map_err(|error| Error::Map(error.to_string()))?
                        .map_err(Error::Map)?;
                    let view = staging
                        .get_mapped_range(..)
                        .map_err(|error| Error::Map(error.to_string()))?;
                    let bytes = view.to_vec();
                    drop(view);
                    staging.unmap();
                    bytes
                }
                _ => Vec::new(),
            };
            readbacks.push((readback.node, readback.layout, bytes));
        }

        let upload_bytes = transfers
            .iter()
            .filter(|transfer| transfer.direction == TransferDirection::Upload)
            .map(|transfer| transfer.bytes)
            .sum();
        let readback_bytes = transfers
            .iter()
            .filter(|transfer| transfer.direction == TransferDirection::Readback)
            .map(|transfer| transfer.bytes)
            .sum();
        Ok(GraphOutput {
            report: GraphReport {
                transfers,
                upload_bytes,
                readback_bytes,
                dispatches,
                indirect_dispatches,
                workgroups,
                derived_edges: graph.edge_count(true),
                declared_edges: graph.edge_count(false),
                resident_bytes: graph.resident_bytes(),
                host_elapsed: started.elapsed(),
            },
            readbacks,
        })
    }
}
