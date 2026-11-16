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

use gust::{Access, Call, GpuPod, TypeLayout, pod_slice_as_bytes, target::GpuKernel};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::{
    BufferBinding, BufferDispatch, Error, GpuBuffer, GpuBufferAccess, GpuUniform, HeadlessDevice,
    PassTimestamps, Raw, TIMESTAMP_RESULT_BYTES, completion::blocking::recv_mapping,
    error::INDIRECT_ARGUMENTS, pod_vec_from_bytes, render,
};

/// Handle to one node of a [`StagedGraph`], in insertion order.
///
/// Every node-adding method returns one; pass it as a dependency or to [`GraphOutput::readback`]:
///
/// ```
/// use gust_wgpu::{GpuBufferAccess, HeadlessDevice, StagedGraph};
///
/// let device = HeadlessDevice::new()?;
/// let buffer = device.create_typed_buffer("values", &[0u32; 2], GpuBufferAccess::ReadWrite);
/// let values = [5u32, 6];
/// let mut graph = StagedGraph::new();
/// let first = graph.write(&buffer, &values);
/// let second = graph.read(&buffer);
/// assert_eq!((first.index(), second.index()), (0, 1));
/// assert_eq!(graph.dependency_indices(second), [first.index()]);
/// # Ok::<(), gust_wgpu::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

impl NodeId {
    /// The node's position in insertion order.
    pub const fn index(self) -> usize {
        self.0
    }
}

/// The direction of a transfer between the host and the GPU.
///
/// The report of an executed graph tags each [`TransferRecord`] with one:
///
/// ```
/// use gust_wgpu::{GpuBufferAccess, HeadlessDevice, StagedGraph, TransferDirection};
///
/// let device = HeadlessDevice::new()?;
/// let buffer = device.create_typed_buffer("values", &[0u32; 2], GpuBufferAccess::ReadWrite);
/// let values = [5u32, 6];
/// let mut graph = StagedGraph::new();
/// graph.write(&buffer, &values);
/// graph.read(&buffer);
/// let report = device.execute_graph(&graph)?.report;
/// let directions: Vec<_> = report.transfers.iter().map(|record| record.direction).collect();
/// assert_eq!(directions, [TransferDirection::Upload, TransferDirection::Readback]);
/// # Ok::<(), gust_wgpu::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferDirection {
    /// Host data written to a GPU buffer.
    Upload,
    /// GPU buffer contents copied back to the host.
    Readback,
}

/// One host/GPU transfer performed by an executed graph.
///
/// Only the crate builds one; read the fields of [`GraphReport::transfers`]:
///
/// ```
/// use gust_wgpu::{GpuBufferAccess, HeadlessDevice, StagedGraph, TransferDirection};
///
/// let device = HeadlessDevice::new()?;
/// let buffer = device.create_typed_buffer("values", &[0u32; 4], GpuBufferAccess::ReadWrite);
/// let values = [1u32, 2, 3, 4];
/// let mut graph = StagedGraph::new();
/// let upload = graph.write(&buffer, &values);
/// let report = device.execute_graph(&graph)?.report;
/// let record = report.transfers[0];
/// assert_eq!(record.node, upload);
/// assert_eq!(record.direction, TransferDirection::Upload);
/// assert_eq!(record.bytes, 16);
/// # Ok::<(), gust_wgpu::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct TransferRecord {
    /// The node that made the transfer.
    pub node: NodeId,
    /// Whether it uploaded or read back.
    pub direction: TransferDirection,
    /// The number of bytes moved.
    pub bytes: u64,
}

/// What an execution moved across the host boundary and what stayed resident.
///
/// `GraphOutput::report` holds one. A graph that uploads four `u32` values and reads them back moves 16 bytes each
/// way and runs no dispatch:
///
/// ```
/// use gust_wgpu::{GpuBufferAccess, HeadlessDevice, StagedGraph};
///
/// let device = HeadlessDevice::new()?;
/// let buffer = device.create_typed_buffer("values", &[0u32; 4], GpuBufferAccess::ReadWrite);
/// let values = [1u32, 2, 3, 4];
/// let mut graph = StagedGraph::new();
/// graph.write(&buffer, &values);
/// graph.read(&buffer);
/// let report = device.execute_graph(&graph)?.report;
/// assert_eq!((report.upload_bytes, report.readback_bytes), (16, 16));
/// assert_eq!(report.dispatches, 0);
/// assert_eq!(report.derived_edges, 1);
/// assert_eq!(report.declared_edges, 0);
/// # Ok::<(), gust_wgpu::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct GraphReport {
    /// Every transfer of the execution.
    pub transfers: Vec<TransferRecord>,
    /// The total number of bytes uploaded.
    pub upload_bytes: u64,
    /// The total number of bytes read back.
    pub readback_bytes: u64,
    /// Dispatch nodes that produced GPU work (zero-element dispatches are skipped).
    pub dispatches: usize,
    /// Dispatches whose workgroup count was read from a GPU buffer; the host
    /// never learned that count, so they are excluded from `workgroups`.
    pub indirect_dispatches: usize,
    /// The workgroups launched by dispatches whose count the host knew; indirect dispatches are not counted.
    pub workgroups: u64,
    /// Dependency edges the graph derived for typed nodes (`write`, `run`, `run_indirect`, `read`).
    pub derived_edges: usize,
    /// Distinct dependency edges the host declared on explicit nodes.
    pub declared_edges: usize,
    /// Bytes of buffers touched by dispatches but by no transfer in this execution.
    pub resident_bytes: u64,
    /// The host wall-clock time that `execute_graph` took.
    pub host_elapsed: Duration,
    /// The GPU time of the graph's dispatch passes, first begin to last end, from
    /// [`HeadlessDevice::execute_graph_timed`] only. `None` from `execute_graph`, on an adapter without timestamp
    /// queries, and for a graph that ran no dispatch pass.
    pub gpu_elapsed: Option<Duration>,
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
///
/// Build the graph with `write`, `run` and `read`, then run it with [`HeadlessDevice::execute_graph`]. Data that no
/// node reads back never leaves the GPU. This graph squares four numbers in two dispatches over one resident buffer
/// and moves only the input and the result across the host boundary:
///
/// ```
/// use gust::gpu;
/// use gust_wgpu::{GpuBufferAccess, HeadlessDevice, StagedGraph};
///
/// #[gpu]
/// mod ops {
///     #[kernel(workgroup_size(64, 1, 1))]
///     pub fn square(id: SV_DispatchThreadID, mut data: RWStructuredBuffer<float>) {
///         let i = id.x;
///         if i < data.len() {
///             data[i] = data[i] * data[i];
///         }
///     }
/// }
///
/// fn main() -> Result<(), gust_wgpu::Error> {
///     let device = HeadlessDevice::new()?;
///     let data = device.create_typed_buffer("data", &[0.0f32; 4], GpuBufferAccess::ReadWrite);
///     let input = [1.0f32, 2.0, 3.0, 4.0];
///     let first = ops::square::call(4, &data);
///     let second = ops::square::call(4, &data);
///
///     let mut graph = StagedGraph::new();
///     let write = graph.write(&data, &input);
///     let one = graph.run(&first);
///     let two = graph.run(&second);
///     let read = graph.read(&data);
///     assert_eq!(graph.len(), 4);
///     assert_eq!(graph.dependency_indices(two), [one.index()]);
///     assert!(graph.dependency_indices(one).contains(&write.index()));
///
///     let output = device.execute_graph(&graph)?;
///     assert_eq!(output.readback::<f32>(read)?, [1.0, 16.0, 81.0, 256.0]);
///     assert_eq!(output.report.dispatches, 2);
///     Ok(())
/// }
/// ```
#[derive(Default)]
pub struct StagedGraph<'a> {
    nodes: Vec<Node<'a>>,
}

impl std::fmt::Debug for StagedGraph<'_> {
    /// The number of nodes.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StagedGraph")
            .field("nodes", &self.nodes.len())
            .finish()
    }
}

impl<'a> StagedGraph<'a> {
    /// An empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph has no node.
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
    pub fn run<'r: 'a, K: GpuKernel, const N: usize>(
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
    pub fn run_indirect<'r: 'a, K: GpuKernel, const N: usize, A: GpuPod>(
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
                        other == buffer && other_access == Access::ReadWrite
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
                        .any(|(other, _)| other == buffer)
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

    /// Resources a node touches, with the access the hazard check must assume.
    fn accesses(node: &Node<'a>) -> Vec<(Raw<'a>, Access)> {
        Self::kind_accesses(&node.kind)
    }

    fn kind_accesses(kind: &NodeKind<'a>) -> Vec<(Raw<'a>, Access)> {
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
                    // A `pixel` node over an extent is the shape's own refusal, before any binding is looked at.
                    render::check_pixel_extent(dispatch.kernel, dispatch.extent())?;
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
                    // A `pixel` kernel has no indirect form: this wins over every binding error below.
                    render::refuse_indirect(dispatch.kernel)?;
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
                        if binding.access == Access::ReadWrite && binding.raw == args.raw {
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
                                other == buffer
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
        let transferred: Vec<Raw<'_>> = self
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                NodeKind::Upload { target, .. } => Some(target.raw),
                NodeKind::Readback { source, .. } => Some(source.raw),
                NodeKind::Dispatch(_) | NodeKind::DispatchIndirect { .. } => None,
            })
            .collect();
        let mut seen: Vec<Raw<'_>> = Vec::new();
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
///
/// [`HeadlessDevice::execute_graph`] returns one. Decode a readback by the node that made it, with the element type
/// of its buffer; another type is [`Error::GraphReadbackType`]:
///
/// ```
/// use gust_wgpu::{Error, GpuBufferAccess, HeadlessDevice, StagedGraph};
///
/// let device = HeadlessDevice::new()?;
/// let buffer = device.create_typed_buffer("values", &[0u32; 3], GpuBufferAccess::ReadWrite);
/// let values = [7u32, 8, 9];
/// let mut graph = StagedGraph::new();
/// graph.write(&buffer, &values);
/// let read = graph.read(&buffer);
/// let output = device.execute_graph(&graph)?;
/// assert_eq!(output.readback::<u32>(read)?, [7, 8, 9]);
/// assert!(matches!(output.readback::<f32>(read), Err(Error::GraphReadbackType { .. })));
/// assert_eq!(output.report.readback_bytes, 12);
/// # Ok::<(), Error>(())
/// ```
pub struct GraphOutput {
    /// What the execution moved and kept resident.
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
    /// the GPU; the report counts their bytes as resident. Encoding a `pixel` node creates its attachment, so the
    /// graph can fail with [`Error::OutOfMemory`] named `GUST pixel attachment`; nothing is submitted then.
    pub fn execute_graph(&self, graph: &StagedGraph<'_>) -> Result<GraphOutput, Error> {
        self.execute_graph_inner(graph, false)
    }

    /// [`execute_graph`](Self::execute_graph) that also measures the GPU time of the graph's dispatches.
    ///
    /// Everything `execute_graph` does, with the same outputs and report, plus [`GraphReport::gpu_elapsed`]: the GPU
    /// time from the beginning of the first dispatch pass to the end of the last, in node order, indirect and `pixel`
    /// passes included. Upload and read-back copies that sit before the first or after the last dispatch are outside
    /// the span; copies between two dispatches are inside it (a timestamp exists only at a pass boundary). Per-node
    /// times are not reported.
    ///
    /// `gpu_elapsed` is `None` when the adapter lacks timestamp queries ([`HeadlessDevice::supports_gpu_timestamps`])
    /// and when the graph ran no dispatch pass (only transfers, or only zero-element dispatches).
    ///
    /// The device has one timestamp profiler. A timed graph holds it from encoding until the timestamps are read, so
    /// a concurrent `execute_graph_timed` or [`HeadlessDevice::dispatch_batch_timed`] on the same device waits for it;
    /// `execute_graph` never takes it, so a program that does not call this pays nothing.
    ///
    /// ```
    /// use gust_wgpu::{GpuBufferAccess, HeadlessDevice, StagedGraph};
    ///
    /// let device = HeadlessDevice::new()?;
    /// let buffer = device.create_typed_buffer("values", &[0u32; 4], GpuBufferAccess::ReadWrite);
    /// let mut graph = StagedGraph::new();
    /// graph.write(&buffer, &[1u32, 2, 3, 4]);
    /// let read = graph.read(&buffer);
    /// let output = device.execute_graph_timed(&graph)?;
    /// assert_eq!(output.readback::<u32>(read)?, [1, 2, 3, 4]);
    /// // No dispatch pass ran, so there is nothing to time.
    /// assert_eq!(output.report.gpu_elapsed, None);
    /// assert_eq!(device.execute_graph(&graph)?.report.gpu_elapsed, None);
    /// # Ok::<(), gust_wgpu::Error>(())
    /// ```
    pub fn execute_graph_timed(&self, graph: &StagedGraph<'_>) -> Result<GraphOutput, Error> {
        self.execute_graph_inner(graph, true)
    }

    fn execute_graph_inner(
        &self,
        graph: &StagedGraph<'_>,
        timed: bool,
    ) -> Result<GraphOutput, Error> {
        let started = Instant::now();
        graph.validate(self)?;

        // The profiler is taken only by a timed run on an adapter that has one. The first and last pass of the
        // graph are not known while encoding (a zero-element dispatch has no pass), so a timed run prepares every
        // dispatch node first (pipeline cache hits) and then stamps the first and last of the passes.
        let profiler = if timed {
            self.timestamp_profiler.as_ref().map(|profiler| {
                profiler
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
            })
        } else {
            None
        };
        let mut pre_prepared = None;
        let mut total_passes = 0_usize;
        if profiler.is_some() {
            let mut all = Vec::with_capacity(graph.nodes.len());
            for node in &graph.nodes {
                let list = match &node.kind {
                    NodeKind::Dispatch(dispatch) => {
                        self.prepare_batch(std::slice::from_ref(dispatch))?
                    }
                    NodeKind::DispatchIndirect { dispatch, .. } => {
                        let placeholder =
                            BufferDispatch::new(dispatch.kernel, 1, dispatch.bindings);
                        self.prepare_indirect(&placeholder)?
                    }
                    _ => Vec::new(),
                };
                total_passes += list.len();
                all.push(list);
            }
            pre_prepared = Some(all);
        }
        let stamps_of = |pass: usize| {
            profiler.as_ref().and_then(|profiler| {
                let begin = (pass == 0).then_some(0);
                let end = (pass + 1 == total_passes).then_some(1);
                (begin.is_some() || end.is_some()).then_some(PassTimestamps {
                    query_set: &profiler.query_set,
                    begin,
                    end,
                })
            })
        };
        let mut pass_index = 0_usize;

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
                            target.buffer(),
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
                    let list = match pre_prepared.as_mut() {
                        Some(all) => std::mem::take(&mut all[index]),
                        None => self.prepare_batch(std::slice::from_ref(dispatch))?,
                    };
                    for prepared in list {
                        prepared.encode(
                            &self.device,
                            &mut encoder,
                            "GUST staged graph pass",
                            stamps_of(pass_index),
                        )?;
                        pass_index += 1;
                        dispatches += 1;
                        workgroups += u64::from(prepared.workgroup_count());
                    }
                }
                NodeKind::DispatchIndirect { dispatch, args } => {
                    // Validation already fixed every binding length; the count is
                    // irrelevant to pipeline and bind-group preparation, and the workgroup count limit
                    // does not apply to it.
                    let placeholder = BufferDispatch::new(dispatch.kernel, 1, dispatch.bindings);
                    let list = match pre_prepared.as_mut() {
                        Some(all) => std::mem::take(&mut all[index]),
                        None => self.prepare_indirect(&placeholder)?,
                    };
                    for prepared in list {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("GUST staged graph indirect compute pass"),
                            timestamp_writes: stamps_of(pass_index).map(|stamps| {
                                wgpu::ComputePassTimestampWrites {
                                    query_set: stamps.query_set,
                                    beginning_of_pass_write_index: stamps.begin,
                                    end_of_pass_write_index: stamps.end,
                                }
                            }),
                        });
                        pass_index += 1;
                        pass.set_pipeline(prepared.cached_kernel.pipeline.compute());
                        pass.set_bind_group(0, &prepared.bind_group, &[]);
                        pass.dispatch_workgroups_indirect(args.buffer(), 0);
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
                        encoder.copy_buffer_to_buffer(source.buffer(), 0, &staging, 0, byte_size);
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

        if let (Some(profiler), 1..) = (&profiler, total_passes) {
            encoder.resolve_query_set(&profiler.query_set, 0..2, &profiler.resolve_buffer, 0);
            encoder.copy_buffer_to_buffer(
                &profiler.resolve_buffer,
                0,
                &profiler.readback_buffer,
                0,
                TIMESTAMP_RESULT_BYTES,
            );
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
        self.wait_for_submission(submission.clone())?;
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| Error::Poll(error.to_string()))?;
        let gpu_elapsed = match (&profiler, total_passes) {
            (Some(profiler), 1..) => Some(self.read_timestamp_duration(profiler, submission)?),
            _ => None,
        };
        drop(profiler);
        drop(staging_uploads);

        let mut readbacks = Vec::with_capacity(pending.len());
        for (readback, receiver) in pending.into_iter().zip(receivers) {
            let bytes = match (readback.staging, receiver) {
                (Some(staging), Some(receiver)) => {
                    recv_mapping(&receiver)?;
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
                gpu_elapsed,
            },
            readbacks,
        })
    }
}
