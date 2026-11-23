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
    collections::HashMap,
    sync::mpsc,
    time::{Duration, Instant},
};

use gust::{Access, Call, GpuPod, TypeLayout, pod_slice_as_bytes, target::GpuKernel};

use crate::{
    BindGroupSlot, BufferBinding, BufferDispatch, Error, GpuBuffer, GpuBufferAccess, GpuUniform,
    HeadlessDevice, PassTimestamps, Raw, TIMESTAMP_RESULT_BYTES,
    completion::blocking::recv_mapping,
    encode::{encode_indirect_pass, encode_readback_copy, encode_upload},
    error::INDIRECT_ARGUMENTS,
    pod_vec_from_bytes,
    pool::Lease,
    render,
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
    /// The host wall-clock time of the call (`execute_graph` or `execute_graph_timed`; for the latter it includes any
    /// wait for the device's timestamp profiler).
    pub host_elapsed: Duration,
    /// The GPU time of the graph's dispatch passes, first begin to last end, from
    /// [`HeadlessDevice::execute_graph_timed`] only. `None` from `execute_graph`, on an adapter without timestamp
    /// queries, and for a graph that ran no dispatch pass (only transfers, or only zero-element direct dispatches; an
    /// indirect dispatch always runs a pass, even with zero arguments, so it gives `Some`).
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
    /// The bind group of a dispatch node, kept from one execution to the next (unused by a transfer node).
    bind_group: BindGroupSlot,
}

/// A set of node indices below a bound, one bit each.
struct NodeSet(Vec<u64>);

impl NodeSet {
    fn new(bound: usize) -> Self {
        Self(vec![0; bound.div_ceil(64)])
    }

    fn insert(&mut self, index: usize) {
        self.0[index / 64] |= 1 << (index % 64);
    }

    fn contains(&self, index: usize) -> bool {
        self.0[index / 64] & (1 << (index % 64)) != 0
    }

    /// Add every member of `other`, a set of a smaller or equal bound.
    fn union(&mut self, other: &Self) {
        for (word, added) in self.0.iter_mut().zip(&other.0) {
            *word |= added;
        }
    }
}

/// The earlier nodes that touched one buffer or texture, for the hazard check of [`StagedGraph::validate`].
#[derive(Default)]
struct Touches {
    /// Every earlier node that touched it, ascending.
    all: Vec<usize>,
    /// The earlier nodes that wrote it, ascending.
    writers: Vec<usize>,
    /// The latest of the writers.
    last_writer: Option<usize>,
    /// The nodes that only read it since the latest writer.
    readers_since_writer: Vec<usize>,
}

impl Touches {
    /// The lowest earlier node whose access conflicts with an `access` by a new node and is not one of `ancestors`
    /// (the new node's ancestors), if there is one. A write conflicts with every earlier toucher, a read with every
    /// earlier writer.
    ///
    /// The quick answer looks at the latest writer and the readers since it only: every toucher before that writer
    /// was one of its ancestors when it was validated, so the new node's ancestors cover them when they cover the
    /// writer. Only a violation pays for the scan that names the lowest producer.
    fn first_unordered(&self, access: Access, ancestors: &NodeSet) -> Option<usize> {
        let covered = |index: &usize| ancestors.contains(*index);
        if self.last_writer.iter().all(covered)
            && (access == Access::ReadOnly || self.readers_since_writer.iter().all(covered))
        {
            return None;
        }
        let conflicting = if access == Access::ReadWrite {
            &self.all
        } else {
            &self.writers
        };
        conflicting
            .iter()
            .copied()
            .find(|&index| !ancestors.contains(index))
    }

    fn record(&mut self, node: usize, access: Access) {
        if self.all.last() != Some(&node) {
            self.all.push(node);
        }
        if access == Access::ReadWrite {
            if self.writers.last() != Some(&node) {
                self.writers.push(node);
            }
            self.last_writer = Some(node);
            self.readers_since_writer.clear();
        } else if self.last_writer != Some(node) && self.readers_since_writer.last() != Some(&node)
        {
            self.readers_since_writer.push(node);
        }
    }
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
/// use gust::gpu_module;
/// use gust_wgpu::{GpuBufferAccess, HeadlessDevice, StagedGraph};
///
/// #[gpu_module]
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
            bind_group: BindGroupSlot::default(),
        });
        NodeId(self.nodes.len() - 1)
    }

    fn push_derived(&mut self, kind: NodeKind<'a>) -> NodeId {
        let dependencies = self.derive_dependencies(&kind);
        self.nodes.push(Node {
            kind,
            dependencies,
            derived: true,
            bind_group: BindGroupSlot::default(),
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
        let mut ancestors: Vec<NodeSet> = Vec::with_capacity(self.nodes.len());
        // wgpu compares and hashes a buffer or texture by its id, which never changes, whatever its interior state.
        #[allow(clippy::mutable_key_type)]
        let mut history: HashMap<Raw<'a>, Touches> = HashMap::new();
        for (index, node) in self.nodes.iter().enumerate() {
            let mut reachable = NodeSet::new(index);
            for dependency in &node.dependencies {
                if dependency.0 >= index {
                    return Err(Error::GraphDependencyOrder {
                        node: index,
                        dependency: dependency.0,
                    });
                }
                reachable.insert(dependency.0);
                reachable.union(&ancestors[dependency.0]);
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

            // Every earlier conflicting access to the same buffer must be an ancestor. The accesses of this node are
            // checked against the earlier nodes' only, then recorded.
            let accesses = Self::accesses(node);
            for (buffer, access) in &accesses {
                if let Some(producer) = history
                    .get(buffer)
                    .and_then(|touches| touches.first_unordered(*access, &reachable))
                {
                    return Err(Error::GraphMissingDependency {
                        node: index,
                        producer,
                    });
                }
            }
            for (buffer, access) in accesses {
                history.entry(buffer).or_default().record(index, access);
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
    /// The bytes of the buffer: what the copy, the mapping and the read cover of the (possibly larger) staging buffer.
    bytes: u64,
    /// The pooled staging buffer the copy goes to; `None` for an empty buffer.
    staging: Option<Lease>,
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

/// The query slots pass number `pass` of `total` dispatch passes writes: slot 0 at the beginning of the first pass,
/// slot 1 at the end of the last, so a lone pass writes both and a middle pass none.
fn stamp_slots(pass: usize, total: usize) -> (Option<u32>, Option<u32>) {
    ((pass == 0).then_some(0), (pass + 1 == total).then_some(1))
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
    /// and when the graph ran no dispatch pass (only transfers, or only zero-element direct dispatches; an indirect
    /// dispatch always runs a pass, even with zero arguments, so it gives `Some`).
    ///
    /// The device has one timestamp profiler. A timed graph holds it from encoding until the timestamps are read, so
    /// a concurrent `execute_graph_timed` or [`HeadlessDevice::dispatch_batch_timed`] on the same device waits for it;
    /// `execute_graph` never takes it, so a program that does not call this pays nothing. The wait is part of
    /// [`GraphReport::host_elapsed`].
    ///
    /// A timed run prepares every dispatch node before it encodes any, so when a graph has two faults, a preparation
    /// error of a later node can be returned instead of an earlier `pixel` node's attachment error
    /// ([`Error::OutOfMemory`]), which `execute_graph` would have returned.
    ///
    /// Known limitation, shared with [`HeadlessDevice::dispatch_batch_timed`]: if reading the timestamps fails (a
    /// mapping error or timeout), the profiler's read-back buffer may stay mapped and a later timed call on the device
    /// can then fail.
    ///
    /// ```
    /// use gust::gpu_module;
    /// use gust_wgpu::{GpuBufferAccess, HeadlessDevice, StagedGraph};
    ///
    /// #[gpu_module]
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
    ///     let call = ops::square::call(4, &data);
    ///
    ///     let mut graph = StagedGraph::new();
    ///     graph.write(&data, &input);
    ///     graph.run(&call);
    ///     let read = graph.read(&data);
    ///
    ///     let output = device.execute_graph_timed(&graph)?;
    ///     assert_eq!(output.readback::<f32>(read)?, [1.0, 4.0, 9.0, 16.0]);
    ///     // `Some` where the adapter has timestamp queries, `None` otherwise.
    ///     assert_eq!(
    ///         output.report.gpu_elapsed.is_some(),
    ///         device.supports_gpu_timestamps()
    ///     );
    ///     // `execute_graph` takes no timestamps.
    ///     assert_eq!(device.execute_graph(&graph)?.report.gpu_elapsed, None);
    ///     Ok(())
    /// }
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

        // A timed run on an adapter with timestamp queries stamps the first and last pass. They are not known while
        // encoding (a zero-element dispatch has no pass), so every dispatch node is prepared first, and only then is
        // the device's profiler locked (as `dispatch_batch_timed` does), so a run-time compile on a cache miss never
        // holds it. An untimed run takes no lock and prepares inline.
        let stamp = timed && self.timestamp_profiler.is_some();
        let mut pre_prepared = None;
        let mut total_passes = 0_usize;
        if stamp {
            let mut all = Vec::with_capacity(graph.nodes.len());
            for node in &graph.nodes {
                let list = match &node.kind {
                    NodeKind::Dispatch(dispatch) => {
                        self.prepare_node(dispatch, &node.bind_group)?
                    }
                    NodeKind::DispatchIndirect { dispatch, .. } => {
                        let placeholder =
                            BufferDispatch::new(dispatch.kernel, 1, dispatch.bindings);
                        self.prepare_indirect(&placeholder, &node.bind_group)?
                    }
                    _ => Vec::new(),
                };
                total_passes += list.len();
                all.push(list);
            }
            pre_prepared = Some(all);
        }
        let profiler = if stamp {
            self.timestamp_profiler.as_ref().map(|profiler| {
                profiler
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
            })
        } else {
            None
        };
        let stamps_of = |pass: usize| {
            profiler.as_ref().and_then(|profiler| {
                let (begin, end) = stamp_slots(pass, total_passes);
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
                    staging_uploads.extend(encode_upload(
                        self,
                        &mut encoder,
                        target,
                        bytes,
                        "GUST staged graph upload",
                    )?);
                    transfers.push(TransferRecord {
                        node: NodeId(index),
                        direction: TransferDirection::Upload,
                        bytes: bytes.len() as u64,
                    });
                }
                NodeKind::Dispatch(dispatch) => {
                    let list = match pre_prepared.as_mut() {
                        Some(all) => std::mem::take(&mut all[index]),
                        None => self.prepare_node(dispatch, &node.bind_group)?,
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
                        None => self.prepare_indirect(&placeholder, &node.bind_group)?,
                    };
                    for prepared in list {
                        encode_indirect_pass(
                            &mut encoder,
                            &prepared,
                            args,
                            "GUST staged graph indirect compute pass",
                            stamps_of(pass_index),
                        );
                        pass_index += 1;
                        dispatches += 1;
                        indirect_dispatches += 1;
                    }
                }
                NodeKind::Readback { source, layout } => {
                    let byte_size = source.byte_len();
                    let staging = encode_readback_copy(
                        self,
                        &mut encoder,
                        source,
                        "GUST staged graph readback",
                    )?;
                    pending.push(PendingReadback {
                        node: NodeId(index),
                        layout: *layout,
                        bytes: byte_size,
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
        let submission = self.submit_counted([encoder.finish()]);
        // A staging buffer with a mapping requested never goes back to the pool unless it is recycled after the read.
        let receivers: Vec<_> = pending
            .iter_mut()
            .map(|readback| {
                let bytes = readback.bytes;
                readback.staging.as_mut().map(|staging| {
                    let (sender, receiver) = mpsc::channel();
                    staging.mark_mapped();
                    staging
                        .buffer()
                        .map_async(wgpu::MapMode::Read, 0..bytes, move |result| {
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
                        .buffer()
                        .get_mapped_range(0..readback.bytes)
                        .map_err(|error| Error::Map(error.to_string()))?;
                    let bytes = view.to_vec();
                    drop(view);
                    staging.recycle(true);
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

#[cfg(test)]
mod tests {
    use super::{NodeSet, StagedGraph, Touches, stamp_slots};
    use crate::{GpuBufferAccess, HeadlessDevice};
    use gust::Access;

    #[gust::gpu_module]
    mod ops {
        #[kernel(workgroup_size(64, 1, 1))]
        pub fn add_one(
            id: SV_DispatchThreadID,
            src: StructuredBuffer<uint>,
            mut dst: RWStructuredBuffer<uint>,
        ) {
            let i = id.x;
            if i < dst.len() && i < src.len() {
                dst[i] = src[i] + 1u32;
            }
        }
    }

    #[test]
    fn a_node_set_holds_the_members_added_and_those_of_its_unions() {
        let mut low = NodeSet::new(70);
        low.insert(3);
        low.insert(69);
        let mut high = NodeSet::new(130);
        high.insert(64);
        high.insert(129);
        high.union(&low);
        let members: Vec<usize> = (0..130).filter(|&index| high.contains(index)).collect();
        assert_eq!(members, [3, 64, 69, 129]);
        assert!(!NodeSet::new(0).0.iter().any(|&word| word != 0));
    }

    #[test]
    fn touches_name_the_lowest_conflicting_node_that_is_not_an_ancestor() {
        // Node 0 writes, nodes 1 and 2 read, node 3 writes, node 4 reads.
        let mut touches = Touches::default();
        touches.record(0, Access::ReadWrite);
        touches.record(1, Access::ReadOnly);
        touches.record(2, Access::ReadOnly);
        touches.record(3, Access::ReadWrite);
        touches.record(4, Access::ReadOnly);
        let ancestors = |members: &[usize]| {
            let mut set = NodeSet::new(6);
            for &member in members {
                set.insert(member);
            }
            set
        };
        // The sets are closed the way a validated graph's are: an ancestor brings its own ancestors (1 and 2 follow 0,
        // 3 follows 0, 1 and 2, 4 follows 3).
        // A new write conflicts with all five.
        let everything = [0, 1, 2, 3, 4];
        assert_eq!(
            touches.first_unordered(Access::ReadWrite, &ancestors(&everything)),
            None
        );
        // The reader 4 is the only one left out: found by the readers since the latest writer.
        assert_eq!(
            touches.first_unordered(Access::ReadWrite, &ancestors(&[0, 1, 2, 3])),
            Some(4)
        );
        // The latest writer is left out: the lowest of the nodes left out is named, not the writer.
        assert_eq!(
            touches.first_unordered(Access::ReadWrite, &ancestors(&[0, 2])),
            Some(1)
        );
        assert_eq!(
            touches.first_unordered(Access::ReadWrite, &ancestors(&[])),
            Some(0)
        );
        // A new read conflicts with the writers 0 and 3 only: the readers 1, 2 and 4 are no conflict.
        assert_eq!(
            touches.first_unordered(Access::ReadOnly, &ancestors(&[0, 1, 2, 3])),
            None
        );
        assert_eq!(
            touches.first_unordered(Access::ReadOnly, &ancestors(&[0, 1, 2])),
            Some(3)
        );
        assert_eq!(
            touches.first_unordered(Access::ReadOnly, &ancestors(&[])),
            Some(0)
        );
    }

    #[test]
    fn a_node_that_touches_a_buffer_twice_is_recorded_once() {
        let mut touches = Touches::default();
        touches.record(0, Access::ReadOnly);
        touches.record(0, Access::ReadWrite);
        touches.record(0, Access::ReadOnly);
        assert_eq!(touches.all, [0]);
        assert_eq!(touches.writers, [0]);
        assert_eq!(touches.last_writer, Some(0));
        assert!(touches.readers_since_writer.is_empty());
    }

    #[test]
    fn an_unchanged_node_keeps_its_bind_group_until_its_pipeline_is_replaced() {
        let device = HeadlessDevice::new().expect("Vulkan adapter required");
        let src = device.create_typed_buffer("src", &[1u32, 2, 3], GpuBufferAccess::ReadWrite);
        let dst = device.create_typed_buffer("dst", &[0u32; 3], GpuBufferAccess::ReadWrite);
        let call = ops::add_one::call(3, &src, &dst);
        let nothing = device.create_typed_buffer("nothing", &[0u32; 0], GpuBufferAccess::ReadWrite);
        let nowhere = device.create_typed_buffer("nowhere", &[0u32; 0], GpuBufferAccess::ReadWrite);
        let skipped = ops::add_one::call(0, &nothing, &nowhere);
        let mut graph = StagedGraph::new();
        graph.run(&call);
        graph.run(&skipped);
        assert!(graph.nodes[0].bind_group.kept_group().is_none());

        device.execute_graph(&graph).unwrap();
        let first = graph.nodes[0].bind_group.kept_group().expect("kept");
        assert_eq!(device.read_typed_buffer(&dst).unwrap(), [2, 3, 4]);
        // A node that launches nothing builds nothing to keep.
        assert!(graph.nodes[1].bind_group.kept_group().is_none());

        // The kept group still binds the live buffers: new contents give new results. A timed execution takes the same
        // group as an untimed one.
        device.write_typed_buffer(&src, &[10u32, 20, 30]).unwrap();
        device.execute_graph(&graph).unwrap();
        assert_eq!(
            graph.nodes[0].bind_group.kept_group().as_ref(),
            Some(&first)
        );
        assert_eq!(device.read_typed_buffer(&dst).unwrap(), [11, 21, 31]);
        device
            .write_typed_buffer(&src, &[100u32, 200, 300])
            .unwrap();
        device.execute_graph_timed(&graph).unwrap();
        assert_eq!(
            graph.nodes[0].bind_group.kept_group().as_ref(),
            Some(&first)
        );
        assert_eq!(device.read_typed_buffer(&dst).unwrap(), [101, 201, 301]);

        // A pipeline compiled again is another cache entry: the group is built for it and replaces the kept one.
        device.clear_pipeline_cache();
        device.write_typed_buffer(&src, &[4u32, 5, 6]).unwrap();
        device.execute_graph(&graph).unwrap();
        let rebuilt = graph.nodes[0].bind_group.kept_group().expect("kept");
        assert_ne!(rebuilt, first);
        assert_eq!(device.read_typed_buffer(&dst).unwrap(), [5, 6, 7]);
    }

    #[test]
    fn two_nodes_of_one_kernel_keep_a_bind_group_each() {
        let device = HeadlessDevice::new().expect("Vulkan adapter required");
        let a = device.create_typed_buffer("a", &[1u32, 2], GpuBufferAccess::ReadWrite);
        let b = device.create_typed_buffer("b", &[0u32; 2], GpuBufferAccess::ReadWrite);
        let c = device.create_typed_buffer("c", &[0u32; 2], GpuBufferAccess::ReadWrite);
        let first = ops::add_one::call(2, &a, &b);
        let second = ops::add_one::call(2, &b, &c);
        let mut graph = StagedGraph::new();
        graph.run(&first);
        graph.run(&second);
        for _ in 0..2 {
            device.execute_graph(&graph).unwrap();
            assert_eq!(device.read_typed_buffer(&c).unwrap(), [3, 4]);
        }
        let groups: Vec<_> = graph
            .nodes
            .iter()
            .map(|node| node.bind_group.kept_group().expect("kept"))
            .collect();
        assert_ne!(groups[0], groups[1]);
    }

    #[test]
    fn the_first_pass_stamps_the_beginning_and_the_last_the_end() {
        // One pass carries both slots.
        assert_eq!(stamp_slots(0, 1), (Some(0), Some(1)));
        // Two passes: the first begins, the last ends.
        assert_eq!(stamp_slots(0, 2), (Some(0), None));
        assert_eq!(stamp_slots(1, 2), (None, Some(1)));
        // Three passes: the middle one carries nothing.
        assert_eq!(stamp_slots(0, 3), (Some(0), None));
        assert_eq!(stamp_slots(1, 3), (None, None));
        assert_eq!(stamp_slots(2, 3), (None, Some(1)));
    }
}
