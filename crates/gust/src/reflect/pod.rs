use super::{
    error::Error,
    model::{
        PodCrossCheck, ReflectedResource, ReflectedResourceKind, ReflectedType, ReflectedTypeKind,
    },
};
use crate::{GpuPod, ShaderLayoutRules, TypeLayout, TypeLayoutKind, abi::MATRIX_LAYOUT};

/// Check all available compiler field metadata against `T`, retaining explicit gaps.
///
/// ```
/// use gust::reflect::{ReflectedResource, ReflectedResourceKind, ReflectedType, ReflectedTypeKind, cross_check_pod};
/// use gust::{Access, ScalarKind};
///
/// let resource = ReflectedResource {
///     name: "out".to_string(),
///     kind: ReflectedResourceKind::StructuredBuffer,
///     access: Access::ReadWrite,
///     binding: 0,
///     space: 0,
///     element: ReflectedType {
///         name: Some("float".to_string()),
///         size: Some(4),
///         alignment: Some(4),
///         stride: Some(4),
///         kind: ReflectedTypeKind::Scalar(ScalarKind::F32),
///     },
///     element_stride: Some(4),
/// };
/// assert!(cross_check_pod::<f32>(&resource)?.is_full_abi_match());
/// // A `u32` buffer does not match the scalar kind the document reports.
/// assert!(cross_check_pod::<u32>(&resource).is_err());
/// # Ok::<(), gust::reflect::Error>(())
/// ```
pub fn cross_check_pod<T: GpuPod>(resource: &ReflectedResource) -> Result<PodCrossCheck, Error> {
    let host = T::LAYOUT;
    if !host.is_storage_v2()
        || host.size as usize != size_of::<T>()
        || host.alignment as usize != align_of::<T>()
    {
        return Err(Error::Mismatch(
            "host layout is not a valid storage-v2 Rust layout".into(),
        ));
    }
    cross_check_layout(host, ShaderLayoutRules::StorageV2, resource)
}

/// Check `host`, declared under `rules`, against one reflected resource.
pub(super) fn cross_check_layout(
    host: TypeLayout,
    rules: ShaderLayoutRules,
    resource: &ReflectedResource,
) -> Result<PodCrossCheck, Error> {
    let (host_fits, expected_kind) = match rules {
        ShaderLayoutRules::StorageV1 => (
            host.is_storage_v1(),
            ReflectedResourceKind::StructuredBuffer,
        ),
        ShaderLayoutRules::StorageV2 => (
            host.is_storage_v2(),
            ReflectedResourceKind::StructuredBuffer,
        ),
        ShaderLayoutRules::Std140V1 => (host.is_std140_v1(), ReflectedResourceKind::ConstantBuffer),
        // A texture is checked by `Reflection::cross_check_kernel` against the document's texture list, never as a
        // buffer.
        ShaderLayoutRules::TextureV1(_) => {
            return Err(Error::Mismatch(format!("host layout is not {rules:?}")));
        }
    };
    if !host_fits {
        return Err(Error::Mismatch(format!("host layout is not {rules:?}")));
    }
    if resource.kind != expected_kind {
        return Err(Error::Mismatch(format!(
            "{}.kind: host {expected_kind:?}, compiler {:?}",
            resource.name, resource.kind
        )));
    }
    let mut report = PodCrossCheck {
        checked_fields: 0,
        aggregate_size_verified: true,
        element_stride_verified: resource.element_stride.is_some(),
        alignment_verified: true,
    };
    check_type(host, &resource.element, &resource.name, &mut report)?;
    if let Some(stride) = resource.element_stride {
        equal(host.stride, stride, &format!("{}.stride", resource.name))?;
    }
    Ok(report)
}
pub(super) fn equal(host: u32, compiler: u32, path: &str) -> Result<(), Error> {
    if host == compiler {
        Ok(())
    } else {
        Err(Error::Mismatch(format!(
            "{path}: host {host}, compiler {compiler}"
        )))
    }
}
fn check_type(
    host: TypeLayout,
    compiler: &ReflectedType,
    path: &str,
    report: &mut PodCrossCheck,
) -> Result<(), Error> {
    if let Some(size) = compiler.size {
        equal(host.size, size, &format!("{path}.size"))?;
    } else {
        report.aggregate_size_verified = false;
    }
    if let Some(alignment) = compiler.alignment {
        equal(host.alignment, alignment, &format!("{path}.alignment"))?;
    } else {
        report.alignment_verified = false;
    }
    if let Some(stride) = compiler.stride {
        // The stride of a matrix is its column stride (schema 2); every other type reports its array stride.
        let host_stride = match host.kind {
            TypeLayoutKind::Matrix { column_stride, .. } => column_stride,
            _ => host.stride,
        };
        equal(host_stride, stride, &format!("{path}.stride"))?;
    } else {
        report.element_stride_verified = false;
    }
    match (host.kind, &compiler.kind) {
        (TypeLayoutKind::Scalar(a), ReflectedTypeKind::Scalar(b)) if a == *b => Ok(()),
        (
            TypeLayoutKind::Vector {
                scalar: a,
                count: n,
            },
            ReflectedTypeKind::Vector {
                scalar: b,
                count: m,
            },
        ) if a == *b && n == *m => Ok(()),
        (
            TypeLayoutKind::Matrix {
                scalar: a,
                rows,
                cols,
                ..
            },
            ReflectedTypeKind::Matrix {
                scalar: b,
                rows: compiler_rows,
                cols: compiler_cols,
                layout,
            },
        ) if a == *b && rows == *compiler_rows && cols == *compiler_cols => {
            if *layout == MATRIX_LAYOUT {
                Ok(())
            } else {
                Err(Error::Mismatch(format!(
                    "{path}.matrixLayout: host {}, compiler {}",
                    MATRIX_LAYOUT.as_str(),
                    layout.as_str()
                )))
            }
        }
        (TypeLayoutKind::Struct(a), ReflectedTypeKind::Struct(b)) if a.len() == b.len() => {
            for (host, compiler) in a.iter().zip(b) {
                if host.name != compiler.name {
                    return Err(Error::Mismatch(format!("{path}: field name/order differs")));
                }
                let path = format!("{path}.{}", host.name);
                equal(host.offset, compiler.offset, &format!("{path}.offset"))?;
                equal(host.ty.size, compiler.size, &format!("{path}.size"))?;
                report.checked_fields += 1;
                check_type(host.ty, &compiler.ty, &path, report)?;
            }
            Ok(())
        }
        // Schema 4 (m9.r2b): the array's size, alignment and stride were compared above; here the count, the distance
        // between elements and the element type, recursively.
        (
            TypeLayoutKind::Array { element, count },
            ReflectedTypeKind::Array {
                element: compiler_element,
                count: compiler_count,
                element_stride,
            },
        ) => {
            equal(count, *compiler_count, &format!("{path}.elementCount"))?;
            equal(
                element.stride,
                *element_stride,
                &format!("{path}.elementStride"),
            )?;
            check_type(*element, compiler_element, &format!("{path}[]"), report)
        }
        _ => Err(Error::Mismatch(format!(
            "{path}: type or field count differs"
        ))),
    }
}
