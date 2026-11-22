use crate::Error;
use gust::{GpuPod, pod_slice_as_bytes};

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn f32_as_bytes(values: &[f32]) -> &[u8] {
    pod_slice_as_bytes(values)
}

pub(crate) fn pod_vec_from_bytes<T: GpuPod>(bytes: &[u8]) -> Result<Vec<T>, Error> {
    let element_size = size_of::<T>();
    if element_size == 0 || !bytes.len().is_multiple_of(element_size) {
        return Err(Error::Map(format!(
            "{} mapped bytes cannot be decoded as `{}` elements",
            bytes.len(),
            T::LAYOUT.name
        )));
    }
    let len = bytes.len() / element_size;
    let mut values = Vec::<T>::with_capacity(len);
    // SAFETY: `values` owns enough correctly aligned storage for `len` values.
    // `GpuPod` guarantees all copied bit patterns are valid and contain no padding.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            values.as_mut_ptr().cast::<u8>(),
            bytes.len(),
        );
        values.set_len(len);
    }
    Ok(values)
}
