//! The guards of the SPIR-V route (slice m9.rt2; D172, D174).
//!
//! The WGSL route runs through naga, which adds run-time guards that slangc's own SPIR-V does not have: an array index
//! is clamped to the length, a zero divisor (and signed `MIN / -1`) divides by one, a signed `%` avoids `OpSRem`, and
//! workgroup memory starts as zeros. A kernel that takes the SPIR-V route must give the same values, so the macro emits
//! those guards as explicit Slang. They are on whenever the Cargo feature `spirv-route` is on (D174c), for both routes
//! (the text is shared, and a kernel that falls back to WGSL gets the same guarded text), and never a run-time switch.
//! With the feature off nothing here changes the emitted text.

use std::cell::Cell;

thread_local! {
    /// A unit test's override of [`enabled`] for the current thread.
    static FORCED: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Whether the emitter adds the guards of the SPIR-V route: the Cargo feature `spirv-route`. This crate's own unit tests
/// (the goldens pin the text without the guards) run with them off whatever the feature says, and reach the guarded text
/// through [`Forced`]; the proc macro as `gust` and its dependents use it follows the feature.
pub(crate) fn enabled() -> bool {
    FORCED
        .with(Cell::get)
        .unwrap_or(cfg!(all(feature = "spirv-route", not(test))))
}

/// Forces [`enabled`] on the current thread until dropped, so a unit test reaches both texts from one build.
#[cfg(test)]
pub(crate) struct Forced(Option<bool>);

#[cfg(test)]
impl Forced {
    pub(crate) fn new(on: bool) -> Self {
        Self(FORCED.with(|cell| cell.replace(Some(on))))
    }
}

#[cfg(test)]
impl Drop for Forced {
    fn drop(&mut self) {
        FORCED.with(|cell| cell.set(self.0));
    }
}

/// One `Workgroup<[T; N]>` parameter: its emitted name, the Slang type of one element and the length.
#[derive(Clone, Debug)]
pub(crate) struct WorkgroupArray {
    pub(crate) name: String,
    pub(crate) element: String,
    pub(crate) length: u32,
}

/// The Slang zero of an element type: `false`, or the type's constructor applied to 0 (`uint(0)`, `float2(0)`).
fn zero(element: &str) -> String {
    if element == "bool" {
        "false".to_owned()
    } else {
        format!("{element}(0)")
    }
}

/// Zero-initialisation of workgroup memory (D174a): the invocations of the group zero the arrays together, each
/// stepping through the array by the size of the group from its flat local index, then a barrier makes the zeros visible
/// before any read. It is the first statement of the kernel, so every invocation of the group reaches the barrier.
/// `invocation` is the dispatch-thread id, `size` the workgroup size. Empty when the kernel has no workgroup memory.
pub(crate) fn workgroup_zeroing(
    arrays: &[WorkgroupArray],
    invocation: &str,
    size: [u32; 3],
) -> String {
    if arrays.is_empty() {
        return String::new();
    }
    let [x, y, z] = size;
    let threads = u64::from(x) * u64::from(y) * u64::from(z);
    let local = format!(
        "(({invocation}.z % {z}u) * {y}u + ({invocation}.y % {y}u)) * {x}u + ({invocation}.x % {x}u)"
    );
    let mut text =
        String::from("    // Workgroup memory starts as zeros on every route (D174a).\n");
    for array in arrays {
        text.push_str(&format!(
            "    for (uint __gust_zero = {local}; __gust_zero < {}u; __gust_zero += {threads}u) {{ {}[__gust_zero] = {}; }}\n",
            array.length,
            array.name,
            zero(&array.element)
        ));
    }
    text.push_str("    GroupMemoryBarrierWithGroupSync();\n");
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        diagnostics::testing::parse_str, expand::kernel_options, slang::emit_kernel,
        validate::validate_module,
    };

    /// The Slang of every kernel of `source` (a `mod t { .. }`), emitted with the guards forced on or off.
    fn emitted(source: &str, on: bool) -> String {
        let _forced = Forced::new(on);
        let module: syn::ItemMod = parse_str(source).unwrap();
        validate_module(&module).unwrap();
        let mut text = String::new();
        for item in &module.content.as_ref().unwrap().1 {
            let syn::Item::Fn(function) = item else {
                continue;
            };
            let Some(options) = kernel_options(function).unwrap() else {
                continue;
            };
            text.push_str(&emit_kernel(&module, function, options).unwrap().source);
        }
        text.replace("\r\n", "\n")
    }

    const KERNEL: &str = "mod t {
        #[kernel(workgroup_size(64, 1, 1), cpu = \"unsupported\")]
        pub fn k(id: SV_DispatchThreadID, lid: LocalId, src: StructuredBuffer<i32>, mut out: RWStructuredBuffer<i32>,
                 mut scratch: Workgroup<[i32; 64]>) {
            let mut local: [i32; 4] = [1i32, 2i32, 3i32, 4i32];
            let k = src[id.x];
            local[k as usize] = 7i32;
            scratch[lid.x] = local[0u32] + local[2u32];
            workgroup_barrier();
            out[id.x] = scratch[(lid.x + 1u32) as usize] / k + (k % 3i32) + local[2u32] + scratch[7u32];
        }
    }";

    fn array(name: &str, element: &str, length: u32) -> WorkgroupArray {
        WorkgroupArray {
            name: name.to_owned(),
            element: element.to_owned(),
            length,
        }
    }

    #[test]
    fn no_workgroup_memory_no_zeroing() {
        assert_eq!(workgroup_zeroing(&[], "id", [64, 1, 1]), "");
    }

    #[test]
    fn the_zeroing_steps_by_the_group_size_and_ends_with_a_barrier() {
        let text = workgroup_zeroing(
            &[
                array("a", "uint", 64),
                array("b", "float2", 5),
                array("c", "bool", 3),
            ],
            "id",
            [8, 4, 2],
        );
        let local = "((id.z % 2u) * 4u + (id.y % 4u)) * 8u + (id.x % 8u)";
        assert!(
            text.contains(&format!(
                "for (uint __gust_zero = {local}; __gust_zero < 64u; __gust_zero += 64u) {{ a[__gust_zero] = uint(0); }}"
            )),
            "{text}"
        );
        assert!(
            text.contains("< 5u; __gust_zero += 64u) { b[__gust_zero] = float2(0); }"),
            "{text}"
        );
        assert!(
            text.contains("< 3u; __gust_zero += 64u) { c[__gust_zero] = false; }"),
            "{text}"
        );
        assert!(
            text.ends_with("    GroupMemoryBarrierWithGroupSync();\n"),
            "{text}"
        );
        assert_eq!(text.matches("GroupMemoryBarrierWithGroupSync").count(), 1);
    }

    #[test]
    fn with_the_guards_off_the_text_has_none_of_them() {
        let text = emitted(KERNEL, false);
        for absent in [
            "min(uint(",
            "__gust_zero",
            "GroupMemoryBarrierWithGroupSync();\n    int[4]",
            "__gust_rem",
        ] {
            assert!(!text.contains(absent), "{absent}\n{text}");
        }
        // The overloads are the plain ones, the literal `% 3` stays the plain operator, and the one barrier is the kernel's.
        assert!(
            text.contains("int __gust_div(int a, int b) { return a / b; }\n"),
            "{text}"
        );
        assert!(text.contains("(k % int(3))"), "{text}");
        assert_eq!(
            text.matches("GroupMemoryBarrierWithGroupSync();").count(),
            1,
            "{text}"
        );
        assert!(
            text.contains("local[uint(2)]") && text.contains("scratch[lid.x]"),
            "{text}"
        );
    }

    #[test]
    fn with_the_guards_on_the_text_has_the_clamps_the_helpers_and_the_zeroing() {
        let text = emitted(KERNEL, true);
        // Dynamic indices of a local array and of a workgroup array are clamped; a literal index is not.
        assert!(
            text.contains("local[min(uint(uint(k)), 3u)] = int(7);"),
            "{text}"
        );
        assert!(text.contains("scratch[min(uint(lid.x), 63u)] = "), "{text}");
        assert!(
            text.contains("local[uint(0)]") && text.contains("scratch[uint(7)]"),
            "{text}"
        );
        // The literal `% 3` goes through the helper; both overload families are the guarded ones.
        assert!(text.contains("(__gust_rem(k, int(3)))"), "{text}");
        assert!(text.contains("return a - d * (a / d); }\n"), "{text}");
        assert!(text.contains("? 1 : b; return a / d; }\n"), "{text}");
        // The zeroing is the first statement after the prologue: before the body's first line, then a barrier.
        let zeroing = "    for (uint __gust_zero = ((id.z % 1u) * 1u + (id.y % 1u)) * 64u + (id.x % 64u); __gust_zero < 64u; \
                       __gust_zero += 64u) { scratch[__gust_zero] = int(0); }\n    GroupMemoryBarrierWithGroupSync();\n    int[4] local";
        assert!(text.contains(zeroing), "{text}");
        assert!(
            text.find("uint3 lid").unwrap() < text.find("__gust_zero").unwrap(),
            "{text}"
        );
        assert_eq!(
            text.matches("GroupMemoryBarrierWithGroupSync();").count(),
            2,
            "{text}"
        );
    }

    #[test]
    fn a_kernel_without_workgroup_memory_gets_no_zeroing_and_no_extra_barrier() {
        let source = "mod t { #[kernel(workgroup_size(64, 1, 1))] pub fn k(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) { out[id.x] = id.x; } }";
        assert_eq!(emitted(source, true), emitted(source, false));
    }

    #[test]
    fn the_override_is_scoped_to_the_thread_and_restored() {
        let before = enabled();
        {
            let _on = Forced::new(true);
            assert!(enabled());
            {
                let _off = Forced::new(false);
                assert!(!enabled());
            }
            assert!(enabled());
        }
        assert_eq!(enabled(), before);
    }
}
