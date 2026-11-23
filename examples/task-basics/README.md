# task-basics

`gpu_task!` as a free task (m14.s2a1): a stage written at the call site, returned as a `Task` that you start with
`submit()` (a future of the read-back, polled by you) or `dispatch()` (blocks until done, native targets).

- **scale**: 1-D, `a[i] * k + b`, a host slice and two scalars; the result is a fresh `Vec<f32>`.
- **grid**: 2-D, `bias[x] + y * 100`, row-major result of a `(WIDTH, HEIGHT)` launch.
- **halve**: in place, `data[i] = data[i] * 0.5 + 1.0` on a `&mut [f32]`.
- **weigh**: 1-D, a second host slice read at a computed index (`weights[i % 4]`).

All four run on the CPU twin (the tests) and on the GPU (`cargo run -p task-basics`, or the ignored test
`gpu_matches_the_host`), each with `dispatch` and with `submit`, and are compared bit for bit with a host reference
written from the rules (every value is exact in `f32`).

Queue cost (D251): on the GPU device a `submit()` is one queue submission, the dispatch and the read-back copy
together, with a fresh or a recycled output, in 1-D, 2-D and in place, as counted by `gust_submissions()` in the
ignored `gpu_*` tests (Radeon 890M only; the RTX 5090 run is pending). The generated code only builds the kernel's
`Call`; the output buffer, the dispatch and the read-back are made by `MapHost::start_read`, which the GPU device
overrides. The uploads of the captures are `write_buffer` calls, not submissions; an empty captured slice on a recycled
pool buffer is the one corner that is cleared in a submission of its own, and the launch then refuses the task
(`IndependentLengthEmpty`), so that submission belongs to a task that fails (pinned by
`gpu_an_empty_capture_on_a_recycled_buffer_makes_its_own_submission`).
