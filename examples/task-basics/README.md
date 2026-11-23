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

Queue cost: today a `submit()` makes two queue submissions with a fresh output and three when the pool recycles a zeroed
output (the pool clears it itself). That is a known gap against D251, counted from the code paths of
`gust::task::submit_and_read` and not measured on a queue. The generated code only builds the kernel's `Call`; the
dispatch, the read-back and the output buffer are made in `submit_and_read`, so the one-submission entry point of slice
m14.encb needs no change to generated code (its encode-plus-read-back entry point itself lives in the backend). (The uploads of the captures are `write_buffer` calls, not
submissions; an empty captured slice on a recycled pool buffer is the one corner that is cleared in a submission of its
own.)
