#[allow(clippy::assign_op_pattern)]
#[gust::gpu_module]
mod results {
    fn half_if_even(value: uint) -> Result<uint, uint> {
        if value & 1u32 == 0u32 {
            Ok(value / 2u32)
        } else {
            Err(value)
        }
    }

    fn score(value: Result<uint, uint>, fallback: uint) -> uint {
        value.unwrap_or(fallback) * 10u32
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<uint>,
        mut flags: RWStructuredBuffer<uint>,
        mut values: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < input.len() {
            let value = input[i];
            let result = half_if_even(value);
            let mut flag = 0u32;
            if result.is_ok() {
                flag = flag + 1u32;
            }
            if result.is_err() {
                flag = flag + 2u32;
            }
            flags[i] = flag + score(result, 3u32);
            let mut total = result.unwrap_or(7u32);
            match result {
                Err(error) => {
                    total = total + error;
                }
                Ok(half) => {
                    total = total + half;
                }
            };
            values[i] = total;
        }
    }
}