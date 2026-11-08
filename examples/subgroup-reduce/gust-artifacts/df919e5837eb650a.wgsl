@binding(0) @group(0) var<storage, read> data_0 : array<u32>;

@binding(1) @group(0) var<storage, read_write> out_0 : array<u32>;

var<private> _S1 : u32;

var<private> _S2 : u32;

fn _gust_subgroup_count_0( value_0 : bool) -> u32
{
    var ballot_0 : vec4<u32> = subgroupBallot(value_0);
    return countOneBits(ballot_0.x) + countOneBits(ballot_0.y) + countOneBits(ballot_0.z) + countOneBits(ballot_0.w);
}

fn _gust_subgroup_inclusive_add_0( value_1 : u32) -> u32
{
    var _S3 : u32 = subgroupExclusiveAdd(value_1);
    return _S3 + value_1;
}

@compute
@workgroup_size(64, 1, 1)
fn gpu_lanes_survey_subgroups(@builtin(global_invocation_id) id_0 : vec3<u32>, @builtin(subgroup_size) _builtinWaveLaneCount_0 : u32, @builtin(subgroup_invocation_id) _builtinWaveLaneIndex_0 : u32)
{
    _S1 = _builtinWaveLaneCount_0;
    _S2 = _builtinWaveLaneIndex_0;
    var _S4 : vec2<u32> = vec2<u32>(arrayLength(&data_0), 4);
    var _gpu_len_data_0 : u32 = _S4.x;
    var _S5 : vec2<u32> = vec2<u32>(arrayLength(&out_0), 4);
    var _S6 : u32 = id_0.x;
    var _S7 : bool = (_S6 < _gpu_len_data_0);
    var x_0 : u32;
    if(_S7)
    {
        x_0 = data_0[_S6];
    }
    else
    {
        x_0 = u32(0);
    }
    ;
    var _S8 : u32 = _S1;
    ;
    var _S9 : u32 = _S2;
    var count_0 : u32 = _gust_subgroup_count_0((x_0 > u32(10)));
    var through_0 : u32 = _gust_subgroup_inclusive_add_0(x_0);
    if(_S7)
    {
        var base_0 : u32 = _S6 * u32(4);
        out_0[base_0] = _S8;
        out_0[base_0 + u32(1)] = _S9;
        out_0[base_0 + u32(2)] = count_0;
        out_0[base_0 + u32(3)] = through_0;
    }
    return;
}

