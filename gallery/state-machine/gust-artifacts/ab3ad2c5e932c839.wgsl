struct _GustLaunchBase_std140_0
{
    @align(16) x_0 : u32,
    @align(4) y_0 : u32,
    @align(8) z_0 : u32,
    @align(4) pad_0 : u32,
};

@binding(2) @group(0) var<uniform> _gust_launch_base_0 : _GustLaunchBase_std140_0;
@binding(0) @group(0) var<storage, read> text_0 : array<u32>;

struct Summary_std430_0
{
    @align(4) words_0 : u32,
    @align(4) longest_word_0 : u32,
    @align(4) numbers_0 : u32,
    @align(4) longest_number_0 : u32,
};

@binding(1) @group(0) var<storage, read_write> summary_0 : array<Summary_std430_0>;

struct Machine_0
{
     inside_0 : u32,
     tokens_0 : u32,
     run_0 : u32,
     longest_0 : u32,
};

struct Numbers_0
{
     tag_0 : u32,
};

struct Ascii_0
{
     code_0 : u32,
};

fn Ascii_gust_r_class_0( this_0 : Ascii_0) -> u32
{
    if((this_0.code_0) == u32(32))
    {
        return u32(0);
    }
    var _S1 : bool;
    if(((this_0.code_0) >= u32(48)))
    {
        _S1 = ((this_0.code_0) <= u32(57));
    }
    else
    {
        _S1 = false;
    }
    if(_S1)
    {
        return u32(1);
    }
    if(((this_0.code_0) >= u32(65)))
    {
        _S1 = ((this_0.code_0) <= u32(90));
    }
    else
    {
        _S1 = false;
    }
    if(_S1)
    {
        _S1 = true;
    }
    else
    {
        if(((this_0.code_0) >= u32(97)))
        {
            _S1 = ((this_0.code_0) <= u32(122));
        }
        else
        {
            _S1 = false;
        }
    }
    if(_S1)
    {
        return u32(2);
    }
    return u32(3);
}

fn Numbers_accepts_0( this_1 : Numbers_0,  _gust_r_class_0 : u32) -> bool
{
    return _gust_r_class_0 == u32(1);
}

struct Tuple_0
{
     value0_0 : u32,
     value1_0 : u32,
};

fn transition_0( state_0 : u32,  accepted_0 : bool) -> Tuple_0
{
    var next_0 : u32;
    var started_0 : u32;
    switch(state_0)
    {
    case u32(0):
        {
            if(accepted_0)
            {
                next_0 = u32(1);
            }
            else
            {
                next_0 = u32(0);
            }
            started_0 = next_0;
        }
    default :
        {
            if(accepted_0)
            {
                next_0 = u32(1);
            }
            else
            {
                next_0 = u32(0);
            }
            started_0 = u32(0);
        }
    }
    var _S2 : Tuple_0 = Tuple_0( next_0, started_0 );
    return _S2;
}

fn Machine_feed_0( this_2 : ptr<function, Machine_0>,  rules_0 : Numbers_0,  symbol_0 : Ascii_0)
{
    var accepted_1 : bool = Numbers_accepts_0(rules_0, Ascii_gust_r_class_0(symbol_0));
    var state_1 : u32;
    if(((*this_2).inside_0) == u32(1))
    {
        state_1 = u32(1);
    }
    else
    {
        state_1 = u32(0);
    }
    var _gust_tuple_3_0 : Tuple_0 = transition_0(state_1, accepted_1);
    (*this_2).tokens_0 = (*this_2).tokens_0 + _gust_tuple_3_0.value1_0;
    if(accepted_1)
    {
        if((_gust_tuple_3_0.value1_0) == u32(1))
        {
            (*this_2).run_0 = u32(1);
        }
        else
        {
            (*this_2).run_0 = (*this_2).run_0 + u32(1);
        }
        if((((*this_2).run_0) > ((*this_2).longest_0)))
        {
            (*this_2).longest_0 = (*this_2).run_0;
        }
    }
    (*this_2).inside_0 = _gust_tuple_3_0.value0_0;
    return;
}

struct Words_0
{
     tag_1 : u32,
};

fn Words_accepts_0( this_3 : Words_0,  _gust_r_class_1 : u32) -> bool
{
    return _gust_r_class_1 == u32(2);
}

fn Machine_feed_1( this_4 : ptr<function, Machine_0>,  rules_1 : Words_0,  symbol_1 : Ascii_0)
{
    var accepted_2 : bool = Words_accepts_0(rules_1, Ascii_gust_r_class_0(symbol_1));
    var state_2 : u32;
    if(((*this_4).inside_0) == u32(1))
    {
        state_2 = u32(1);
    }
    else
    {
        state_2 = u32(0);
    }
    var _gust_tuple_3_1 : Tuple_0 = transition_0(state_2, accepted_2);
    (*this_4).tokens_0 = (*this_4).tokens_0 + _gust_tuple_3_1.value1_0;
    if(accepted_2)
    {
        if((_gust_tuple_3_1.value1_0) == u32(1))
        {
            (*this_4).run_0 = u32(1);
        }
        else
        {
            (*this_4).run_0 = (*this_4).run_0 + u32(1);
        }
        if((((*this_4).run_0) > ((*this_4).longest_0)))
        {
            (*this_4).longest_0 = (*this_4).run_0;
        }
    }
    (*this_4).inside_0 = _gust_tuple_3_1.value0_0;
    return;
}

fn Machine_totals_0( this_5 : Machine_0) -> Tuple_0
{
    var _S3 : Tuple_0 = Tuple_0( this_5.tokens_0, this_5.longest_0 );
    return _S3;
}

struct Summary_0
{
     words_0 : u32,
     longest_word_0 : u32,
     numbers_0 : u32,
     longest_number_0 : u32,
};

@compute
@workgroup_size(64, 1, 1)
fn gpu_scanner_scan(@builtin(global_invocation_id) _gust_launch_id_0 : vec3<u32>)
{
    var id_0 : vec3<u32> = _gust_launch_id_0 + vec3<u32>(_gust_launch_base_0.x_0, _gust_launch_base_0.y_0, _gust_launch_base_0.z_0);
    var _S4 : vec2<u32> = vec2<u32>(arrayLength(&text_0), 4);
    var _gpu_len_text_0 : u32 = _S4.x;
    var _S5 : vec2<u32> = vec2<u32>(arrayLength(&summary_0), 16);
    var _gpu_len_summary_0 : u32 = _S5.x;
    var r_0 : u32 = id_0.x;
    var _S6 : bool;
    if((r_0 < (_gpu_len_text_0 / u32(16))))
    {
        _S6 = (r_0 < _gpu_len_summary_0);
    }
    else
    {
        _S6 = false;
    }
    if(_S6)
    {
        var _gust_struct_0_0 : Machine_0;
        _gust_struct_0_0.inside_0 = u32(0);
        _gust_struct_0_0.tokens_0 = u32(0);
        _gust_struct_0_0.run_0 = u32(0);
        _gust_struct_0_0.longest_0 = u32(0);
        var words_1 : Machine_0 = _gust_struct_0_0;
        var _gust_struct_1_0 : Machine_0;
        _gust_struct_1_0.inside_0 = u32(0);
        _gust_struct_1_0.tokens_0 = u32(0);
        _gust_struct_1_0.run_0 = u32(0);
        _gust_struct_1_0.longest_0 = u32(0);
        var numbers_1 : Machine_0 = _gust_struct_1_0;
        var k_0 : u32 = u32(0);
        loop
        {
            if((k_0 < u32(16)))
            {
            }
            else
            {
                break;
            }
            var _gust_struct_1_1 : Ascii_0;
            _gust_struct_1_1.code_0 = text_0[r_0 * u32(16) + k_0];
            var _gust_struct_2_0 : Words_0;
            _gust_struct_2_0.tag_1 = u32(0);
            var _gust_struct_3_0 : Numbers_0;
            _gust_struct_3_0.tag_0 = u32(0);
            Machine_feed_1(&(words_1), _gust_struct_2_0, _gust_struct_1_1);
            Machine_feed_0(&(numbers_1), _gust_struct_3_0, _gust_struct_1_1);
            k_0 = k_0 + u32(1);
        }
        var _gust_tuple_3_2 : Tuple_0 = Machine_totals_0(words_1);
        var _gust_tuple_4_0 : Tuple_0 = Machine_totals_0(numbers_1);
        var _gust_struct_5_0 : Summary_0;
        _gust_struct_5_0.words_0 = _gust_tuple_3_2.value0_0;
        _gust_struct_5_0.longest_word_0 = _gust_tuple_3_2.value1_0;
        _gust_struct_5_0.numbers_0 = _gust_tuple_4_0.value0_0;
        _gust_struct_5_0.longest_number_0 = _gust_tuple_4_0.value1_0;
        summary_0[r_0].words_0 = _gust_struct_5_0.words_0;
        summary_0[r_0].longest_word_0 = _gust_struct_5_0.longest_word_0;
        summary_0[r_0].numbers_0 = _gust_struct_5_0.numbers_0;
        summary_0[r_0].longest_number_0 = _gust_struct_5_0.longest_number_0;
    }
    return;
}

