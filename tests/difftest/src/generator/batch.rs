//! Batches: the discard reasons and the fill loop.

use super::*;

/// The in-domain programs of one seed and what was discarded on the way.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Batch {
    pub seed: u64,
    pub programs: Vec<Program>,
    /// Attempts made, accepted or not (both parts of the batch).
    pub attempts: u64,
    /// Discarded attempts by reason (`Violation::label`, `static`, `no-input-read` or, in the vector part,
    /// `no-aggregate`), both parts of the batch.
    pub discarded: BTreeMap<&'static str, u32>,
    /// Of `attempts`, the attempts of the vector part, and of `discarded`, the discards of the vector part.
    pub vector_attempts: u64,
    pub vector_discarded: BTreeMap<&'static str, u32>,
    /// Of `attempts`, the attempts of the method part, and of `discarded`, the discards of the method part.
    pub method_attempts: u64,
    pub method_discarded: BTreeMap<&'static str, u32>,
    /// Of `attempts`, the attempts of the enum part, and of `discarded`, the discards of the enum part.
    pub enum_attempts: u64,
    pub enum_discarded: BTreeMap<&'static str, u32>,
    /// Of `attempts`, the attempts of the value part, and of `discarded`, the discards of the value part.
    pub value_attempts: u64,
    pub value_discarded: BTreeMap<&'static str, u32>,
    /// Of `attempts`, the attempts of the array part, and of `discarded`, the discards of the array part.
    pub array_attempts: u64,
    pub array_discarded: BTreeMap<&'static str, u32>,
    /// Of `attempts`, the attempts of the closure part, and of `discarded`, the discards of the closure part.
    pub closure_attempts: u64,
    pub closure_discarded: BTreeMap<&'static str, u32>,
    /// Of `attempts`, the attempts of the trait part, and of `discarded`, the discards of the trait part.
    pub trait_attempts: u64,
    pub trait_discarded: BTreeMap<&'static str, u32>,
    /// Of `attempts`, the attempts of the math part, and of `discarded`, the discards of the math part.
    pub math_attempts: u64,
    pub math_discarded: BTreeMap<&'static str, u32>,
}

impl Batch {
    pub fn discarded_total(&self) -> u32 {
        self.discarded.values().sum()
    }
}

/// Why a program is not a test: it reads no input at all, holds a closed constant outside the domain anywhere (`static`,
/// dead code included), or leaves the domain for some input.
pub fn discard_reason(program: &Program) -> Option<&'static str> {
    if !program.reads_input() {
        return Some("no-input-read");
    }
    if interp::static_violation(program).is_some() {
        return Some("static");
    }
    interp::run(program)
        .err()
        .map(|failure| failure.violation.label())
}

/// Generates the `PROGRAMS_PER_SEED` in-domain programs of `seed`: the first `SCALAR_PROGRAMS_PER_SEED` of the float
/// space (attempts `0..`, the programs of DIFFTEST_FORMAT 5), then `VECTOR_PROGRAMS_PER_SEED` of the vector space
/// (attempts `VECTOR_ATTEMPT_BASE..`), then `METHOD_PROGRAMS_PER_SEED` of the method space (attempts
/// `METHOD_ATTEMPT_BASE..`).
pub fn generate_batch(seed: u64) -> Result<Batch, String> {
    let mut batch = empty_batch(seed);
    fill(&mut batch, Space::Float, SCALAR_PROGRAMS_PER_SEED)?;
    fill(&mut batch, Space::Vector, VECTOR_PROGRAMS_PER_SEED)?;
    fill(&mut batch, Space::Method, METHOD_PROGRAMS_PER_SEED)?;
    fill(&mut batch, Space::Enum, ENUM_PROGRAMS_PER_SEED)?;
    fill(&mut batch, Space::Value, VALUE_PROGRAMS_PER_SEED)?;
    fill(&mut batch, Space::Array, ARRAY_PROGRAMS_PER_SEED)?;
    fill(&mut batch, Space::Closure, CLOSURE_PROGRAMS_PER_SEED)?;
    fill(&mut batch, Space::Trait, TRAIT_PROGRAMS_PER_SEED)?;
    fill(&mut batch, Space::Math, MATH_PROGRAMS_PER_SEED)?;
    Ok(batch)
}

/// Generates the `SPACE_PROGRAMS_PER_SEED` in-domain programs of `seed` all in `space`, in attempt order.
pub fn generate_batch_in(space: Space, seed: u64) -> Result<Batch, String> {
    let mut batch = empty_batch(seed);
    fill(&mut batch, space, SPACE_PROGRAMS_PER_SEED)?;
    Ok(batch)
}

fn empty_batch(seed: u64) -> Batch {
    Batch {
        seed,
        programs: Vec::new(),
        attempts: 0,
        discarded: BTreeMap::new(),
        vector_attempts: 0,
        vector_discarded: BTreeMap::new(),
        method_attempts: 0,
        method_discarded: BTreeMap::new(),
        enum_attempts: 0,
        enum_discarded: BTreeMap::new(),
        value_attempts: 0,
        value_discarded: BTreeMap::new(),
        array_attempts: 0,
        array_discarded: BTreeMap::new(),
        closure_attempts: 0,
        closure_discarded: BTreeMap::new(),
        trait_attempts: 0,
        trait_discarded: BTreeMap::new(),
        math_attempts: 0,
        math_discarded: BTreeMap::new(),
    }
}

/// Adds `want` more in-domain programs of `space` to `batch`, from the first attempt number of the space.
fn fill(batch: &mut Batch, space: Space, want: usize) -> Result<(), String> {
    let first = match space {
        Space::Vector => VECTOR_ATTEMPT_BASE,
        Space::Method => METHOD_ATTEMPT_BASE,
        Space::Enum => ENUM_ATTEMPT_BASE,
        Space::Value => VALUE_ATTEMPT_BASE,
        Space::Array => ARRAY_ATTEMPT_BASE,
        Space::Closure => CLOSURE_ATTEMPT_BASE,
        Space::Trait => TRAIT_ATTEMPT_BASE,
        Space::Math => MATH_ATTEMPT_BASE,
        _ => 0,
    };
    let target = batch.programs.len() + want;
    for attempt in first..first + MAX_ATTEMPTS {
        batch.attempts += 1;
        batch.vector_attempts += u64::from(space == Space::Vector);
        batch.method_attempts += u64::from(space == Space::Method);
        batch.enum_attempts += u64::from(space == Space::Enum);
        batch.value_attempts += u64::from(space == Space::Value);
        batch.array_attempts += u64::from(space == Space::Array);
        batch.closure_attempts += u64::from(space == Space::Closure);
        batch.trait_attempts += u64::from(space == Space::Trait);
        batch.math_attempts += u64::from(space == Space::Math);
        let program = generate_attempt_in(space, batch.seed, attempt);
        // A vector program that came out scalar tests nothing new, nor does a method program that calls nothing, nor
        // an enum program without a `match`.
        let reason = if space == Space::Vector && !program.has_aggregate() {
            Some("no-aggregate")
        } else if space == Space::Method && !program.has_call() {
            Some("no-call")
        } else if space == Space::Enum && !program.has_match() {
            Some("no-match")
        } else if space == Space::Enum && !program.has_call() {
            Some("no-call")
        } else if space == Space::Value && !program.has_value_form() {
            Some("no-value")
        } else if space == Space::Closure && !program.has_closure() {
            Some("no-closure")
        } else if space == Space::Trait && !program.has_trait_forms() {
            Some("no-trait")
        } else if space == Space::Math && !program.has_math_forms() {
            Some("no-math")
        } else if matches!(space, Space::Array | Space::Closure) && !program.has_array() {
            Some("no-array")
        } else if space == Space::Trait && !program.has_match() {
            Some("no-match")
        } else if matches!(space, Space::Value | Space::Array | Space::Closure)
            && !program.has_call()
        {
            Some("no-call")
        } else {
            discard_reason(&program)
        };
        match reason {
            Some(reason) => {
                *batch.discarded.entry(reason).or_default() += 1;
                if space == Space::Vector {
                    *batch.vector_discarded.entry(reason).or_default() += 1;
                }
                if space == Space::Method {
                    *batch.method_discarded.entry(reason).or_default() += 1;
                }
                if space == Space::Enum {
                    *batch.enum_discarded.entry(reason).or_default() += 1;
                }
                if space == Space::Array {
                    *batch.array_discarded.entry(reason).or_default() += 1;
                }
                if space == Space::Closure {
                    *batch.closure_discarded.entry(reason).or_default() += 1;
                }
                if space == Space::Trait {
                    *batch.trait_discarded.entry(reason).or_default() += 1;
                }
                if space == Space::Math {
                    *batch.math_discarded.entry(reason).or_default() += 1;
                }
                if space == Space::Value {
                    *batch.value_discarded.entry(reason).or_default() += 1;
                }
            }
            None => batch.programs.push(program),
        }
        if batch.programs.len() == target {
            return Ok(());
        }
    }
    Err(format!(
        "seed {}: only {} in-domain programs of {space:?} in {MAX_ATTEMPTS} attempts",
        batch.seed,
        batch.programs.len()
    ))
}
