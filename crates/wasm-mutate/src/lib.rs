//! A WebAssembly test case mutator.
//!
//! `wasm-mutate` takes in an existing Wasm module and then applies a
//! pseudo-random transformation to it, producing a new, mutated Wasm
//! module. This new, mutated Wasm module can be fed as a test input to your
//! Wasm parser, validator, compiler, or any other Wasm-consuming
//! tool. `wasm-mutate` can serve as a custom mutator for mutation-based
//! fuzzing.

#![cfg_attr(not(feature = "clap"), deny(missing_docs))]

mod error;
mod info;
mod module;
mod mutators;

pub use error::*;

use crate::mutators::{
    Item, add_export::AddExportMutator, add_function::AddFunctionMutator,
    add_global::AddGlobalMutator, add_type::AddTypeMutator, codemotion::CodemotionMutator,
    custom::AddCustomSectionMutator, custom::CustomSectionMutator,
    custom::ReorderCustomSectionMutator, function_body_unreachable::FunctionBodyUnreachable,
    modify_const_exprs::ConstExpressionMutator, modify_data::ModifyDataMutator,
    peephole::PeepholeMutator, remove_export::RemoveExportMutator, remove_item::RemoveItemMutator,
    remove_section::RemoveSection, rename_export::RenameExportMutator, snip_function::SnipMutator,
    start::RemoveStartSection,
};
use info::ModuleInfo;
use mutators::Mutator;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use std::sync::Arc;

#[cfg(feature = "clap")]
use clap::Parser;

// NB: only add this doc comment if we are not building the CLI, since otherwise
// it will override the main CLI's about text.
#[cfg_attr(
    not(feature = "clap"),
    doc = r###"
A WebAssembly test case mutator.

This is the main entry point into this crate. It provides various methods for
configuring what kinds of mutations might be applied to the input Wasm. Once
configured, you can apply a transformation to the input Wasm via the
[`run`][crate::WasmMutate::run] method.

# Example

```
# fn _foo() -> anyhow::Result<()> {
use wasm_mutate::WasmMutate;

let input_wasm = wat::parse_str(r#"
           (module
            (func (export "hello") (result i32)
             (i32.const 1234)
            )
           )
           "#)?;

// Create a `WasmMutate` builder and configure it.
let mut mutate = WasmMutate::default();
mutate
    // Set the RNG seed.
    .seed(42)
    // Allow mutations that change the semantics of the Wasm module.
    .preserve_semantics(false)
    // Use at most this much "fuel" when trying to mutate the Wasm module before
    // giving up.
    .fuel(1_000);

// Run the configured `WasmMutate` to get a sequence of mutations of the input
// Wasm!
for mutated_wasm in mutate.run(&input_wasm)? {
    let mutated_wasm = mutated_wasm?;
    // Feed `mutated_wasm` into your tests...
}
# Ok(())
# }
```
"###
)]
#[cfg_attr(feature = "clap", derive(Parser))]
#[derive(Clone)]
pub struct WasmMutate<'wasm> {
    /// The RNG seed used to choose which transformation to apply. Given the
    /// same input Wasm and same seed, `wasm-mutate` will always generate the
    /// same output Wasm.
    #[cfg_attr(feature = "clap", clap(short, long, default_value = "42"))]
    seed: u64,

    /// Only perform semantics-preserving transformations on the Wasm module.
    #[cfg_attr(feature = "clap", clap(long))]
    preserve_semantics: bool,

    /// Fuel to control the time of the mutation.
    #[cfg_attr(
        feature = "clap",
        clap(
            short,
            long,
            default_value = "18446744073709551615", // u64::MAX
        )
    )]
    fuel: u64,

    /// Only perform size-reducing transformations on the Wasm module. This
    /// allows `wasm-mutate` to be used as a test case reducer.
    #[cfg_attr(feature = "clap", clap(long))]
    reduce: bool,

    /// Never add a global to the Wasm module. This disables the `add_global`
    /// mutator and the peephole rules that spill values through a new global.
    #[cfg_attr(feature = "clap", clap(long))]
    no_new_globals: bool,

    /// Never use the mutators with this name. A name matches a mutator when it
    /// equals a segment of the mutator's path, so `custom` skips every mutator
    /// in `mutators::custom` and `RemoveItemMutator` skips that one type. May
    /// be given more than once.
    #[cfg_attr(feature = "clap", clap(long = "skip-mutator", value_name = "NAME"))]
    skip_mutators: Vec<String>,

    // Note: this is only exposed via the programmatic interface, not via the
    // CLI.
    #[cfg_attr(feature = "clap", clap(skip = None))]
    raw_mutate_func: Option<Arc<dyn Fn(&mut Vec<u8>, usize) -> Result<()>>>,

    #[cfg_attr(feature = "clap", clap(skip = None))]
    rng: Option<SmallRng>,

    #[cfg_attr(feature = "clap", clap(skip = None))]
    info: Option<ModuleInfo<'wasm>>,
}

impl Default for WasmMutate<'_> {
    fn default() -> Self {
        let seed = 3;
        WasmMutate {
            seed,
            preserve_semantics: false,
            reduce: false,
            no_new_globals: false,
            skip_mutators: Vec::new(),
            raw_mutate_func: None,
            fuel: u64::MAX,
            rng: None,
            info: None,
        }
    }
}

impl<'wasm> WasmMutate<'wasm> {
    /// Set the RNG seed used to choose which transformation to apply.
    ///
    /// Given the same input Wasm and same seed, `wasm-mutate` will always
    /// generate the same output Wasm.
    pub fn seed(&mut self, seed: u64) -> &mut Self {
        self.seed = seed;
        self
    }

    /// Configure whether we will only perform semantics-preserving
    /// transformations on the Wasm module.
    pub fn preserve_semantics(&mut self, preserve_semantics: bool) -> &mut Self {
        self.preserve_semantics = preserve_semantics;
        self
    }

    /// Configure the fuel used during the mutation
    pub fn fuel(&mut self, fuel: u64) -> &mut Self {
        self.fuel = fuel;
        self
    }

    /// Configure whether we will only perform size-reducing transformations on
    /// the Wasm module.
    ///
    /// Setting this to `true` allows `wasm-mutate` to be used as a test case
    /// reducer.
    pub fn reduce(&mut self, reduce: bool) -> &mut Self {
        self.reduce = reduce;
        self
    }

    /// Configure whether mutations may add globals to the Wasm module.
    ///
    /// Setting this to `true` disables the `add_global` mutator and the
    /// peephole rewrites that spill a value through a new global.
    pub fn no_new_globals(&mut self, no_new_globals: bool) -> &mut Self {
        self.no_new_globals = no_new_globals;
        self
    }

    /// Never use the mutators with this name.
    ///
    /// A name matches a mutator when it equals a segment of the mutator's
    /// path, so `custom` skips every mutator in `mutators::custom` and
    /// `RemoveItemMutator` skips that one type. Running with a name that
    /// matches no mutator is an error.
    pub fn skip_mutator(&mut self, name: impl Into<String>) -> &mut Self {
        self.skip_mutators.push(name.into());
        self
    }

    /// Set a custom raw mutation function.
    ///
    /// This is used when we need some underlying raw bytes, for example when
    /// mutating the contents of a data segment.
    ///
    /// You can override this to use `libFuzzer`'s `LLVMFuzzerMutate` function
    /// to get raw bytes from `libFuzzer`, for example.
    ///
    /// The function is given the raw data buffer and the maximum size the
    /// mutated data should be. After mutating the data, the function should
    /// `resize` the data to its final, mutated size, which should be less than
    /// or equal to the maximum size.
    pub fn raw_mutate_func(
        &mut self,
        raw_mutate_func: Option<Arc<dyn Fn(&mut Vec<u8>, usize) -> Result<()>>>,
    ) -> &mut Self {
        self.raw_mutate_func = raw_mutate_func;
        self
    }

    pub(crate) fn consume_fuel(&mut self, qt: u64) -> Result<()> {
        if qt > self.fuel {
            log::info!("Out of fuel");
            return Err(Error::out_of_fuel());
        }
        self.fuel -= qt;
        Ok(())
    }

    /// Run this configured `WasmMutate` on the given input Wasm.
    pub fn run<'a>(
        &'a mut self,
        input_wasm: &'wasm [u8],
    ) -> Result<Box<dyn Iterator<Item = Result<Vec<u8>>> + 'a>> {
        self.setup(input_wasm)?;

        const MUTATORS: &[&dyn Mutator] = &[
            &PeepholeMutator::new(2),
            &RemoveExportMutator,
            &AddExportMutator { max_name_size: 100 },
            &RenameExportMutator { max_name_size: 100 },
            &SnipMutator,
            &CodemotionMutator,
            &FunctionBodyUnreachable,
            &AddCustomSectionMutator,
            &ReorderCustomSectionMutator,
            &CustomSectionMutator,
            &AddTypeMutator {
                max_params: 20,
                max_results: 20,
            },
            &AddFunctionMutator,
            &AddGlobalMutator,
            &RemoveSection::Custom,
            &RemoveSection::Empty,
            &ConstExpressionMutator::Global,
            &ConstExpressionMutator::ElementOffset,
            &ConstExpressionMutator::ElementFunc,
            &RemoveItemMutator(Item::Function),
            &RemoveItemMutator(Item::Global),
            &RemoveItemMutator(Item::Memory),
            &RemoveItemMutator(Item::Table),
            &RemoveItemMutator(Item::Type),
            &RemoveItemMutator(Item::Data),
            &RemoveItemMutator(Item::Element),
            &RemoveItemMutator(Item::Tag),
            &ModifyDataMutator {
                max_data_size: 10 << 20, // 10MB
            },
            &RemoveStartSection,
        ];

        let is_named = |mutator: &&dyn Mutator, name: &str| {
            mutator.name().split("::").any(|segment| segment == name)
        };
        for name in &self.skip_mutators {
            if !MUTATORS.iter().any(|mutator| is_named(mutator, name)) {
                return Err(Error::other(format!("`{name}` does not name a mutator")));
            }
        }
        let mutators: Vec<&dyn Mutator> = MUTATORS
            .iter()
            .copied()
            .filter(|mutator| {
                !self
                    .skip_mutators
                    .iter()
                    .any(|name| is_named(mutator, name))
            })
            .collect();
        if mutators.is_empty() {
            return Err(Error::no_mutations_applicable());
        }

        // Attempt all mutators, but start at an arbitrary index.
        let start = self.rng().random_range(0..mutators.len());
        for m in mutators.iter().cycle().skip(start).take(mutators.len()) {
            let can_mutate = m.can_mutate(self);
            log::trace!("Can `{}` mutate? {}", m.name(), can_mutate);
            if !can_mutate {
                continue;
            }
            log::debug!("attempting to mutate with `{}`", m.name());
            match m.mutate(self) {
                Ok(iter) => {
                    log::debug!("mutator `{}` succeeded", m.name());
                    return Ok(Box::new(iter.into_iter().map(|r| r.map(|m| m.finish()))));
                }
                Err(e) => {
                    log::debug!("mutator `{}` failed: {}", m.name(), e);
                    return Err(e);
                }
            }
        }

        Err(Error::no_mutations_applicable())
    }

    fn setup(&mut self, input_wasm: &'wasm [u8]) -> Result<()> {
        self.info = Some(ModuleInfo::new(input_wasm)?);
        self.rng = Some(SmallRng::seed_from_u64(self.seed));
        Ok(())
    }

    pub(crate) fn rng(&mut self) -> &mut SmallRng {
        self.rng.as_mut().unwrap()
    }

    pub(crate) fn info(&self) -> &ModuleInfo<'wasm> {
        self.info.as_ref().unwrap()
    }

    fn raw_mutate(&mut self, data: &mut Vec<u8>, max_size: usize) -> Result<()> {
        // If a raw mutation function is configured then that's prioritized.
        if let Some(mutate) = &self.raw_mutate_func {
            return mutate(data, max_size);
        }

        // If no raw mutation function is configured then we apply a naive
        // default heuristic. For now that heuristic is to simply replace a
        // subslice of data with a random slice of other data.
        //
        // First up start/end indices are picked.
        let a = self.rng().random_range(0..=data.len());
        let b = self.rng().random_range(0..=data.len());
        let start = a.min(b);
        let end = a.max(b);

        // Next a length of the replacement is chosen. Note that the replacement
        // is always smaller than the input if reduction is requested, otherwise
        // we choose some arbitrary length of bytes to insert.
        let max_size = if self.reduce || self.rng().random() {
            0
        } else {
            max_size
        };
        let len = self
            .rng()
            .random_range(0..=end - start + max_size.saturating_sub(data.len()));

        // With parameters chosen the `Vec::splice` method is used to replace
        // the data in the input.
        data.splice(start..end, self.rng().random_iter().take(len));

        Ok(())
    }
}

#[cfg(test)]
mod skip_mutator_tests {
    use super::WasmMutate;

    fn has_type_section(bytes: &[u8]) -> bool {
        wasmparser::Parser::new(0)
            .parse_all(bytes)
            .any(|payload| matches!(payload, Ok(wasmparser::Payload::TypeSection(_))))
    }

    fn mutate_empty_module(seed: u64, skipped: &[&str]) -> Option<Vec<u8>> {
        let wasm = wat::parse_str("(module)").unwrap();
        let mut config = WasmMutate::default();
        config
            .seed(seed)
            .preserve_semantics(true)
            .no_new_globals(true);
        for name in skipped {
            config.skip_mutator(*name);
        }
        let mut mutations = config.run(&wasm).ok()?;
        let first = mutations.next()?;
        first.ok()
    }

    #[test]
    fn skip_mutator_rejects_an_unknown_name() {
        let wasm = wat::parse_str("(module)").unwrap();
        let mut config = WasmMutate::default();
        config.skip_mutator("nonsense");
        assert!(config.run(&wasm).is_err());
    }

    #[test]
    fn skip_mutator_matches_path_segments() {
        let skipped = ["custom", "remove_section", "remove_item"];
        let mut applied = 0;
        for seed in 0..100 {
            if let Some(bytes) = mutate_empty_module(seed, &skipped) {
                assert!(has_type_section(&bytes));
                applied += 1;
            }
        }
        assert!(applied > 0);
        assert!(
            (0..100)
                .filter_map(|seed| mutate_empty_module(seed, &[]))
                .any(|bytes| !has_type_section(&bytes))
        );
    }
}

#[cfg(test)]
pub(crate) fn validate(bytes: &[u8]) {
    use wasmparser::WasmFeatures;

    let mut validator = wasmparser::Validator::new_with_features(
        WasmFeatures::default() | WasmFeatures::MEMORY64 | WasmFeatures::MULTI_MEMORY,
    );
    let err = match validator.validate_all(bytes) {
        Ok(_) => return,
        Err(e) => e,
    };
    drop(std::fs::write("test.wasm", &bytes));
    if let Ok(text) = wasmprinter::print_bytes(bytes) {
        drop(std::fs::write("test.wat", &text));
    }

    panic!("wasm failed to validate: {err} (written to test.wasm)");
}
