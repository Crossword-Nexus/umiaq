use wasm_bindgen::prelude::*;
use crate::bindings::Bindings;
use crate::errors::ParseError;
use crate::solver::solve_equation;
use crate::wordlist::WordList;

use serde_wasm_bindgen::to_value;

/// Implement `Box<ParseError>` for `JsValue`s
impl From<Box<ParseError>> for JsValue {
    fn from(e: Box<ParseError>) -> JsValue { JsValue::from_str(format!("parse error: {}", *e).as_str()) }
}

#[wasm_bindgen(start)]
fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

// Pull just the bound word ("*") out of a Bindings
fn binding_to_word(b: &Bindings) -> Option<String> {
    b.get_word().cloned()
}

/// JS entry: (input: string, word_list: string[], num_results_requested: number)
/// returns Array<Array<string>> — only the bound words
#[wasm_bindgen]
pub fn solve_equation_wasm(
    input: &str,
    word_list: JsValue,
    num_results_requested: usize,
) -> Result<JsValue, JsValue> {
    // word_list: string[] -> Vec<String>
    let words: Vec<String> = serde_wasm_bindgen::from_value(word_list)
        .map_err(|e| JsValue::from_str(&format!("word_list must be string[]: {e}")))?;
    // Borrow as &[&str] for the solver
    let refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();

    let raw = solve_equation(input, &refs, num_results_requested)?; // Vec<Vec<Bindings>>

    // Keep only the "*" word from each Bindings
    let js_ready: Vec<Vec<String>> = raw
        .into_iter()
        .map(|row| row.into_iter().filter_map(|b| binding_to_word(&b)).collect())
        .collect();

    serde_wasm_bindgen::to_value(&js_ready)
        .map_err(|e| JsValue::from_str(&format!("serialization failed: {e}")))
}

#[wasm_bindgen]
pub fn parse_wordlist(text: &str, min_score: i32) -> JsValue {
    let wl = WordList::parse_from_str(text, min_score);
    // Convert Vec<String> to a real JS array
    to_value(&wl.entries).expect("serde_wasm_bindgen conversion failed")
}
