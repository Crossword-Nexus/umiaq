/* tslint:disable */
/* eslint-disable */
export function init(): void;
/**
 * Parse a newline-separated word list string into a `WordList`.
 *
 * Each line of the input should be in the `word;score` format.
 * Words with a score below `min_score` are filtered out.
 * Returns the surviving entries as a `JsValue` array of strings,
 * suitable for consumption in JavaScript.
 *
 * # Errors
 * Returns a `JsValue` error if parsing fails (e.g. malformed input).
 */
export function parse_word_list(text: string, min_score: number): any;
/**
 * JS entry: (input: string, word_list: string[], num_results_requested: number)
 * returns Array<Array<string>> — only the bound words
 */
export function solve_equation_wasm(input: string, word_list: any, num_results_requested: number): any;
/**
 * Generate a debug report for troubleshooting.
 *
 * This function creates a formatted debug report that users can copy/paste
 * when reporting issues. It includes the error message, input pattern,
 * configuration details, and environment information.
 *
 * # Arguments
 * * `input_pattern` - The equation pattern that was being solved
 * * `error_message` - The error message that was displayed
 * * `word_list_size` - Number of words in the word list
 * * `num_results_requested` - How many results were requested
 *
 * # Returns
 * A formatted string containing all debug information
 */
export function get_debug_info(input_pattern: string, error_message: string, word_list_size: number, num_results_requested: number): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly get_debug_info: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
  readonly init: () => void;
  readonly parse_word_list: (a: number, b: number, c: number) => [number, number, number];
  readonly solve_equation_wasm: (a: number, b: number, c: any, d: number) => [number, number, number];
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_export_4: WebAssembly.Table;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
