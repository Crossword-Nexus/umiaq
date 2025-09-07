import init, { solve_equation_wasm } from './pkg/umiaq.js';

let ready = (async () => { await init(); })();

self.onmessage = async (e) => {
    const { type, input, wordlist, numResults } = e.data;
    if (type === 'init') {
        await ready;
        self.postMessage({ type: 'ready' });
        return;
    }
    if (type === 'solve') {
        await ready;
        try {
            const { results, timed_out } = solve_equation_wasm(input, wordlist, numResults);
            if (timed_out) {
                console.log("Solver timed out.");
                self.postMessage({ type: 'err', results: "Solver timed out." });
            } else {
                self.postMessage({ type: 'ok', results: results });
            }
        } catch (err) {
            self.postMessage({ type: 'err', error: String(err) });
        }
    }
};
