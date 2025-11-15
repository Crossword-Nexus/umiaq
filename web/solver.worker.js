import init, { solve_equation_wasm } from './pkg/umiaq.js';

let ready = (async () => { await init(); })();

self.onmessage = async (e) => {
    const { type, input, entryList: entryList, numResults } = e.data;
    if (type === 'init') {
        await ready;
        self.postMessage({ type: 'ready' });
        return;
    }
    if (type === 'solve') {
        await ready;
        try {
            const out = solve_equation_wasm(input, entryList, numResults);
            self.postMessage({ type: 'ok', results: out });
        } catch (err) {
            self.postMessage({ type: 'err', error: String(err) });
        }
    }
};
