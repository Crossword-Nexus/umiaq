import init, { solve_equation_wasm, initialize } from './pkg/umiaq.js';

self.onmessage = async (e) => {
    const { type, input, entryList: entryList, numResults, debugEnabled } = e.data;
    if (type === 'init') {
        // Load WASM module then initialize logging with debug setting
        await init();
        initialize(debugEnabled);
        self.postMessage({ type: 'ready' });
        return;
    }
    if (type === 'solve') {
        // No need to await ready - main thread won't send 'solve' until after receiving 'ready'
        try {
            const out = solve_equation_wasm(input, entryList, numResults);
            self.postMessage({ type: 'ok', results: out });
        } catch (err) {
            self.postMessage({ type: 'err', error: String(err) });
        }
    }
};
