import type { AceRequest, AceHealth } from './types.js';
import { FETCH_TIMEOUT_MS } from './config.js';

// POST lm: partial request -> enriched request(s)
export async function lmGenerate(req: AceRequest): Promise<AceRequest[]> {
	const res = await fetch('lm', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(req)
	});
	if (res.status === 503) throw new Error('Server busy');
	if (!res.ok) {
		const err = await res.json().catch(() => ({ error: res.statusText }));
		throw new Error(err.error || res.statusText);
	}
	return res.json();
}

// POST synth[?wav=1]: enriched request -> audio blob(s) + headers
export interface SynthResult {
	audio: Blob;
	seed: number;
	duration: number;
	computeMs: number;
}

export async function synthGenerate(reqs: AceRequest[], format: string): Promise<SynthResult[]> {
	const url = format === 'wav' ? 'synth?wav=1' : 'synth';
	const body = reqs.length === 1 ? JSON.stringify(reqs[0]) : JSON.stringify(reqs);
	const res = await fetch(url, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body
	});
	if (res.status === 503) throw new Error('Server busy');
	if (!res.ok) {
		const err = await res.json().catch(() => ({ error: res.statusText }));
		throw new Error(err.error || res.statusText);
	}

	const ct = res.headers.get('Content-Type') || '';

	// single track: headers on the response itself
	if (!ct.startsWith('multipart/')) {
		return [
			{
				audio: await res.blob(),
				seed: Number(res.headers.get('X-Seed') || 0),
				duration: Number(res.headers.get('X-Duration') || 0),
				computeMs: Number(res.headers.get('X-Compute-Ms') || 0)
			}
		];
	}

	// batch: multipart/mixed with per-part headers
	const match = ct.match(/boundary=([^\s;]+)/);
	if (!match) throw new Error('Missing boundary in multipart response');
	const mime = format === 'wav' ? 'audio/wav' : 'audio/mpeg';
	return parseMultipart(new Uint8Array(await res.arrayBuffer()), match[1], mime);
}

// parse a header value from a block of MIME headers
function getHeader(headers: string, name: string): string {
	for (const line of headers.split('\r\n')) {
		const colon = line.indexOf(':');
		if (colon < 0) continue;
		if (line.substring(0, colon).trim().toLowerCase() === name.toLowerCase()) {
			return line.substring(colon + 1).trim();
		}
	}
	return '';
}

// parse multipart/mixed binary response into SynthResult[].
// each part has text headers (X-Seed, X-Duration, X-Compute-Ms) and a binary body.
function parseMultipart(buf: Uint8Array, boundary: string, mime: string): SynthResult[] {
	const enc = new TextEncoder();
	const dec = new TextDecoder();
	const delim = enc.encode('--' + boundary);
	const results: SynthResult[] = [];

	// find all boundary positions
	const positions: number[] = [];
	for (let i = 0; i <= buf.length - delim.length; i++) {
		let ok = true;
		for (let j = 0; j < delim.length; j++) {
			if (buf[i + j] !== delim[j]) {
				ok = false;
				break;
			}
		}
		if (ok) positions.push(i);
	}

	// each consecutive pair of boundaries delimits one part.
	// after the boundary marker: \r\n (part) or -- (end sentinel).
	for (let p = 0; p < positions.length - 1; p++) {
		// skip "--boundary\r\n" to get to part content
		const partStart = positions[p] + delim.length + 2;
		// part ends at "\r\n" before next boundary
		const partEnd = positions[p + 1] - 2;
		if (partStart >= partEnd) continue;

		// split headers from body at \r\n\r\n
		let splitAt = -1;
		for (let i = partStart; i < partEnd - 3; i++) {
			if (buf[i] === 13 && buf[i + 1] === 10 && buf[i + 2] === 13 && buf[i + 3] === 10) {
				splitAt = i;
				break;
			}
		}
		if (splitAt < 0) continue;

		const headers = dec.decode(buf.slice(partStart, splitAt));
		const body = buf.slice(splitAt + 4, partEnd);

		results.push({
			audio: new Blob([body], { type: mime }),
			seed: Number(getHeader(headers, 'X-Seed') || 0),
			duration: Number(getHeader(headers, 'X-Duration') || 0),
			computeMs: Number(getHeader(headers, 'X-Compute-Ms') || 0)
		});
	}

	return results;
}

// GET props: server config, pipeline status, default request (2s timeout)
export async function props(): Promise<AceHealth> {
	const res = await fetch('props', {
		signal: AbortSignal.timeout(FETCH_TIMEOUT_MS)
	});
	if (!res.ok) throw new Error('Server unreachable');
	return res.json();
}
