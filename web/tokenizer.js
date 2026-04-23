// SuomiChat tokenizer (browser side).
//
// Two halves:
//   • DECODE:  id → bytes (concatenate, then UTF-8 decode at the end so
//              bytes that split a multibyte char rejoin cleanly).
//   • ENCODE:  text → ids via GPT-4-style split regex + tiktoken-compatible
//              BPE merging. Specials handled by splitting on their literal
//              text first, then encoding the gaps as ordinary text.
//
// All merge ranks come from `tokenizer.json` (`vocab_b64[id]` for id <
// `n_merges` is the byte sequence for that merge; `id` IS the rank).
// We build a Map<string, int> keyed by stringified bytes for O(1) lookup
// during the BPE inner loop.

export async function loadTokenizer(url = "./tokenizer.json") {
  const t = await (await fetch(url)).json();
  // Decode base64 once into a Uint8Array per token id.
  const vocabBytes = t.vocab_b64.map(b64 => {
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return arr;
  });
  // Build name -> id and id -> name maps for special tokens
  const specialNameById = new Map();
  for (const [name, id] of Object.entries(t.special_tokens)) specialNameById.set(id, name);

  // ranksByKey: stringified-bytes → rank (= token id) for non-special tokens.
  // The key is the byte sequence joined by commas: O(n) to build, O(1) to lookup.
  const ranksByKey = new Map();
  const nMerges = t.n_merges ?? t.n_vocab;
  for (let id = 0; id < nMerges; id++) {
    ranksByKey.set(_bytesKey(vocabBytes[id]), id);
  }

  // Convert tiktoken's Python-style pat_str to a JS-compatible regex.
  // - Possessive quantifiers (?+, ++) become greedy (?, +) — equivalent
  //   here because the surrounding character classes have no overlap.
  // - The (?i:...) inline-case-insensitive group is hand-expanded for
  //   the four English-contraction alternatives (the only place tiktoken
  //   patterns use it). \p{L}, \p{N} need the /u flag.
  const PAT = /'(?:[sdmtSDMT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+/gu;

  return {
    nVocab: t.n_vocab,
    nMerges,
    bosId: t.bos_id,
    eotId: t.eot_id,
    specialTokens: t.special_tokens,
    specialNameById,
    vocabBytes,
    ranksByKey,
    pattern: PAT,
  };
}

// ---- BPE encode --------------------------------------------------------

function _bytesKey(bytes) {
  // Comma-joined u8 ints. Chosen over String.fromCharCode because raw
  // ASCII bytes mixed with high bytes get fiddly with charCode/atob.
  return bytes.join(",");
}

function _concat(a, b) {
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

// BPE inner loop — naive O(n^2) per chunk. Chunks are word-sized so this
// is fine for the personal-learning use case.
function _bpeChunk(bytes, ranksByKey) {
  if (bytes.length <= 1) {
    const id = ranksByKey.get(_bytesKey(bytes));
    if (id === undefined) throw new Error("BPE: unknown 1-byte token (vocab leak?)");
    return [id];
  }
  // Initialize as one piece per byte.
  const parts = [];
  for (let i = 0; i < bytes.length; i++) parts.push(bytes.subarray(i, i + 1));

  while (parts.length > 1) {
    let bestRank = Infinity, bestIdx = -1, bestMerged = null;
    for (let i = 0; i < parts.length - 1; i++) {
      const merged = _concat(parts[i], parts[i + 1]);
      const r = ranksByKey.get(_bytesKey(merged));
      if (r !== undefined && r < bestRank) {
        bestRank = r; bestIdx = i; bestMerged = merged;
      }
    }
    if (bestIdx === -1) break;   // no more applicable merges
    parts.splice(bestIdx, 2, bestMerged);
  }

  return parts.map(p => {
    const id = ranksByKey.get(_bytesKey(p));
    if (id === undefined) throw new Error("BPE: leftover piece has no id");
    return id;
  });
}

// Encode plain text — equivalent to tiktoken Encoding.encode_ordinary.
// Special tokens in the input are encoded literally (NOT as their special
// id). Use renderForCompletion / explicit special-token IDs to inject specials.
export function encodeOrdinary(tokenizer, text) {
  const enc = new TextEncoder();
  const ids = [];
  // matchAll requires the /g flag on the regex — already set in pattern.
  for (const m of text.matchAll(tokenizer.pattern)) {
    const chunkBytes = enc.encode(m[0]);
    for (const id of _bpeChunk(chunkBytes, tokenizer.ranksByKey)) ids.push(id);
  }
  return ids;
}

// Render a chat for completion: same as PyTorch's render_for_completion.
//   messages: [{role: "user"|"assistant", content: string}, ...]
// Last message MUST be user (assistant is implicit and we'll prime with
// `<|assistant_start|>`). Returns flat token-id array.
export function renderForCompletion(tokenizer, messages) {
  if (messages.length === 0) throw new Error("renderForCompletion: empty");
  if (messages[messages.length - 1].role !== "user") {
    throw new Error("renderForCompletion: last message must be 'user'");
  }
  const sp = tokenizer.specialTokens;
  const out = [sp["<|bos|>"]];
  for (const m of messages) {
    if (m.role === "user") {
      out.push(sp["<|user_start|>"]);
      for (const id of encodeOrdinary(tokenizer, m.content)) out.push(id);
      out.push(sp["<|user_end|>"]);
    } else if (m.role === "assistant") {
      out.push(sp["<|assistant_start|>"]);
      for (const id of encodeOrdinary(tokenizer, m.content)) out.push(id);
      out.push(sp["<|assistant_end|>"]);
    } else {
      throw new Error("renderForCompletion: unknown role " + m.role);
    }
  }
  out.push(sp["<|assistant_start|>"]);  // prime assistant
  return out;
}

// Decode a list of token ids to a string. Handles special tokens by
// substituting their name; other tokens contribute their raw bytes.
// We accumulate bytes between special tokens so multi-byte UTF-8
// characters that span token boundaries decode correctly.
export function decodeTokens(tokenizer, ids, { skipSpecial = false } = {}) {
  const decoder = new TextDecoder("utf-8", { fatal: false });
  let parts = [];
  let buf = [];
  const flush = () => {
    if (buf.length === 0) return;
    parts.push(decoder.decode(new Uint8Array(buf)));
    buf = [];
  };
  for (const id of ids) {
    const specName = tokenizer.specialNameById.get(id);
    if (specName !== undefined) {
      flush();
      if (!skipSpecial) parts.push(specName);
      continue;
    }
    const b = tokenizer.vocabBytes[id];
    if (!b) continue;   // unknown id — skip silently
    for (let i = 0; i < b.length; i++) buf.push(b[i]);
  }
  flush();
  return parts.join("");
}
