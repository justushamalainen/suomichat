// SuomiChat tokenizer (browser side, decode-only Phase 14a).
//
// Phase 14a covers DECODE only — Phase 14b will add the BPE encoder so
// users can type arbitrary prompts. For now the chat UI offers a fixed
// set of demo prompts whose token IDs were computed offline by PyTorch
// and embedded as constants in index.html.
//
// Token IDs map to raw bytes (UTF-8 fragments). We concatenate those
// bytes and decode as UTF-8 — bytes can split across token boundaries,
// so we accumulate then decode the whole string at the end.

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

  return {
    nVocab: t.n_vocab,
    bosId: t.bos_id,
    eotId: t.eot_id,
    specialTokens: t.special_tokens,
    specialNameById,
    vocabBytes,
  };
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
