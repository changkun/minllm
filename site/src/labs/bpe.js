// Byte pair encoding in the browser, with the vocabulary that stage 1
// trained: the same steps as minllm/tokenizer.py.

// The pattern that cuts text into chunks is written for Python's regex
// module. JavaScript has no possessive quantifiers and no inline flags, and
// for this pattern the plain greedy forms match the same text.
export function pattern(source) {
  return new RegExp(source.replace("(?i:", "(?:").replaceAll("?+", "?").replaceAll("++", "+"), "giu");
}

export class Tokenizer {
  constructor({ pattern: source, merges, special_tokens: special }) {
    this.pattern = pattern(source);
    this.merges = merges; // merges[i] made token 256 + i out of a pair of earlier tokens
    this.rank = new Map(merges.map(([a, b], i) => [a * 65536 + b, i]));
    this.bytes = Array.from({ length: 256 }, (_, i) => [i]);
    for (const [a, b] of merges) this.bytes.push([...this.bytes[a], ...this.bytes[b]]);
    this.special = new Map(special.map((s, i) => [s, 256 + merges.length + i]));
    this.size = 256 + merges.length + special.length;
  }

  // The merges of one chunk, as the list of token lists it goes through.
  steps(chunk) {
    let ids = [...new TextEncoder().encode(chunk)];
    const states = [ids];
    for (;;) {
      // the pair that was learned earliest merges first
      let best = Infinity;
      for (let i = 0; i + 1 < ids.length; i++) best = Math.min(best, this.rank.get(ids[i] * 65536 + ids[i + 1]) ?? Infinity);
      if (best === Infinity) return states;
      const [a, b] = this.merges[best];
      const next = [];
      for (let i = 0; i < ids.length; i++) {
        if (ids[i] === a && ids[i + 1] === b) next.push(256 + best), i++;
        else next.push(ids[i]);
      }
      states.push((ids = next));
    }
  }

  // Text to token ids, without special tokens: they are not looked for in text.
  encode(text) {
    return (text.match(this.pattern) || []).flatMap((chunk) => this.steps(chunk).at(-1));
  }

  // What a token spells. A token can end in the middle of a character, so its bytes are decoded leniently.
  text(id) {
    if (id >= 256 + this.merges.length) return [...this.special.keys()][id - 256 - this.merges.length];
    return new TextDecoder().decode(new Uint8Array(this.bytes[id]));
  }

  decode(ids) {
    return new TextDecoder().decode(new Uint8Array(ids.flatMap((id) => this.bytes[id] ?? [])));
  }
}
