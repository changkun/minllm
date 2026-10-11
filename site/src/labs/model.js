// The model of stage 2 with the cache of stage 4, for one sequence and
// without a library: what minllm/model.py and minllm/inference.py compute,
// one token at a time. The weights are those of a quantized model, as
// minllm/quantize.py stores them.

import { toHalf } from "./safetensors.js";

// The matrix that packed codes stand for: weight = code * scale + offset,
// with a scale and an offset for every group of weights in a row. A 32-bit
// word holds 32 / bits codes, the first in its lowest bits. The arithmetic is
// in half precision, as in quantize.dequantize, so the weights are the same
// numbers as there.
export function expand(tensors, name, bits) {
  const words = tensors[`${name}.words`];
  const scale = tensors[`${name}.scale`].data;
  const offset = tensors[`${name}.offset`].data;
  const [rows, wordsPerRow] = words.shape;
  const perWord = 32 / bits;
  const cols = wordsPerRow * perWord;
  const group = cols / tensors[`${name}.scale`].shape[1];
  const weights = new Float32Array(rows * cols);
  const values = new Float32Array(1 << bits); // what each code stands for in the current group
  for (let r = 0, g = -1; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (c % group === 0) {
        g++;
        for (let code = 0; code < values.length; code++) values[code] = toHalf(Math.fround(toHalf(code * scale[g]) + offset[g]));
      }
      const word = words.data[r * wordsPerRow + Math.floor(c / perWord)];
      weights[r * cols + c] = values[(word >>> (bits * (c % perWord))) & (values.length - 1)];
    }
  }
  return { rows, cols, weights };
}

// y = W x
function apply({ rows, cols, weights }, x, y) {
  for (let r = 0, at = 0; r < rows; r++, at += cols) {
    let a = 0, b = 0, c = 0, d = 0; // four sums side by side: the columns are a multiple of four
    for (let j = 0; j < cols; j += 4) {
      a += weights[at + j] * x[j];
      b += weights[at + j + 1] * x[j + 1];
      c += weights[at + j + 2] * x[j + 2];
      d += weights[at + j + 3] * x[j + 3];
    }
    y[r] = a + b + c + d;
  }
  return y;
}

export class Model {
  // tensors: a quantized model's file, read by safetensors.js. meta: its meta.json.
  constructor(tensors, meta) {
    const cfg = (this.cfg = meta.config);
    const bits = meta.quant.bits;
    const matrix = (name) => expand(tensors, name, bits);
    this.embed = matrix("embed"); // also the output layer: the two are tied
    this.layers = Array.from({ length: cfg.n_layers }, (_, i) => ({
      attnNorm: tensors[`layers.${i}.attn_norm.weight`].data,
      q: matrix(`layers.${i}.attn.q_proj`),
      k: matrix(`layers.${i}.attn.k_proj`),
      v: matrix(`layers.${i}.attn.v_proj`),
      o: matrix(`layers.${i}.attn.o_proj`),
      mlpNorm: tensors[`layers.${i}.mlp_norm.weight`].data,
      gate: matrix(`layers.${i}.mlp.gate_proj`),
      up: matrix(`layers.${i}.mlp.up_proj`),
      down: matrix(`layers.${i}.mlp.down_proj`),
    }));
    this.norm = tensors["norm.weight"].data;

    // Rotary embeddings: pair i of a head's dimensions turns by theta^(-2i / headDim) per token.
    this.headDim = cfg.dim / cfg.n_heads;
    this.half = this.headDim / 2;
    this.freqs = Float64Array.from({ length: this.half }, (_, i) => cfg.rope_theta ** ((-2 * i) / this.headDim));

    const kv = cfg.n_kv_heads * this.headDim;
    this.x = new Float32Array(cfg.dim);
    this.h = new Float32Array(cfg.dim);
    this.q = new Float32Array(cfg.dim);
    this.k = new Float32Array(kv);
    this.v = new Float32Array(kv);
    this.y = new Float32Array(cfg.dim);
    this.out = new Float32Array(cfg.dim);
    this.gate = new Float32Array(cfg.mlp_dim);
    this.up = new Float32Array(cfg.mlp_dim);
    this.scores = new Float64Array(cfg.max_seq_len);
    this.logits = new Float32Array(cfg.vocab_size);
    // The cache: the keys and values of every token so far, in every layer.
    this.keys = this.layers.map(() => new Float32Array(cfg.max_seq_len * kv));
    this.values = this.layers.map(() => new Float32Array(cfg.max_seq_len * kv));
    this.pos = 0;
  }

  // Forget the sequence.
  reset() {
    this.pos = 0;
  }

  // RMSNorm: x at unit root-mean-square, times a learned gain.
  normalize(x, gain, y) {
    let sum = 0;
    for (let i = 0; i < x.length; i++) sum += x[i] * x[i];
    const scale = 1 / Math.sqrt(sum / x.length + this.cfg.norm_eps);
    for (let i = 0; i < x.length; i++) y[i] = x[i] * scale * gain[i];
    return y;
  }

  // Turn every head's vector by the position: its two halves are the coordinates of points in the plane.
  rotate(x, heads) {
    const { half, headDim, freqs, pos } = this;
    for (let h = 0; h < heads; h++) {
      for (let i = 0; i < half; i++) {
        const cos = Math.cos(pos * freqs[i]);
        const sin = Math.sin(pos * freqs[i]);
        const a = x[h * headDim + i];
        const b = x[h * headDim + half + i];
        x[h * headDim + i] = a * cos - b * sin;
        x[h * headDim + half + i] = a * sin + b * cos;
      }
    }
  }

  // Take in the next token of the sequence. Returns the logits of the token
  // after it, unless they are not asked for: the tokens of a prompt, all but
  // the last, are only there to be attended to.
  step(id, predict = true) {
    const { cfg, headDim, x, h, q, k, v, y, out, gate, up, scores } = this;
    if (this.pos >= cfg.max_seq_len) throw new Error(`the sequence is longer than the ${cfg.max_seq_len} tokens the model was trained on`);
    const kv = cfg.n_kv_heads * headDim;
    const groups = cfg.n_heads / cfg.n_kv_heads; // query heads that share a key/value head
    x.set(this.embed.weights.subarray(id * cfg.dim, (id + 1) * cfg.dim));
    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l];
      const keys = this.keys[l];
      const values = this.values[l];
      // Attention: this token's query against the keys of all tokens so far.
      this.normalize(x, layer.attnNorm, h);
      apply(layer.q, h, q);
      apply(layer.k, h, k);
      apply(layer.v, h, v);
      this.rotate(q, cfg.n_heads);
      this.rotate(k, cfg.n_kv_heads);
      keys.set(k, this.pos * kv);
      values.set(v, this.pos * kv);
      for (let head = 0; head < cfg.n_heads; head++) {
        const shared = Math.floor(head / groups) * headDim;
        let peak = -Infinity;
        for (let t = 0; t <= this.pos; t++) {
          let dot = 0;
          for (let i = 0; i < headDim; i++) dot += q[head * headDim + i] * keys[t * kv + shared + i];
          scores[t] = dot / Math.sqrt(headDim);
          if (scores[t] > peak) peak = scores[t];
        }
        let sum = 0;
        for (let t = 0; t <= this.pos; t++) sum += scores[t] = Math.exp(scores[t] - peak);
        y.fill(0, head * headDim, (head + 1) * headDim);
        for (let t = 0; t <= this.pos; t++) {
          const weight = scores[t] / sum;
          for (let i = 0; i < headDim; i++) y[head * headDim + i] += weight * values[t * kv + shared + i];
        }
      }
      apply(layer.o, y, out);
      for (let i = 0; i < x.length; i++) x[i] += out[i];
      // The MLP, SwiGLU: one projection gates the other.
      this.normalize(x, layer.mlpNorm, h);
      apply(layer.gate, h, gate);
      apply(layer.up, h, up);
      for (let i = 0; i < gate.length; i++) gate[i] = (gate[i] / (1 + Math.exp(-gate[i]))) * up[i];
      apply(layer.down, gate, out);
      for (let i = 0; i < x.length; i++) x[i] += out[i];
    }
    this.pos++;
    return predict ? apply(this.embed, this.normalize(x, this.norm, h), this.logits) : null;
  }
}
