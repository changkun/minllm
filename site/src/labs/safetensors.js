// A safetensors file, as minllm/pretrained.py reads it: eight bytes with the
// length of a JSON header, the header, which names every tensor with its
// type, shape and place, and then the tensors' bytes.

// The number that 16 bits of a half-precision float stand for.
export function half(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 31;
  const fraction = bits & 1023;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

// The nearest half-precision float to x, a tie going to the even one, which
// is what storing x in 16 bits does to it.
export function toHalf(x) {
  const size = Math.abs(x);
  if (size === 0 || !Number.isFinite(x)) return x;
  let exponent = Math.floor(Math.log2(size));
  if (2 ** exponent > size) exponent--;
  else if (2 ** (exponent + 1) <= size) exponent++;
  const step = 2 ** (Math.max(exponent, -14) - 10); // the distance between neighbouring halves near x
  const steps = size / step;
  let whole = Math.floor(steps);
  const rest = steps - whole;
  if (rest > 0.5 || (rest === 0.5 && whole % 2 === 1)) whole++;
  const rounded = whole * step;
  return rounded > 65504 ? Math.sign(x) * Infinity : Math.sign(x) * rounded;
}

// Every tensor of the file by name: { dtype, shape, data }. Whole numbers of
// 32 bits come as they are, half-precision floats as ordinary ones.
export function read(buffer) {
  const length = Number(new DataView(buffer).getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, length)));
  const tensors = {};
  for (const [name, { dtype, shape, data_offsets: [from, to] }] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const bytes = buffer.slice(8 + length + from, 8 + length + to); // a copy, so that it starts at a multiple of its width
    if (dtype === "U32") tensors[name] = { dtype, shape, data: new Uint32Array(bytes) };
    else if (dtype === "F16") tensors[name] = { dtype, shape, data: Float32Array.from(new Uint16Array(bytes), half) };
    else throw new Error(`${name}: tensors of type ${dtype} are not read here`);
  }
  return tensors;
}
