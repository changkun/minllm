// The teacher's prediction at a temperature: what distillation shows the
// student when the distributions are flattened.

import sampling from "./sampling.js";

export default (el) => sampling(el, { source: "/runs/distill/next-token.json", knobs: false });
