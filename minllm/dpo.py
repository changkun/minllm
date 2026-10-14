"""
Direct preference optimization (Rafailov et al., 2023, arXiv:2305.18290):
the policy is trained on the comparisons themselves, with no reward model
and no sampling during training.

Every reply is scored by how much likelier the policy has made it than the
model it started from, its gain:

    gain(reply) = log p(reply) - log p_start(reply)

and the loss asks the preferred reply of a pair to gain more than the
other. The comparisons can come from a person, from the simulated
annotator of stage 8, or from a program that checks the answer, as in
docs/08b-dpo.md. See docs/08-rlhf.md.
"""

import torch.nn.functional as F


def loss(gain_preferred, gain_other, beta):
    """
    -log sigmoid(beta * (gain_preferred - gain_other)), averaged over the
    pairs. beta says how far the policy may move from the start before the
    loss stops pulling: the larger, the sooner a difference counts as large.
    Only the difference enters. Both gains may fall, as long as the other
    reply's falls faster.
    """
    return -F.logsigmoid(beta * (gain_preferred - gain_other)).mean()


def pairs(texts, right, samples, per_prompt=2):
    """
    Comparisons made by a verifier instead of a person. texts holds `samples`
    replies for each prompt, one prompt after the other, and right says for
    each whether it is right. A prompt with both right and wrong replies
    gives up to per_prompt pairs of a right reply and a wrong one, no text
    used twice. A prompt whose replies are all right, or all wrong, gives
    none: there is nothing to prefer. Returns (preferred, other) positions.
    """
    out = []
    for start in range(0, len(texts), samples):
        good, bad, seen = [], [], set()
        for i in range(start, start + samples):
            if texts[i] not in seen:
                seen.add(texts[i])
                (good if right[i] else bad).append(i)
        out += list(zip(good, bad))[:per_prompt]
    return out
