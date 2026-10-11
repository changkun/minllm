"""
Self-training: the model answers each prompt several times, a program keeps
the answers that are right, and the model is fine-tuned on what was kept.
Then the better model answers again.

It is the best-of-N selection of stage 8 with the verifier of stage 9 doing
the judging, followed by the fine-tuning of stage 6 on the winners. Nothing
of reinforcement learning is needed: no advantage, and no gradient through
the rewards. See docs/09a-self-training.md.
"""


def keep(texts, scores, finished, samples, per_prompt=2):
    """
    Which replies to learn from. texts holds `samples` replies for each prompt,
    one prompt after the other. A reply is kept if the model ended it itself
    and it is entirely right, a score of 1. Of the replies to one prompt, the
    same text counts once and at most per_prompt are kept, so that the prompts
    the model finds easy do not crowd out the others. Returns their positions.
    """
    kept = []
    for start in range(0, len(texts), samples):
        seen = set()
        for i in range(start, start + samples):
            if finished[i] and scores[i] == 1 and texts[i] not in seen and len(seen) < per_prompt:
                seen.add(texts[i])
                kept.append(i)
    return kept
