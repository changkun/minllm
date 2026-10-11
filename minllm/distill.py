"""
Distillation: a small model, the student, learns from the predictions of a
larger one, the teacher, and not from the data alone (Hinton, Vinyals and
Dean, 2015, arXiv:1503.02531).

The data says which token came next. The teacher says how likely every token
was to come next, which is more to learn from at each position: after "Once
upon a time, there was a little", the text continues with one word, and the
teacher knows that "girl", "boy" and "bird" were all good guesses. See
docs/13-distillation.md.
"""

import torch.nn.functional as F


def soft_loss(student_logits, teacher_logits, temperature=1.0):
    """
    How far the student's distribution over the next token is from the
    teacher's: their KL divergence, averaged over the positions. It is zero
    when the two agree. A temperature above 1 flattens both distributions
    first, which gives the tokens that the teacher finds unlikely, but not
    impossible, more say. The factor T^2 keeps the size of the gradients the
    same at every temperature.
    """
    student = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    return (teacher.exp() * (teacher - student)).sum(-1).mean() * temperature**2
