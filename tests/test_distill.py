import torch
import torch.nn.functional as F

from minllm import distill


def test_zero_when_student_and_teacher_agree():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 16)
    assert distill.soft_loss(logits, logits).abs() < 1e-6
    assert distill.soft_loss(logits, logits + 3.0).abs() < 1e-6  # a shift of all logits changes no probability
    assert distill.soft_loss(torch.randn(2, 5, 16), logits) > 0.1


def test_a_certain_teacher_is_a_label():
    # A teacher that puts all its probability on one token teaches what the
    # data's label teaches: the loss is the cross-entropy against that token.
    torch.manual_seed(0)
    student, labels = torch.randn(6, 16), torch.randint(0, 16, (6,))
    teacher = F.one_hot(labels, 16) * 60.0
    assert torch.allclose(distill.soft_loss(student, teacher), F.cross_entropy(student, labels), atol=1e-4)


def test_temperature():
    torch.manual_seed(0)
    student, teacher = torch.randn(4, 16, requires_grad=True), torch.randn(4, 16) * 3
    # the definition: the KL divergence of the softened distributions, times T^2
    soft = lambda x: F.softmax(x / 2.0, dim=-1)
    expected = (soft(teacher) * (soft(teacher) / soft(student)).log()).sum(-1).mean() * 4.0
    assert torch.allclose(distill.soft_loss(student, teacher, temperature=2.0), expected, atol=1e-5)
    # Following the gradient brings the student to the teacher.
    logits = torch.zeros(4, 16, requires_grad=True)
    for _ in range(300):
        loss = distill.soft_loss(logits, teacher)
        (grad,) = torch.autograd.grad(loss, logits)
        logits = (logits - 20 * grad).detach().requires_grad_()
    assert torch.allclose(F.softmax(logits, -1), F.softmax(teacher, -1), atol=0.01)
