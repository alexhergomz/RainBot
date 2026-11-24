"""The autoregressive action head: rollout (one decision at a time, carrying state across tokens)
must give exactly the teacher-forced logits of the decisions it chose, resets included."""
import torch

from rainbot.nn.action_head import ActionHead
from rainbot.nn.agent import sample


def test_step_matches_teacher_forcing():
    torch.manual_seed(0)
    B, T, K, d = 3, 7, 4, 64
    head = ActionHead(d)
    logit_layer = torch.nn.Linear(d, 12)
    slots = torch.randn(B, T, K, d)
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[:, 0] = True
    reset[1, 3] = True
    reset[2, 5] = True

    st = head.initial_state(B)
    lgs, acts = [], []
    for t in range(T):
        def choose(x):
            lg = logit_layer(x)
            return lg, sample(lg)
        lg, a, st = head.step(slots[:, t:t + 1], st, reset[:, t:t + 1], choose)
        lgs.append(lg)
        acts.append(a)
    lg_roll, act = torch.cat(lgs, 1), torch.cat(acts, 1)

    # teacher-forced in two calls, carrying the state between them
    st0 = head.initial_state(B)
    x1, s1 = head(slots[:, :4], act[:, :4], st0, reset[:, :4])
    x2, _ = head(slots[:, 4:], act[:, 4:], s1, reset[:, 4:])
    lg_tf = logit_layer(torch.cat([x1, x2], 1))
    assert torch.allclose(lg_roll, lg_tf, atol=1e-5), (lg_roll - lg_tf).abs().max()


def test_reset_isolates_episodes():
    torch.manual_seed(1)
    head = ActionHead(64)
    slots = torch.randn(1, 4, 4, 64)
    act = torch.randint(0, 2, (1, 4, 4, 5))
    reset = torch.tensor([[True, False, True, False]])
    x, _ = head(slots, act, head.initial_state(1), reset)
    act2 = act.clone()
    act2[:, :2] = 1 - act2[:, :2]                               # change only the first episode
    slots2 = slots.clone()
    slots2[:, :2] += 1.0
    y, _ = head(slots2, act2, head.initial_state(1), reset)
    assert torch.allclose(x[:, 2:], y[:, 2:], atol=1e-6)
