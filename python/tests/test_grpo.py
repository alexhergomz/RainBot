"""GRPO helpers: group advantages with refills, and per-member temperatures."""
import numpy as np
import torch
import torch.nn.functional as F

from rainbot.nn.agent import max_entropy, split_logits
from rainbot.train.rollout import Member, fit_tau_max, group_advantages, member_tau


def _return_to_go(r, gamma):
    g, out = 0.0, []
    for x in reversed(r):
        g = x + gamma * g
        out.append(g)
    return np.array(out[::-1])


def test_refill_members_align_by_their_own_step():
    """A member that starts later in the pass is compared with the others at its own episode step,
    and a member cut by the budget is left out."""
    L, N = 10, 2
    rew = np.zeros((N, L), np.float32)
    # a: game 0, steps 0-3; refill b: game 0 from step 5, 3 steps; c: game 1, whole pass;
    # refill d: game 0 from step 9, cut by the budget.
    rew[0, 0:4] = [1, 0, 0, 2]
    rew[0, 5:8] = [1, 0, 0]
    rew[1, :] = 0.5
    rew[0, 9] = 100.0
    members = [Member(0, s=0, n=4), Member(1, s=0, n=10), Member(0, s=5, n=3), Member(0, s=9, n=1, cut=True)]
    adv = np.zeros((N, L), np.float32)
    group_advantages(rew, [(0, L, members)], 0.9, adv)

    Ga = _return_to_go(np.r_[[1, 0, 0, 2], np.zeros(6)], 0.9)
    Gb = _return_to_go(np.r_[[1, 0, 0], np.zeros(7)], 0.9)
    Gc = _return_to_go(np.full(10, 0.5), 0.9)
    allG = np.stack([Ga, Gc, Gb])
    mu, sd = allG.mean(0), allG.std(0)
    assert np.allclose(adv[0, 5:8], ((Gb - mu) / (sd + 1e-6))[:3], atol=1e-5)   # refill b at its steps 0-2
    assert np.allclose(adv[0, 0:4], ((Ga - mu) / (sd + 1e-6))[:4], atol=1e-5)
    assert adv[0, 9] == 0.0                                                     # the cut refill: no advantage


def test_member_temperatures_span_one_to_the_fitted_maximum():
    """The hottest member samples at the entropy floor; the coldest is the policy itself."""
    logits = (torch.randn(4, 50, 4, 12) * 8.0).numpy().astype(np.float16)      # a confident policy
    target = 0.30 * max_entropy()
    tau_max = fit_tau_max(logits, np.ones((4, 50), bool), target)
    assert tau_max > 1.0
    x = torch.as_tensor(logits.reshape(-1, 4, 12), dtype=torch.float32) / tau_max
    e = sum(-(F.log_softmax(p, -1).exp() * F.log_softmax(p, -1)).sum(-1).sum(-1) for p in split_logits(x)).mean()
    assert abs(float(e) / max_entropy() - 0.30) < 0.02
    taus = [member_tau(tau_max, k, 4) for k in range(4)]
    assert taus[0] == 1.0 and abs(taus[-1] - tau_max) < 1e-9
    assert all(a < b for a, b in zip(taus, taus[1:]))
