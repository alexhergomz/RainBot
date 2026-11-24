"""KDA and SWA: parallel forms agree with step-by-step forms, resets are exact, sinks merge exactly."""
import torch
import torch.nn.functional as F

from rainbot.nn.kda import KDA, kda_chunked, kda_recurrence
from rainbot.nn.swa import SWA, merge_sinks, windowed_attention_with_lse

torch.manual_seed(0)


def _rand_kda(B=3, T=37, H=4, K=16, V=16):
    q = F.normalize(torch.randn(B, T, H, K), dim=-1)
    k = F.normalize(torch.randn(B, T, H, K), dim=-1)
    v = torch.randn(B, T, H, V)
    g = -torch.rand(B, T, H, K) * 0.5
    beta = torch.rand(B, T, H)
    S0 = torch.randn(B, H, K, V)
    return q, k, v, g, beta, S0


def test_chunked_matches_recurrence_with_resets():
    q, k, v, g, beta, S0 = _rand_kda()
    reset = torch.zeros(3, 37, dtype=torch.bool)
    reset[0, 0] = True
    reset[1, 5] = True
    reset[1, 20] = True
    reset[2, 36] = True
    o1, S1 = kda_recurrence(q, k, v, g, beta, S0, reset, 0.25)
    for C in (4, 16):
        o2, S2 = kda_chunked(q, k, v, g, beta, S0, reset, 0.25, chunk=C)
        assert torch.allclose(o1, o2, atol=1e-4), (C, (o1 - o2).abs().max())
        assert torch.allclose(S1, S2, atol=1e-4), (C, (S1 - S2).abs().max())


def test_recurrence_matches_fla_reference():
    from fla.ops.kda.naive import naive_recurrent_kda
    q, k, v, g, beta, S0 = _rand_kda()
    reset = torch.zeros(3, 37, dtype=torch.bool)
    o1, S1 = kda_recurrence(q, k, v, g, beta, S0, reset, 0.25)
    o2, S2 = naive_recurrent_kda(q, k, v, g, beta, scale=0.25, initial_state=S0, output_final_state=True)
    assert torch.allclose(o1, o2, atol=1e-5) and torch.allclose(S1, S2, atol=1e-5)


def test_kda_layer_parallel_equals_stepwise():
    m = KDA(64, heads=2, head_dim=32)
    x = torch.randn(2, 20, 64)
    reset = torch.zeros(2, 20, dtype=torch.bool)
    reset[1, 7] = True
    y_par, st_par = m(x, None, reset)
    st = m.initial_state(2)
    ys = []
    for t in range(20):
        y, st = m(x[:, t:t + 1], st, reset[:, t:t + 1])
        ys.append(y)
    y_seq = torch.cat(ys, 1)
    assert torch.allclose(y_par, y_seq, atol=1e-4), (y_par - y_seq).abs().max()
    assert torch.allclose(st_par.S, st.S, atol=1e-4)
    assert torch.allclose(st_par.conv, st.conv, atol=1e-6)


def test_kda_reset_equals_fresh_start():
    m = KDA(64, heads=2, head_dim=32)
    x = torch.randn(1, 12, 64)
    reset = torch.zeros(1, 12, dtype=torch.bool)
    reset[0, 5] = True
    y, _ = m(x, None, reset)
    y_fresh, _ = m(x[:, 5:], None, None)
    assert torch.allclose(y[:, 5:], y_fresh, atol=1e-5)


def test_swa_parallel_equals_stepwise_and_window():
    m = SWA(64, q_heads=4, kv_heads=2, head_dim=16, window=6, sinks=2)
    torch.nn.init.normal_(m.sink_v, std=0.5)
    x = torch.randn(2, 25, 64)
    reset = torch.zeros(2, 25, dtype=torch.bool)
    reset[0, 11] = True
    # Two calls with carried state == one call == token by token.
    y_full, _ = m(x, None, reset)
    ya, st = m(x[:, :13], None, reset[:, :13])
    yb, _ = m(x[:, 13:], st, reset[:, 13:])
    assert torch.allclose(y_full, torch.cat([ya, yb], 1), atol=1e-5)
    st = m.initial_state(2)
    ys = []
    for t in range(25):
        y, st = m(x[:, t:t + 1], st, reset[:, t:t + 1])
        ys.append(y)
    assert torch.allclose(y_full, torch.cat(ys, 1), atol=1e-5)
    # Window: perturbing a token more than W-1 back does not change the output.
    x2 = x.clone()
    x2[1, 3] += 10.0
    y2, _ = m(x2, None, reset)
    assert torch.allclose(y_full[1, 3 + 6:], y2[1, 3 + 6:], atol=1e-5)
    assert not torch.allclose(y_full[1, 3 + 5], y2[1, 3 + 5], atol=1e-5)
    # Reset: row 0 after the reset equals a fresh start.
    y_fresh, _ = m(x[:1, 11:], None, None)
    assert torch.allclose(y_full[:1, 11:], y_fresh, atol=1e-5)


def test_merge_sinks_is_exact():
    B, T, H, Hkv, D, R, Wn = 2, 9, 4, 2, 8, 3, 4
    q = torch.randn(B, T, H, D)
    k = torch.randn(B, T, Hkv, D)
    v = torch.randn(B, T, Hkv, D)
    sk = torch.randn(Hkv, R, D)
    sv = torch.randn(Hkv, R, D)
    o_w, lse_w = windowed_attention_with_lse(q, k, v, Wn)
    merged = merge_sinks(o_w, lse_w, q, sk, sv)
    # Reference: sinks as always-visible keys.
    rep = H // Hkv
    keys = torch.cat([sk.transpose(0, 1).unsqueeze(0).expand(B, R, Hkv, D), k], 1).repeat_interleave(rep, 2)
    vals = torch.cat([sv.transpose(0, 1).unsqueeze(0).expand(B, R, Hkv, D), v], 1).repeat_interleave(rep, 2)
    t = torch.arange(T)
    ok = (t.view(1, T) <= t.view(T, 1)) & (t.view(T, 1) - t.view(1, T) <= Wn - 1)
    mask = torch.cat([torch.ones(T, R, dtype=torch.bool), ok], 1)
    logits = torch.einsum("bthd,bshd->bhts", q, keys).masked_fill(~mask, float("-inf"))
    ref = torch.einsum("bhts,bshd->bthd", logits.softmax(-1), vals)
    assert torch.allclose(merged, ref, atol=1e-5)


def test_kda_gradients_flow():
    m = KDA(32, heads=2, head_dim=16)
    x = torch.randn(2, 10, 32, requires_grad=True)
    y, _ = m(x)
    y.square().mean().backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n


def test_xatlu_custom_grad_matches_autograd():
    from rainbot.nn.layers import XATLU, _XATLUFn, xatlu_gate
    m = XATLU()
    with torch.no_grad():
        m.alpha.fill_(0.3)
    z = torch.randn(50, dtype=torch.float64, requires_grad=True)
    a = m.alpha.detach().double().clone().requires_grad_(True)
    ref = (z * xatlu_gate(z, a)).sum()
    gz, ga = torch.autograd.grad(ref, (z, a))
    z2 = z.detach().clone().requires_grad_(True)
    a2 = a.detach().clone().requires_grad_(True)
    out = _XATLUFn.apply(z2, a2).sum()
    gz2, ga2 = torch.autograd.grad(out, (z2, a2))
    assert torch.allclose(gz, gz2) and torch.allclose(ga, ga2)


def test_xattn_window_and_reset():
    from rainbot.nn.xattn import XAttn
    m = XAttn(32, 16, n_tokens=5, q_heads=4, kv_heads=2, head_dim=8, window=3, grid_hw=(1, 4))
    torch.nn.init.normal_(m.age, std=0.5)
    B, T = 2, 10
    x, vis = torch.randn(B, T, 32), torch.randn(B, T, 5, 16)
    valid = torch.ones(B, T, dtype=torch.bool)
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[0, 6] = True
    y, _ = m(x, vis, valid, None, reset)
    # A frame more than window-1 steps old cannot influence the output.
    vis2 = vis.clone()
    vis2[1, 2] += 5.0
    y2, _ = m(x, vis2, valid, None, reset)
    assert torch.allclose(y[1, 5:], y2[1, 5:], atol=1e-6) and not torch.allclose(y[1, 4], y2[1, 4], atol=1e-6)
    # After a reset only the new episode's frames are visible.
    vis3 = vis.clone()
    vis3[0, 5] += 5.0
    y3, _ = m(x, vis3, valid, None, reset)
    assert torch.allclose(y[0, 6:], y3[0, 6:], atol=1e-6)
    # Two calls with carried state == one call.
    ya, st = m(x[:, :4], vis[:, :4], valid[:, :4], None, reset[:, :4])
    yb, _ = m(x[:, 4:], vis[:, 4:], valid[:, 4:], st, reset[:, 4:])
    assert torch.allclose(y, torch.cat([ya, yb], 1), atol=1e-6)


def test_rope2d_is_relative_in_vision_attention():
    """q.k after 2D RoPE depends only on the grid offset between the two tokens."""
    from rainbot.nn.rope import apply_rope, rope2d_angles
    cos, sin = rope2d_angles((6, 10), 32)
    q = torch.randn(1, 1, 1, 32).expand(1, 60, 1, 32)
    k = torch.randn(1, 1, 1, 32).expand(1, 60, 1, 32)
    qr, kr = apply_rope(q, cos, sin, 1, 60), apply_rope(k, cos, sin, 1, 60)
    dot = lambda i, j: float((qr[0, i, 0] * kr[0, j, 0]).sum())
    # (row 1, col 2) -> (row 2, col 5) has the same offset as (row 3, col 4) -> (row 4, col 7).
    assert abs(dot(1 * 10 + 2, 2 * 10 + 5) - dot(3 * 10 + 4, 4 * 10 + 7)) < 1e-4
    # Tokens past n are left unrotated.
    x = torch.randn(2, 70, 2, 32)
    assert torch.equal(apply_rope(x, cos, sin, 1, 60)[:, 60:], x[:, 60:])
