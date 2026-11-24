"""One line per training iteration from metrics.jsonl (other lines pass through), to watch a run.

    tail -F runs/<run>/metrics.jsonl | python3 -u tools/fmt_metrics.py
"""
import json
import sys

for line in sys.stdin:
    line = line.strip()
    if not line.startswith("{"):
        print(line, flush=True)
        continue
    try:
        d = json.loads(line)
    except ValueError:
        print(line, flush=True)
        continue
    if "iter" not in d:
        continue
    c = d.get("curriculum") or {}
    r = lambda v: round(v, 3) if isinstance(v, float) else v
    print(f"iter={d['iter']} stage={c.get('stage')} level={c.get('level')} passes={c.get('passes')} "
          f"fed_hib={r(c.get('fed_hibernation_rate'))} hib={r(c.get('hibernation_rate'))} "
          f"karma_net={c.get('karma_net')} deaths={c.get('deaths_by_cause')} "
          f"pips/cycle={r(c.get('pips_per_cycle'))} need_met={r(c.get('food_need_met'))} "
          f"noop={r(d.get('noop_frac'))} zero_adv={r(d.get('zero_adv_frac'))} "
          f"change={r(d.get('action_change_frac'))} refills={d.get('refills')} event={c.get('event')}",
          flush=True)
