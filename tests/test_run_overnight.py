import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("overnight", REPO / "scripts" / "run_overnight.py")
overnight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overnight)


def _fake(tmp_path, name, body):
    path = tmp_path / f"{name}.py"
    path.write_text(f"calls = []\ndef main(argv=None):\n    calls.append(argv)\n    {body}\n")
    return str(path)


def test_runs_in_order_continues_after_an_error_and_stops_on_interrupt(tmp_path, monkeypatch,
                                                                      capsys):
    monkeypatch.setattr(overnight, "REPO", Path("/"))
    ok = _fake(tmp_path, "ok", "return {}")
    bad = _fake(tmp_path, "bad", "raise ValueError('boom')")
    stop = _fake(tmp_path, "stop", "return None")
    plan = [("A", ok, ["--x"]), ("B", bad, []), ("C", stop, []), ("D", ok, [])]
    assert overnight.run_plan(plan) == 1
    out = capsys.readouterr().out
    assert "A: finished" in out and "B: failed" in out and "C: interrupted" in out
    assert "D: not started" in out


def test_all_finished_returns_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(overnight, "REPO", Path("/"))
    ok = _fake(tmp_path, "ok2", "return {}")
    assert overnight.run_plan([("A", ok, []), ("B", ok, [])]) == 0


def test_default_plan_is_the_remaining_runs():
    monkeypatch_plan = overnight.PLANS["remaining"]
    assert [path for _, path, _ in monkeypatch_plan] == [overnight.EXP7, overnight.EXP8]


def test_plans_point_at_real_scripts_with_main():
    for plan in overnight.PLANS.values():
        for _, path, _ in plan:
            assert callable(overnight.load(path).main)
