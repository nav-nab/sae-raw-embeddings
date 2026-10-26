import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("night1", REPO / "scripts" / "run_night1.py")
night1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(night1)


def _fake(tmp_path, name, body):
    path = tmp_path / f"{name}.py"
    path.write_text(f"calls = []\ndef main(argv=None):\n    calls.append(argv)\n    {body}\n")
    return str(path)


def test_runs_in_order_continues_after_an_error_and_stops_on_interrupt(tmp_path, monkeypatch,
                                                                      capsys):
    monkeypatch.setattr(night1, "REPO", Path("/"))
    ok = _fake(tmp_path, "ok", "return {}")
    bad = _fake(tmp_path, "bad", "raise ValueError('boom')")
    stop = _fake(tmp_path, "stop", "return None")
    monkeypatch.setattr(night1, "PLAN", [("A", ok, ["--x"]), ("B", bad, []), ("C", stop, []),
                                         ("D", ok, [])])
    assert night1.main([]) == 1
    out = capsys.readouterr().out
    assert "A: finished" in out and "B: failed" in out and "C: interrupted" in out
    assert "D: not started" in out


def test_all_finished_returns_zero(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(night1, "REPO", Path("/"))
    ok = _fake(tmp_path, "ok2", "return {}")
    monkeypatch.setattr(night1, "PLAN", [("A", ok, []), ("B", ok, [])])
    assert night1.main([]) == 0


def test_plan_points_at_real_scripts_with_main():
    for _, path, _ in night1.PLAN:
        assert callable(night1.load(path).main)
