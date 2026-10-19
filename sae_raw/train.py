"""Training loop for the SAEs on the embedding table.

Minibatches are drawn at random (with replacement) from the rows of the
embedding table, and the model's loss is averaged over the batch.

Default: uniform sampling, so in expectation the reconstruction term is
(1/V) sum_i |e_i - e_hat_i|^2. Frequency weighting is a toggle, off by default:
if token counts are passed to train(), tokens are drawn with probability
proportional to their count, and evaluation is weighted the same way. Tokens
with zero count are then never sampled.

For the TopK model, k can follow a schedule (k_schedule), lowering it to 1
during training (Experiment 3). A schedule is laid out over a run as a list
of [k, last step] (k_stages), which stretch_stages can lengthen part-way
through, so that a scheduled run can be extended.

At every evaluation the loop also reports revival: how many of the features
that were dead at initialisation (active on no token) have since received
gradient, judged by whether their parameters have changed.

With a run directory, training is resumable. The loop saves a checkpoint
(model, optimiser, sampler state and history) every few minutes, and on the
first Ctrl-C it finishes the current step, saves and stops. Calling train()
again with the same run directory continues from the checkpoint, and gives
the same result as an uninterrupted run. A heartbeat line reports progress
and the estimated time remaining every minute.
"""

import csv
import math
import os
import signal
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch

from sae_raw.evaluation import changed_features, full_pass, revival, summarize
from sae_raw.model import build_model

INIT_FILE = "init.pt"              # parameters at initialisation
CHECKPOINT_FILE = "checkpoint.pt"  # everything needed to resume
MODEL_FILE = "model.pt"            # model parameters at the latest evaluation
EVAL_LOG_FILE = "eval_log.csv"     # one row per evaluation
SNAPSHOT_DIR = "snapshots"         # snapshots/model_step<N>.pt: parameters kept at chosen steps


def default_lr(m):
    """Learning rate from the scaling rule of Gao et al. (2024): 2e-4 / sqrt(m / 2^14).

    This gives about 1.1e-4 for m = 50,257. The rule was derived for TopK SAEs.
    """
    return 2e-4 / math.sqrt(m / 2 ** 14)


class UniformSampler:
    """Draw token ids uniformly at random from range(V), with replacement."""

    def __init__(self, V, seed):
        self.V = V
        self.p = torch.full((V,), 1.0 / V, dtype=torch.float64)
        self.gen = torch.Generator().manual_seed(seed)

    def __call__(self, n):
        return torch.randint(self.V, (n,), generator=self.gen)


class FrequencySampler:
    """Draw token ids with probability proportional to counts."""

    def __init__(self, counts, seed):
        counts = torch.as_tensor(counts, dtype=torch.float64)
        self.p = counts / counts.sum()
        cdf = torch.cumsum(self.p, dim=0)
        # Dividing by the last value makes the cdf reach exactly 1.0 at the last
        # token with positive count, so trailing zero-count tokens cannot be drawn.
        self.cdf = cdf / cdf[-1]
        self.gen = torch.Generator().manual_seed(seed)

    def __call__(self, n):
        u = torch.rand(n, generator=self.gen, dtype=torch.float64)
        return torch.searchsorted(self.cdf, u, right=True)


def k_schedule(step, steps, k0=None, end_frac=None, shape="halving", stages=None):
    """Number of active features k at a training step (1-based), for k-annealing.

    shape "stages": stages is a list of (k, fraction of training) pairs, in
    order, with fractions summing to 1. k0 and end_frac are then ignored.

    shape "halving" or "linear": k starts at k0 and reaches 1 after a fraction
    end_frac of the steps, then stays at 1. "halving" passes through k0, k0/2,
    ..., 2 in equal stages (k0 a power of 2); "linear" rounds to the nearest
    integer.
    """
    if shape == "stages":
        t, acc = (step - 1) / steps, 0.0
        for k, frac in stages:
            acc += frac
            if t < acc - 1e-12:
                return k
        return stages[-1][0]
    t = (step - 1) / steps / end_frac  # progress through the decrease, 0 to 1
    if t >= 1:
        return 1
    if shape == "linear":
        return max(1, round(k0 - (k0 - 1) * t))
    if shape == "halving":
        levels = [k0 >> i for i in range(int(math.log2(k0)))]  # k0, ..., 2
        return levels[int(t * len(levels))]
    raise ValueError(f"unknown schedule shape {shape!r}")


def schedule_start_k(schedule):
    """The k a schedule starts at."""
    return schedule["stages"][0][0] if schedule.get("shape") == "stages" else schedule["k0"]


def k_stages(schedule, steps):
    """A schedule laid out over a run: a list of [k, last step], in order.

    k is used from the step after the previous stage's last step up to and
    including this stage's last step. The last stage ends at steps.
    """
    out = []
    for s in range(1, steps + 1):
        k = k_schedule(s, steps, **schedule)
        if out and out[-1][0] == k:
            out[-1][1] = s
        else:
            out.append([k, s])
    return out


def k_at(step, stages):
    """k at a training step (1-based) under a list of [k, last step]. After the
    last stage, its k continues."""
    for k, last in stages:
        if step <= last:
            return k
    return stages[-1][0]


def stretch_stages(stages, done, steps):
    """Lay out the rest of a schedule over a new total of steps.

    Steps up to done keep their k. The part of the schedule after done is
    stretched in proportion so that it ends at steps. If the schedule had
    already finished by done, its last k simply continues to steps.
    """
    old = stages[-1][1]
    if done >= old:
        return [list(st) for st in stages[:-1]] + [[stages[-1][0], steps]]
    scale = (steps - done) / (old - done)
    out = []
    for k, last in stages:
        new_last = last if last <= done else done + round((last - done) * scale)
        if out and new_last <= out[-1][1]:
            continue  # a stage too short to survive rounding
        out.append([k, new_last])
    out[-1][1] = steps
    return out


def train_step(model, optimizer, x):
    """One optimiser step with the unit-norm decoder constraint. Returns (loss, mse)."""
    loss, mse = model.loss(x)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    model.project_out_parallel_grad_()
    optimizer.step()
    model.normalize_decoder_()
    return float(loss.detach()), float(mse.detach())


class Trainer:
    """Model, optimiser, sampler and evaluation history of one training run."""

    def __init__(self, E, lr, seed, model="top1", l1_coef=None, k=None, counts=None,
                 device="cpu"):
        self.E = torch.as_tensor(E, dtype=torch.float32, device=device)
        V, d = self.E.shape
        self.model = build_model(model, d, V, seed=seed, l1_coef=l1_coef, k=k).to(device)
        self.init_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.sampler = UniformSampler(V, seed) if counts is None else FrequencySampler(counts, seed)
        self.p = self.sampler.p.to(device)
        self.device = device
        self.step = 0
        self.history = []
        self.dead_at_init = None
        self._elapsed_before = 0.0  # training time in earlier sessions of this run
        self._session_start = time.time()

    def seconds(self):
        """Total training time of this run so far, across sessions."""
        return self._elapsed_before + time.time() - self._session_start

    def train_step(self, batch_size):
        x = self.E[self.sampler(batch_size).to(self.device)]
        out = train_step(self.model, self.optimizer, x)
        self.step += 1
        return out

    def evaluate(self, loss=float("nan"), mse=float("nan")):
        sq_err, l0, active = full_pass(self.model, self.E, self.p)
        if self.dead_at_init is None:
            self.dead_at_init = ~active
        record = {"step": self.step, "seconds": round(self.seconds(), 1),
                  "k": getattr(self.model, "k", None if self.model.kind == "l1" else 1),
                  "minibatch_loss": loss, "minibatch_mse": mse,
                  **summarize(sq_err, l0, active, self.E, self.p),
                  **revival(self.dead_at_init, changed_features(self.init_state, self.model))}
        self.history.append(record)
        return record

    def state_dict(self):
        return {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "sampler_rng": self.sampler.gen.get_state(), "step": self.step,
                "history": self.history, "dead_at_init": self.dead_at_init,
                "seconds": self.seconds()}

    def load_state_dict(self, state, init_state):
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.sampler.gen.set_state(state["sampler_rng"])
        self.step = state["step"]
        self.history = state["history"]
        self.dead_at_init = state["dead_at_init"]
        self.init_state = {k: v.to(self.device) for k, v in init_state.items()}
        self._elapsed_before = state["seconds"]
        self._session_start = time.time()


def _atomic_save(obj, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)  # a crash during saving never leaves a half-written file


def _write_eval_log(history, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def fmt_duration(seconds):
    """Format seconds as H:MM:SS."""
    s = int(max(seconds, 0))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _format_record(record):
    return " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in record.items())


class _StopOnInterrupt:
    """On the first Ctrl-C, ask the loop to stop after the current step. A second one aborts."""

    def __init__(self, log):
        self.requested = False
        self.log = log

    def __enter__(self):
        self._previous = signal.signal(signal.SIGINT, self._handle)
        return self

    def _handle(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        self.log("Interrupt received: finishing the current step, then saving a checkpoint "
                 "(press Ctrl-C again to abort without saving).")

    def __exit__(self, *exc):
        signal.signal(signal.SIGINT, self._previous)
        return False


def train(E, steps, batch_size, lr, seed, eval_every, model="top1", l1_coef=None, k=None,
          counts=None, device="cpu", run_dir=None, checkpoint_every=600,
          heartbeat_every=60, extra_eta_seconds=0.0, schedule=None, step_seconds_by_k=None,
          snapshot_steps=(), stages=None, log=print):
    """Train an SAE with m = V on the rows of E.

    model is "top1" (Experiment 1), "l1" (Experiment 2, needs l1_coef) or
    "topk" (needs k; Experiment 3 anneals k).
    Evaluations happen at step 0, every eval_every steps and at the last step.
    If counts (length V) is given, tokens are sampled in proportion to counts
    and evaluation is weighted the same way.

    With run_dir, the run writes init.pt (before training), eval_log.csv and
    model.pt (at every evaluation) and checkpoint.pt (every checkpoint_every
    seconds, on interrupt and at the end), and resumes from checkpoint.pt if it
    exists. heartbeat_every sets the interval of the progress line, whose ETA
    includes extra_eta_seconds for work after training. model.pt is
    overwritten at each evaluation. At evaluations whose step is in
    snapshot_steps, the parameters are also kept in snapshots/model_step<N>.pt.

    schedule, for the "topk" model only, is a dict of keyword arguments for
    k_schedule. k then follows the schedule, and k is ignored. Alternatively,
    stages gives the schedule already laid out over the run, as a list of
    [k, last step] (see k_stages and stretch_stages); it takes precedence.
    step_seconds_by_k optionally maps each k in the schedule to a measured
    step time. The heartbeat then estimates the time left from these, scaled
    to the speed observed so far, since steps at large k are slower.

    Returns (model, history, finished). finished is False if the run was
    interrupted; call train() again with the same run_dir to continue.
    """
    if stages is None and schedule is not None:
        stages = k_stages(schedule, steps)
    scheduled = stages is not None
    if scheduled:
        if model != "topk":
            raise ValueError("a k schedule needs the topk model")
        k = stages[0][0]
    tr = Trainer(E, lr, seed, model=model, l1_coef=l1_coef, k=k, counts=counts, device=device)
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt = run_dir / CHECKPOINT_FILE
        if ckpt.exists():
            tr.load_state_dict(torch.load(ckpt, map_location=device),
                               torch.load(run_dir / INIT_FILE, map_location=device))
            log(f"Resuming from {ckpt} at step {tr.step:,} of {steps:,} "
                f"({fmt_duration(tr.seconds())} of training so far).")
        else:
            _atomic_save(tr.init_state, run_dir / INIT_FILE)

    def after_eval(record):
        log("eval " + _format_record(record))
        if run_dir is not None:
            _write_eval_log(tr.history, run_dir / EVAL_LOG_FILE)
            _atomic_save(tr.model.state_dict(), run_dir / MODEL_FILE)
            if record["step"] in snapshot_steps:
                path = run_dir / SNAPSHOT_DIR / f"model_step{record['step']}.pt"
                path.parent.mkdir(exist_ok=True)
                _atomic_save(tr.model.state_dict(), path)
                log(f"Snapshot saved: {path.name}")

    def save_checkpoint(reason):
        if run_dir is not None:
            t0 = time.time()
            _atomic_save(tr.state_dict(), run_dir / CHECKPOINT_FILE)
            log(f"Checkpoint saved at step {tr.step:,} ({reason}, {time.time() - t0:.0f}s).")

    if scheduled and tr.step > 0:  # resuming part-way through the schedule
        tr.model.k = k_at(tr.step, stages)
    if not tr.history:
        after_eval(tr.evaluate())

    # Timing for the heartbeat, measured in this session only.
    session_step0, step_time, eval_time = tr.step, 0.0, None
    use_k_timing = scheduled and step_seconds_by_k is not None
    expected_done = 0.0  # sum of measured step times over the steps done this session
    remaining_evals = lambda: (sum(1 for s in range(tr.step + 1, steps + 1)
                                   if s % eval_every == 0 or s == steps))
    last_ckpt = last_beat = time.time()
    loss = mse = float("nan")

    def heartbeat():
        done = tr.step - session_step0
        if done == 0:
            return
        per_step = step_time / done
        if use_k_timing:
            remaining = sum(step_seconds_by_k[k_at(s, stages)]
                            for s in range(tr.step + 1, steps + 1))
            train_left = remaining * step_time / expected_done
        else:
            train_left = (steps - tr.step) * per_step
        eta = train_left + remaining_evals() * (eval_time or 0.0) + extra_eta_seconds
        finish = (datetime.now() + timedelta(seconds=eta)).strftime("%a %H:%M")
        last = tr.history[-1]
        k_now = f"k={tr.model.k} | " if scheduled else ""
        log(f"step {tr.step:,}/{steps:,} ({100 * tr.step / steps:.1f}%) | {k_now}"
            f"{per_step:.2f} s/step | elapsed {fmt_duration(tr.seconds())} | "
            f"ETA {fmt_duration(eta)} (about {finish}) | minibatch loss {loss:.4g} | "
            f"last eval (step {last['step']:,}): rel_loss {last['rel_loss']:.4g}, "
            f"active per token {last['mean_l0']:.3g}, alive {last['n_alive']:,}"
            + (f", revived {last['n_revived']:,}/{last['n_dead_at_init']:,} "
               f"({last['revival_pct']:.2f}%)" if last["n_dead_at_init"] else ""))

    stop = _StopOnInterrupt(log)
    with stop:
        while tr.step < steps and not stop.requested:
            t0 = time.time()
            if scheduled:
                tr.model.k = k_at(tr.step + 1, stages)
            loss, mse = tr.train_step(batch_size)
            step_time += time.time() - t0
            if use_k_timing:
                expected_done += step_seconds_by_k[tr.model.k]
            if tr.step % eval_every == 0 or tr.step == steps:
                t0 = time.time()
                after_eval(tr.evaluate(loss, mse))
                eval_time = time.time() - t0
            now = time.time()
            if now - last_ckpt >= checkpoint_every:
                save_checkpoint(f"every {checkpoint_every / 60:g} min")
                last_ckpt = time.time()
            if now - last_beat >= heartbeat_every:
                heartbeat()
                last_beat = now

    finished = tr.step >= steps
    save_checkpoint("finished" if finished else "interrupted")
    if not finished and run_dir is not None:
        log("Stopped early. Run the same command again to resume from the checkpoint.")
    return tr.model, tr.history, finished
