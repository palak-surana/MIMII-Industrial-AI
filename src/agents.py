
"""Phase 8: Four logical agents and coordinator.

Workflow: train -> select -> annotate -> evaluate/budget.

The unseen test machine is reserved for final evaluation.
"""

import math
import numpy as np

from .active_learning import select_random, select_uncertain


class TrainingAgent:
    """Agent 1: Train CNN-LSTM on currently labelled training samples."""

    def __init__(self, trainer_factory, data):
        self.factory = trainer_factory
        self.d = data
        self.trainer = None

    def run(self, pool, round_no, seed):
        idx = pool.labelled_idx

        if len(idx) == 0:
            raise RuntimeError("No labelled training samples available.")

        self.trainer = self.factory(seed)

        info = self.trainer.fit(
            self.d.Xtr[idx],
            pool.labels_of_labelled(idx),
            self.d.Xva,
            self.d.yva
        )

        return info


class SelectionAgent:
    """Agent 2: Select informative unlabelled training samples."""

    def __init__(self, data, strategy, seed):
        self.d = data
        self.strategy = strategy
        self.rng = np.random.default_rng(seed)

    def run(self, trainer, pool, n):
        unl = pool.unlabelled_idx

        if n < 0:
            raise ValueError("Selection count cannot be negative.")

        if n == 0 or len(unl) == 0:
            return np.array([], dtype=int), None

        if self.strategy == "random":
            return select_random(unl, n, self.rng)

        if self.strategy == "uncertainty":
            probs = trainer.predict_proba(self.d.Xtr[unl])
            return select_uncertain(unl, probs, n)

        raise ValueError(
            f"Unknown selection strategy: {self.strategy}"
        )


class AnnotationAgent:
    """Agent 3: Simulated maintenance engineer annotation.

    Labels are revealed only through the actual hidden MIMII
    ground truth stored in LabelPool. No labels are invented.
    """

    def run(self, pool, idx, round_no):
        return pool.annotate(idx, round_no)


class EvaluationBudgetAgent:
    """Agent 4: Evaluate validation F1 and control annotation budget."""

    def __init__(
        self,
        max_frac,
        batch_frac,
        target_f1=None,
        stop_on_target=True
    ):
        if not 0 < max_frac <= 1:
            raise ValueError("max_frac must be in (0, 1].")

        if not 0 < batch_frac <= 1:
            raise ValueError("batch_frac must be in (0, 1].")

        if target_f1 is not None and not 0 <= target_f1 <= 1:
            raise ValueError("target_f1 must be between 0 and 1.")

        self.max_frac = max_frac
        self.batch_frac = batch_frac
        self.target = target_f1
        self.stop_on_target = stop_on_target

    def run(self, val_f1, pool):

        if not np.isfinite(val_f1) or not 0 <= val_f1 <= 1:
            raise ValueError("Validation F1 must be between 0 and 1.")

        cap = int(round(self.max_frac * pool.n))

        if (
            self.stop_on_target
            and self.target is not None
            and val_f1 >= self.target
        ):
            return {
                "go": False,
                "reason": f"target F1 {self.target:.2f} reached",
                "n_next": 0
            }

        n_next = min(
            math.ceil(self.batch_frac * pool.n),
            cap - pool.n_labelled,
            len(pool.unlabelled_idx)
        )

        if n_next <= 0:
            return {
                "go": False,
                "reason": "annotation budget exhausted",
                "n_next": 0
            }

        return {
            "go": True,
            "reason": "target not reached, budget remaining",
            "n_next": n_next
        }


class Coordinator:
    """Coordinate the four agents.

    Validation data controls model selection and budget decisions.
    The unseen test machine is NOT accessed during active learning.
    """

    def __init__(
        self,
        data,
        pool,
        trainer_factory,
        strategy,
        budget_agent,
        seed=0,
        verbose=True
    ):
        if strategy not in ("random", "uncertainty"):
            raise ValueError(f"Unsupported strategy: {strategy}")

        if len(data.Xtr) != pool.n:
            raise ValueError(
                "Training data and label pool sizes do not match."
            )

        if len(data.mtr) != pool.n:
            raise ValueError(
                "Training metadata and label pool sizes do not match."
            )

        self.d = data
        self.pool = pool
        self.strategy = strategy
        self.seed = seed
        self.verbose = verbose

        self.train_agent = TrainingAgent(trainer_factory, data)
        self.select_agent = SelectionAgent(data, strategy, seed)
        self.annot_agent = AnnotationAgent()
        self.budget_agent = budget_agent

        self.log = []
        self.trainer = None

    def run(self):
        r = 0

        while True:

            # 1. Training Agent
            info = self.train_agent.run(
                self.pool,
                r,
                seed=self.seed * 1000 + r
            )

            if "val_f1" not in info:
                raise ValueError(
                    "Trainer did not return validation F1."
                )

            trainer = self.train_agent.trainer
            thr = getattr(trainer, "thr", 0.5)

            # 2. Evaluation/Budget Agent
            # Decisions use validation F1 only.
            dec = self.budget_agent.run(
                info["val_f1"],
                self.pool
            )

            rec = {
                "strategy": self.strategy,
                "seed": self.seed,
                "round": r,
                "n_labelled": self.pool.n_labelled,
                "frac_labelled": self.pool.frac_labelled,
                "n_unlabelled": len(self.pool.unlabelled_idx),
                "val_f1": float(info["val_f1"]),
                "thr": float(thr),
                "target_f1": self.budget_agent.target,
                "decision": (
                    "STOP: " + dec["reason"]
                    if not dec["go"]
                    else f"Request {dec['n_next']} additional segment annotations"
                ),
                "selection_method": (
                    "Highest model uncertainty (entropy)"
                    if self.strategy == "uncertainty"
                    else "Random selection"
                )
            }

            # 3. Selection Agent and Annotation Agent
            if dec["go"]:

                idx, scores = self.select_agent.run(
                    trainer,
                    self.pool,
                    dec["n_next"]
                )

                if len(idx) == 0:
                    raise RuntimeError(
                        "Selection agent returned no samples "
                        "while budget remained."
                    )

                # Record every selected candidate for auditability.
                candidates = []

                if scores is None:
                    score_values = [None] * len(idx)
                else:
                    score_values = scores

                for sample_idx, score in zip(idx, score_values):
                    metadata = self.d.mtr.iloc[int(sample_idx)]

                    candidates.append({
                        "segment": (
                            f"{metadata['machine'].upper()}_"
                            f"{metadata['id']}_"
                            f"{str(metadata['file'])[:-4]}_"
                            f"f{metadata['start_frame']}"
                        ),
                        "uncertainty": (
                            None if score is None else float(score)
                        )
                    })

                rec["candidates"] = candidates

                self.annot_agent.run(
                    self.pool,
                    idx,
                    r + 1
                )

            self.log.append(rec)

            if self.verbose:
                self._print(rec)

            if not dec["go"]:
                self.trainer = trainer
                return self.log

            r += 1

    @staticmethod
    def _print(rec):
        print(
            f"\nCurrent Round: {rec['round']} "
            f"[{rec['strategy']}, seed {rec['seed']}]"
            f"\nCurrently labelled: {rec['n_labelled']:,}"
            f"\nUnlabelled: {rec['n_unlabelled']:,}"
            f"\nValidation F1: {rec['val_f1']:.4f}"
            f"\nTarget F1: {rec['target_f1']}"
            f"\nNext Action: {rec['decision']}"
            f"\nSelection Method: {rec['selection_method']}",
            flush=True
        )


def save_log(log, path):
    """Save active-learning experiment logs as JSON."""

    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1, allow_nan=False)
