"""A small feed-forward network over the same tabular features.

Deliberately a *plain* MLP on the identical feature matrix the trees see. That
is the point: holding inputs, split and labels fixed isolates the effect of
model family, and it lets gradient-based attribution (Integrated Gradients,
DeepLIFT) be compared directly against TreeSHAP on the same axes.

The network keeps preprocessing outside the module (imputation and scaling are
fitted on train and applied as arrays) so that captum sees a clean, everywhere-
differentiable function from standardised features to class logits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .datasets import CONDITIONS
from .metadata import EMOTIONS
from .models import _matrices, evaluate, prepare
from .paths import MODELS_DIR, ensure_dirs, timestamp

SEED = 42


@dataclass
class AnnConfig:
    hidden: tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 200
    patience: int = 25
    label_smoothing: float = 0.05


class EmotionMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 6, cfg: AnnConfig | None = None):
        super().__init__()
        cfg = cfg or AnnConfig()
        layers: list[nn.Module] = []
        prev = n_features
        for width in cfg.hidden:
            layers += [
                nn.Linear(prev, width),
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            ]
            prev = width
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchWrapper:
    """sklearn-shaped adapter so the ANN reuses the shared `evaluate()`."""

    def __init__(self, model: EmotionMLP, imputer: SimpleImputer, scaler: StandardScaler):
        self.model, self.imputer, self.scaler = model, imputer, scaler

    def transform(self, X) -> np.ndarray:
        return self.scaler.transform(self.imputer.transform(X)).astype(np.float32)

    def _logits(self, X) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.from_numpy(self.transform(X)))

    def predict(self, X) -> np.ndarray:
        return self._logits(X).argmax(1).numpy()

    def predict_proba(self, X) -> np.ndarray:
        return torch.softmax(self._logits(X), dim=1).numpy()


def train_ann(
    path: str | None = None,
    train_condition: str = "ref",
    cfg: AnnConfig | None = None,
    verbose: bool = True,
) -> dict:
    ensure_dirs()
    cfg = cfg or AnnConfig()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df, cols, split = prepare(path)
    sub, masks, X, y = _matrices(df, cols, split, train_condition)

    imputer = SimpleImputer(strategy="median").fit(X[masks["train"]])
    scaler = StandardScaler().fit(imputer.transform(X[masks["train"]]))

    def prep(Xd):
        return torch.from_numpy(scaler.transform(imputer.transform(Xd)).astype(np.float32))

    Xtr, ytr = prep(X[masks["train"]]), torch.from_numpy(y[masks["train"]]).long()
    Xva, yva = prep(X[masks["val"]]), torch.from_numpy(y[masks["val"]]).long()

    counts = np.bincount(ytr.numpy(), minlength=len(EMOTIONS))
    class_weight = torch.tensor((counts.sum() / (len(counts) * np.maximum(counts, 1))),
                                dtype=torch.float32)

    model = EmotionMLP(Xtr.shape[1], len(EMOTIONS), cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    lossf = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=cfg.label_smoothing)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr, ytr), batch_size=cfg.batch_size, shuffle=True
    )

    history, best_uar, best_state, stale = [], -1.0, None, 0
    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss) * len(xb)
        sched.step()

        model.eval()
        with torch.no_grad():
            va_logits = model(Xva)
            va_loss = float(lossf(va_logits, yva))
            pred = va_logits.argmax(1).numpy()
        uar = float(np.mean([
            (pred[yva.numpy() == c] == c).mean() for c in range(len(EMOTIONS))
            if (yva.numpy() == c).any()
        ]))
        history.append({"epoch": epoch, "train_loss": total / len(Xtr),
                        "val_loss": va_loss, "val_uar": uar})

        if uar > best_uar:
            best_uar, best_state, stale = uar, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  train {total/len(Xtr):.4f}  val {va_loss:.4f}  UAR {uar:.3f}")

    model.load_state_dict(best_state)
    wrapper = TorchWrapper(model, imputer, scaler)

    metrics = {"val": evaluate(wrapper, X[masks["val"]], y[masks["val"]])}
    for cond in CONDITIONS:
        sub_c, masks_c, Xc, yc = _matrices(df, cols, split, cond)
        metrics[f"test_{cond}"] = evaluate(wrapper, Xc[masks_c["test"]], yc[masks_c["test"]])

    ts = timestamp()
    out_dir = MODELS_DIR / "ann" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
            "feature_names": cols,
            "imputer_statistics": imputer.statistics_,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "train_condition": train_condition,
            "split": {"train": list(split.train), "val": list(split.val), "test": list(split.test)},
        },
        out_dir / "ann.pt",
    )
    # The fitted preprocessing objects are pickled as-is. Rebuilding a
    # SimpleImputer from its statistics_ array alone leaves private sklearn
    # state unset and fails on transform.
    joblib.dump({"imputer": imputer, "scaler": scaler}, out_dir / "preproc.joblib")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump({"train_condition": train_condition, "best_val_uar": best_uar,
                   "epochs_run": len(history), "metrics": metrics,
                   "config": asdict(cfg)}, fh, indent=2)
    with open(MODELS_DIR / "ann" / "LATEST", "w") as fh:
        fh.write(ts)

    if verbose:
        print(f"ANN best val UAR {best_uar:.3f} | test UAR "
              f"{metrics[f'test_{train_condition}']['balanced_accuracy']:.3f} -> {out_dir}")
    return {"dir": str(out_dir), "metrics": metrics, "history": history, "best_val_uar": best_uar}


def load_ann(run_dir=None):
    """Restore a trained ANN plus its preprocessing, ready for captum."""
    base = MODELS_DIR / "ann"
    if run_dir is None:
        run_dir = base / (base / "LATEST").read_text().strip()
    blob = torch.load(run_dir / "ann.pt", weights_only=False)
    cfg = AnnConfig(**blob["config"])
    model = EmotionMLP(len(blob["feature_names"]), len(EMOTIONS), cfg)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    pre = joblib.load(run_dir / "preproc.joblib")
    return model, TorchWrapper(model, pre["imputer"], pre["scaler"]), blob


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train the PyTorch ANN.")
    ap.add_argument("--path", default=None)
    ap.add_argument("--train-condition", default="ref")
    ap.add_argument("--epochs", type=int, default=200)
    a = ap.parse_args()
    train_ann(path=a.path, train_condition=a.train_condition, cfg=AnnConfig(epochs=a.epochs))
