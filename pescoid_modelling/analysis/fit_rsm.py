"""Adaptive response surface model fitting with automatic model selection."""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score  # type: ignore
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline  # type: ignore
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures  # type: ignore
from sklearn.preprocessing import StandardScaler

from pescoid_modelling.visualization import _set_matplotlib_publication_parameters
from pescoid_modelling.visualization.plot_rsm_phase_diagram import plot_rsm_surface

warnings.filterwarnings("ignore")


def _load_param_sweep(
    csv: Path, response: str = "onset_time"
) -> tuple[np.ndarray, np.ndarray, str, str, str]:
    """Load parameter sweep CSV and extract axes and response variable."""
    df = pd.read_csv(csv)

    column_configs = {
        frozenset(["activity", "r"]): ("activity", "r", r"$A$", r"$R$", "AR"),
        frozenset(["beta", "r"]): ("beta", "r", r"$\beta$", r"$R$", "BR"),
        frozenset(["r", "tau_m"]): ("r", "tau_m", r"$R$", r"$\tau_m$", "RTm"),
        frozenset(["activity", "flow"]): ("activity", "flow", r"$A$", r"$F$", "AF"),
    }

    matching_config = None
    for col_set, config in column_configs.items():
        if col_set.issubset(df.columns):
            matching_config = config
            break

    if matching_config is None:
        raise ValueError("Cannot infer x-axes from CSV header.")

    x1, x2, label1, label2, tag = matching_config
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[response])
    X = df[[x1, x2]].to_numpy(float)
    y = df[response].to_numpy(float)

    return X, y, label1, label2, tag


def _create_models() -> dict[str, Pipeline]:
    """Create polynomial and random forest models."""
    return {
        "linear": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=1, include_bias=True),
            LinearRegression(),
        ),
        "quadratic": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=True),
            LinearRegression(),
        ),
        "cubic": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=3, include_bias=True),
            LinearRegression(),
        ),
        "quartic": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=4, include_bias=True),
            LinearRegression(),
        ),
        "random_forest": make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
        ),
    }


def _evaluate_models(
    models: dict[str, Pipeline], X: np.ndarray, y: np.ndarray
) -> dict[str, dict[str, float]]:
    """Evaluate all models using cross-validation."""
    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        try:
            cv_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
            cv_mse = -cross_val_score(
                model, X, y, cv=kf, scoring="neg_mean_squared_error"
            )

            model.fit(X, y)
            y_pred = model.predict(X)

            results[name] = {
                "cv_r2_mean": float(cv_scores.mean()),
                "cv_r2_std": float(cv_scores.std()),
                "cv_mse_mean": float(cv_mse.mean()),
                "cv_mse_std": float(cv_mse.std()),
                "train_r2": float(r2_score(y, y_pred)),
                "train_mse": float(mean_squared_error(y, y_pred)),
            }
        except Exception:
            results[name] = {
                "cv_r2_mean": float(-np.inf),
                "cv_r2_std": float(np.inf),
                "cv_mse_mean": float(np.inf),
                "cv_mse_std": float(np.inf),
                "train_r2": float(-np.inf),
                "train_mse": float(np.inf),
            }

    return results


def _analyze_complexity(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Simple complexity analysis of the response surface."""
    y_range = float(np.ptp(y))
    y_std = float(np.std(y))
    n_points = len(y)

    x1_range = float(np.ptp(X[:, 0]))
    x2_range = float(np.ptp(X[:, 1]))

    return {
        "response_range": y_range,
        "response_std": y_std,
        "response_cv": y_std / np.mean(y) if np.mean(y) != 0 else float("inf"),  # type: ignore
        "n_points": n_points,
        "x1_range": x1_range,
        "x2_range": x2_range,
        "point_density": n_points / (x1_range * x2_range),
    }


def _recommend_model(complexity_metrics: dict[str, float]) -> str:
    """Simple recommendation based on data characteristics."""
    cv = complexity_metrics["response_cv"]
    n_points = complexity_metrics["n_points"]

    if n_points < 50:
        return "quadratic"
    elif cv > 1.0:  # High variability
        return "random_forest"
    elif n_points > 200:
        return "quartic"
    else:
        return "cubic"


def _select_best_model(
    models: dict[str, Pipeline], results: dict[str, dict[str, float]]
) -> tuple[str, Pipeline]:
    """Select best model based on cross-validation performance."""
    valid_results = {k: v for k, v in results.items() if v["cv_r2_mean"] > -1.0}
    if not valid_results:
        return list(models.keys())[0], list(models.values())[0]

    best_name = max(valid_results.keys(), key=lambda k: valid_results[k]["cv_r2_mean"])
    return best_name, models[best_name]


def _print_results(results: dict[str, dict[str, float]], best_name: str) -> None:
    """Print clean model comparison results."""
    print("\nMODEL COMPARISON")
    print("-" * 50)

    sorted_models = sorted(
        results.items(), key=lambda x: x[1]["cv_r2_mean"], reverse=True
    )

    for name, metrics in sorted_models:
        if metrics["cv_r2_mean"] > -1.0:
            marker = " ←" if name == best_name else ""
            print(
                f"{name:<15} CV R²: {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}{marker}"
            )


def _fit_adaptive_model(
    X: np.ndarray, y: np.ndarray, tag: str
) -> tuple[Pipeline, str, dict]:
    """Fit best model automatically based on data characteristics."""
    complexity_metrics = _analyze_complexity(X, y)
    recommended = _recommend_model(complexity_metrics)

    print(f"Data points: {complexity_metrics['n_points']}")
    print(f"Response CV: {complexity_metrics['response_cv']:.2f}")
    print(f"Recommended: {recommended}")

    models = _create_models()
    results = _evaluate_models(models, X, y)

    best_name, best_model = _select_best_model(models, results)
    best_model.fit(X, y)

    _print_results(results, best_name)

    all_metrics = {
        "complexity": complexity_metrics,
        "model_performance": results,
        "recommended_model": recommended,
        "selected_model": best_name,
    }

    return best_model, best_name, all_metrics


def _parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Adaptive RSM fitting")
    parser.add_argument("--csv", type=Path, help="CSV from sweep", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("."), help="Output directory"
    )
    return parser.parse_args()


def main() -> None:
    """Main function to run adaptive RSM fitting."""
    _set_matplotlib_publication_parameters()
    args = _parse_arguments()

    X, y, xlab, ylab, tag = _load_param_sweep(csv=args.csv)
    print(
        f"{tag} parameter space: {len(X)} points, response range [{y.min():.1f}, {y.max():.1f}]"
    )

    best_model, best_name, all_metrics = _fit_adaptive_model(X, y, tag)

    metrics_file = args.output_dir / f"{tag}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)

    plot_rsm_surface(
        model=best_model,
        X=X,
        label1=xlab,
        label2=ylab,
        fname=args.output_dir / f"{tag}_adaptive_rsm.svg",
    )

    print(f"\nBest model: {best_name}")
    print(f"Metrics saved: {metrics_file}")


if __name__ == "__main__":
    main()
