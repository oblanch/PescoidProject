"""Fit a quadratic response surface model (RSM) to a parameter sweep for
continuous responses.
"""

import argparse
from pathlib import Path

from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.pipeline import make_pipeline  # type: ignore
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures  # type: ignore

from pescoid_modelling.visualization import _set_matplotlib_publication_parameters
from pescoid_modelling.visualization.plot_rsm_phase_diagram import plot_rsm_surface


def _load_param_sweep(
    csv: Path, response: str
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Load a parameter sweep CSV, extract the x-axes and response
    variable.
    """
    df = pd.read_csv(csv)

    if {"activity", "flow"}.issubset(df.columns):
        x1, x2 = "activity", "flow"
        label1, label2 = r"$A$", r"$F$"
    elif {"beta", "r"}.issubset(df.columns):
        x1, x2 = "beta", "r"
        label1, label2 = r"$\beta$", r"$R$"
    elif {"r", "tau_m"}.issubset(df.columns):
        x1, x2 = "r", "tau_m"
        label1, label2 = r"$R$", r"$\tau_m$"
    else:
        raise ValueError("Cannot infer x-axes from CSV header.")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[response])
    X = df[[x1, x2]].to_numpy(float)
    y = df[response].to_numpy(float)
    return X, y, label1, label2


def fit_quadratic(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """Fit a quadratic response surface model (RSM) to the data:

    y = β0 + β1 * x1 + β2 * x2 + β11 * x1^2 + β22 * x2^2 + β12 * x1 * x2
    """
    model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=True), LinearRegression()
    )
    model.fit(X, y)
    return model


def _parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Quadratic RSM on sweep CSV")
    parser.add_argument("--csv", type=Path, help="CSV from sweep")
    parser.add_argument(
        "--response",
        default="onset_time",
        help="column to model (continuous)",
    )
    parser.add_argument(
        "--out",
        default="rsm.svg",
        help="save figure here",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to run the script."""
    _set_matplotlib_publication_parameters()
    args = _parse_arguments()

    X, y, xlab, ylab = _load_param_sweep(csv=args.csv, response=args.response)
    model = fit_quadratic(X, y)
    plot_rsm_surface(
        model=model,
        X=X,
        label1=xlab,
        label2=ylab,
        fname=args.out,
    )


if __name__ == "__main__":
    main()
