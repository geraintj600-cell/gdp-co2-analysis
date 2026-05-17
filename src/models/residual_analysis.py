import matplotlib.pyplot as plt
import os


def residual_analysis(y_true, y_pred):

    # Calculate residuals
    residuals = y_true - y_pred

    # ---------- Residual Scatter Plot ----------

    plt.figure(figsize=(8,6))

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.5
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")

    plt.tight_layout()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    output_dir = os.path.join(
        BASE_DIR,
        "../../outputs"
    )

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "residual_plot.png"
    )

    plt.savefig(output_path)

    print(f"Saved residual plot to: {output_path}")

    plt.show()

    # ---------- Residual Histogram ----------

    plt.figure(figsize=(8,6))

    plt.hist(
        residuals,
        bins=30
    )

    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")

    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        "residual_histogram.png"
    )

    plt.savefig(output_path)

    print(f"Saved histogram to: {output_path}")

    plt.show()