import matplotlib.pyplot as plt
import os


def plot_model_comparison(gdp_r2, year_r2, log_r2):

    models = [
        "GDP",
        "GDP + Year",
        "Log GDP"
    ]

    scores = [
        gdp_r2,
        year_r2,
        log_r2
    ]

    plt.figure(figsize=(8,6))

    plt.bar(models, scores)

    plt.ylabel("R² Score")

    plt.title("Model Performance Comparison")

    plt.tight_layout()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    output_dir = os.path.join(
        BASE_DIR,
        "../../outputs"
    )

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "model_comparison.png"
    )

    plt.savefig(output_path)

    plt.show()