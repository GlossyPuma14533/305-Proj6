"""
Part 2 demo: show the "butterfly effect" of a tiny code error.

Idea:
- Simulate a small multi-stage data pipeline (load -> preprocess -> analyze).
- Introduce a very small bug in the normalization step (a "misspelled" variable).
- Run the pipeline twice:
    1) with correct code
    2) with the small error
- Compare the succession of outputs between the correct and buggy runs.
- Compute a "Code Butterfly Severity" (CBS) metric analogous to the distance
  between two Lorenz trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Stage 0: "Data loader" – this is fine in both runs
# ---------------------------------------------------------------------------

def load_data(n=500, random_seed=0):
    """
    Simulate loading raw data from a file or database.
    Here we just generate random values from a normal distribution.
    """
    rng = np.random.default_rng(random_seed)
    data = rng.normal(loc=50.0, scale=10.0, size=n)
    return data


# ---------------------------------------------------------------------------
# Stage 1: Preprocessing – CORRECT vs BUGGY version
# ---------------------------------------------------------------------------

def normalize_correct(data):
    """
    Correct normalization:
        z = (x - mean) / std
    """
    mean = np.mean(data)
    std = np.std(data)
    normalized = (data - mean) / std
    return normalized, mean, std


def normalize_buggy(data):
    """
    BUGGY normalization.

    Intentional small *code* error:
        We accidentally square the standard deviation in the denominator.

        Correct:   (data - mean) / std
        Buggy:     (data - mean) / (std ** 2)

    This is a one-character mistake in code (** instead of nothing),
    but it massively shrinks the values and propagates through later
    stages of the pipeline.
    """
    mean = np.mean(data)
    std = np.std(data)

    # Small textual change, big numerical effect:
    stdd = std ** 2           # <- BUG: should just be std
    normalized = (data - mean) / stdd

    return normalized, mean, stdd



# ---------------------------------------------------------------------------
# Stage 2: Analysis
# ---------------------------------------------------------------------------

def analyze(normalized):
    """
    Compute a few summary statistics from the normalized data.
    These stand in for "downstream components" that depend on Stage 1.
    """
    mean_norm = float(np.mean(normalized))
    var_norm = float(np.var(normalized))

    # Pretend this is some important model output (e.g., risk score)
    # based on how spread out the data is.
    risk_score = float(1.0 / (1.0 + np.exp(-var_norm)))

    return {
        "mean_norm": mean_norm,
        "var_norm": var_norm,
        "risk_score": risk_score,
    }


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(version="correct"):
    """
    Run the pipeline as either:
        version = "correct"  -> uses normalize_correct
        version = "buggy"    -> uses normalize_buggy

    Returns a dictionary with all key intermediate outputs.
    """
    # Stage 0: load data
    data = load_data()

    # Stage 1: preprocessing (correct or buggy)
    if version == "correct":
        normalized, mean, scale_used = normalize_correct(data)
    elif version == "buggy":
        normalized, mean, scale_used = normalize_buggy(data)
    else:
        raise ValueError("version must be 'correct' or 'buggy'")

    # Stage 2: analysis
    stats = analyze(normalized)

    output = {
        "data": data,
        "normalized": normalized,
        "mean": mean,
        "scale": scale_used,
        "stats": stats,
    }
    return output


# ---------------------------------------------------------------------------
# Code Butterfly Severity (CBS) metric
# ---------------------------------------------------------------------------

def compute_cbs(correct_run, buggy_run):
    """
    Compute a simple "Code Butterfly Severity" metric.

    We compare three key quantities between the correct and buggy runs:
        1. mean of normalized data
        2. variance of normalized data
        3. risk score

    We treat those as a 3D vector and measure the relative difference
    between the correct and buggy executions, similar to the distance
    between two Lorenz trajectories.
    """
    vec_correct = np.array([
        correct_run["stats"]["mean_norm"],
        correct_run["stats"]["var_norm"],
        correct_run["stats"]["risk_score"],
    ])

    vec_buggy = np.array([
        buggy_run["stats"]["mean_norm"],
        buggy_run["stats"]["var_norm"],
        buggy_run["stats"]["risk_score"],
    ])

    diff = vec_buggy - vec_correct
    base_norm = np.linalg.norm(vec_correct) + 1e-12
    abs_severity = np.linalg.norm(diff)
    rel_severity = abs_severity / base_norm

    return abs_severity, rel_severity, vec_correct, vec_buggy


# ---------------------------------------------------------------------------
# Visualization of output succession
# ---------------------------------------------------------------------------

def make_plots(correct_run, buggy_run):
    """
    Create side-by-side plots showing how the outputs differ
    when everything is correct vs when the small bug is present.

    This corresponds to:
        1) Succession of outputs (normalized values, stats)
        2) Visual comparison of correct vs buggy behavior
    """
    norm_correct = correct_run["normalized"]
    norm_buggy = buggy_run["normalized"]

    var_c = correct_run["stats"]["var_norm"]
    var_b = buggy_run["stats"]["var_norm"]

    risk_c = correct_run["stats"]["risk_score"]
    risk_b = buggy_run["stats"]["risk_score"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Top-left: histogram of normalized data (correct)
    axes[0, 0].hist(norm_correct, bins=30, alpha=0.9)
    axes[0, 0].set_title("Stage 1 Output (Normalized Data)\nCORRECT Code")
    axes[0, 0].set_xlabel("Normalized value")
    axes[0, 0].set_ylabel("Frequency")

    # Top-right: histogram of normalized data (buggy)
    axes[0, 1].hist(norm_buggy, bins=30, alpha=0.9)
    axes[0, 1].set_title("Stage 1 Output (Normalized Data)\nBUGGY Code")
    axes[0, 1].set_xlabel("Normalized value")
    axes[0, 1].set_ylabel("Frequency")

    # Bottom-left: bar chart of variance comparison
    axes[1, 0].bar(["Correct", "Buggy"], [var_c, var_b])
    axes[1, 0].set_title("Stage 2 Output: Variance of Normalized Data")
    axes[1, 0].set_ylabel("Variance")

    # Bottom-right: bar chart of risk score comparison
    axes[1, 1].bar(["Correct", "Buggy"], [risk_c, risk_b])
    axes[1, 1].set_title("Stage 2 Output: Risk Score")
    axes[1, 1].set_ylabel("Risk score")

    fig.suptitle("Butterfly Effect of a Small Code Error", fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure so you can paste it into your report / slides
    #ig.savefig("butterfly_code_outputs.png", dpi=300)
    #print("\nSaved figure as 'butterfly_code_outputs.png' in the current folder.\n")

    plt.show()


# ---------------------------------------------------------------------------
# Console explanation: "succession of code snippets"
# ---------------------------------------------------------------------------

def print_code_succession():
    """
    Textual description of which code snippets / stages are affected.
    You can copy these bullet points into your write-up.
    """
    print("Succession of affected code components (conceptual):")
    print("  Stage 0 - data_loader: load_data(...)  [NOT affected]")
    print("  Stage 1 - preprocessing: normalize_correct(...) vs normalize_buggy(...)  [BUG HERE]")
    print("  Stage 2 - analysis: analyze(...) uses normalized values and thus gets corrupted")
    print("  Stage 3 - main / reporting: prints risk score, plots results -> user sees bad output")
    print()



# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    # Run pipeline with correct code
    correct_run = run_pipeline(version="correct")

    # Run pipeline with small bug
    buggy_run = run_pipeline(version="buggy")

    # Compute Code Butterfly Severity
    abs_sev, rel_sev, vec_c, vec_b = compute_cbs(correct_run, buggy_run)

    # Print succession of code snippets (text description)
    print_code_succession()

    # Print numeric results for comparison
    print("=== NUMERIC COMPARISON OF OUTPUTS ===")
    print(f"Correct normalized mean:   {vec_c[0]:.6f}")
    print(f"Buggy normalized mean:     {vec_b[0]:.6f}")
    print()
    print(f"Correct normalized var:    {vec_c[1]:.6f}")
    print(f"Buggy normalized var:      {vec_b[1]:.6f}")
    print()
    print(f"Correct risk score:        {vec_c[2]:.6f}")
    print(f"Buggy risk score:          {vec_b[2]:.6f}")
    print()
    print("=== CODE BUTTERFLY SEVERITY (CBS) ===")
    print(f"Absolute severity (L2 distance):   {abs_sev:.6f}")
    print(f"Relative severity (normalized):    {rel_sev:.6f}")
    print("(Relative severity is analogous to the distance between two Lorenz")
    print(" trajectories that start from almost the same initial condition.)")
    print()

    # Create visualizations
    make_plots(correct_run, buggy_run)


if __name__ == "__main__":
    main()

