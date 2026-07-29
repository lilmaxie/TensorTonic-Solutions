def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    p = [count / sum(reference_counts) for count in reference_counts]
    q = [count/ sum(production_counts) for count in production_counts]

    tvd = 0.5 * sum(abs(pi - qi) for pi, qi in zip(p, q))

    drift_detected  = tvd > threshold

    return {"score": float(tvd),
            "drift_detected": drift_detected
           }