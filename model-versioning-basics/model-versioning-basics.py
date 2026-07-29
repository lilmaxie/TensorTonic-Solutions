def promote_model(models):
    """
    Decide which model version to promote to production.
    """
    best_model = None
    for m in models:
        if best_model is None:
            best_model = m
            continue
            
        if m["accuracy"] > best_model["accuracy"]:
            best_model = m
        elif m["accuracy"] == best_model["accuracy"]:
            if m["latency"] < best_model["latency"]:
                best_model = m
            elif m["latency"] == best_model["latency"]:
                if m["timestamp"] > best_model["timestamp"]:
                    best_model = m

    return best_model["name"]