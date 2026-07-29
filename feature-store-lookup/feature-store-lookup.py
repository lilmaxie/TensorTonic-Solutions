def feature_store_lookup(feature_store, requests, defaults):
    """
    Join offline user features with online request-time features.
    """
    results = []

    for req in requests:
        user_id = req["user_id"]
        online_features = req.get("online_features", {})

        offline_features = feature_store.get(user_id, defaults)

        combined_features = {**offline_features, **online_features}

        results.append(combined_features)

    return results