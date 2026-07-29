def deduplicate(records, key_columns, strategy):
    """
    Deduplicate records by key columns using the given strategy.
    """
    if not records:
        return []

    groups = {}
    unique_key_order = []

    for record in records:
        key = tuple(record.get(col) for col in key_columns)

        if key not in groups:
            groups[key] = []
            unique_key_order.append(key)

        groups[key].append(record)

    deduplicated_records = []
    for key in unique_key_order:
        group = groups[key]

        if strategy == "first":
            selected = group[0]
        elif strategy == "last":
            selected = group[-1]
        elif strategy == "most_complete":
            selected = min(group,
                           key=lambda r: sum(1 for v in r.values() if v is None))
        else:
            raise ValueError(f"Error Strategy: {strategy}")

        deduplicated_records.append(selected)
        
    return deduplicated_records