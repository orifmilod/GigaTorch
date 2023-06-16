def one_hot(hot_item, items):
    return [1 if hot_item is item else 0 for item in items]
