"""Prefer small visible objects, after clearing their approach corridors."""
from math import inf, isfinite, prod


def measured_dimensions(target):
    """Return observed envelope dimensions in metres, without catalog sizes."""
    measured = target.get('geometry', {}).get('dimensions_mm', ())
    if len(measured) == 3 and all(isfinite(value) and value > 0 for value in measured):
        return tuple(value / 1000. for value in measured)
    bounds = target.get('bounds', ())
    if len(bounds) == 2 and all(len(row) == 3 for row in bounds):
        dimensions = tuple(hi - lo for lo, hi in zip(*bounds))
        if all(isfinite(value) and value > 0 for value in dimensions):
            return dimensions
    return None


def prioritize_by_size(items, edge_x=None, hand_clearance=.060):
    """Select the smallest accessible item; dependencies override size priority.

    Edge-slide reception requires room on both sides of the shell. Ordinary
    grasp approaches also need a clear corridor, with a narrower hand margin.
    Objects nearer the table edge can obstruct either corridor and go first.
    Item identities are used only to report dependencies, never as priorities.
    """
    remaining = list(items)
    sizes, blockers = {}, {}
    edge_known = edge_x is not None and isfinite(edge_x)
    for index, item in enumerate(remaining):
        dimensions = measured_dimensions(item['target'])
        sizes[index] = prod(dimensions) if dimensions else inf
        blockers[index] = set()
        if not edge_known:
            continue
        low, high = item['target']['bounds']
        if edge_x >= low[0]:
            continue
        clearance = hand_clearance if item.get('approach_mode') == 'edge_slide' else hand_clearance / 2.
        for other_index, other in enumerate(remaining):
            if other_index == index:
                continue
            other_low, other_high = other['target']['bounds']
            # A blocker must be in front of this object, toward the table edge.
            # This also prevents mutual dependencies between adjacent objects.
            if (other_low[0] < low[0] and other_high[0] > edge_x
                    and other_low[1] < high[1] + clearance
                    and other_high[1] > low[1] - clearance):
                blockers[index].add(other_index)

    pending = set(range(len(remaining)))
    result = []
    while pending:
        accessible = [index for index in pending if not blockers[index] & pending]
        # The geometric dependencies are acyclic (strictly increasing front x).
        chosen = min(accessible, key=lambda index: (sizes[index], index))
        item = remaining[chosen]
        result.append(dict(item, order_priority={
            'volume_cm3': round(sizes[chosen] * 1e6, 1) if isfinite(sizes[chosen]) else None,
            'access_after': [remaining[index]['kind'] for index in sorted(blockers[chosen])],
            'rule': 'smallest_accessible_first',
        }))
        pending.remove(chosen)
    return result
