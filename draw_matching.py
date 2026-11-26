"""Match a ticket to a draw within one day without replacing explicit targets."""

from typing import TypeVar

Draw = TypeVar("Draw")


def match_draw(
    draws: list[Draw], *, draw_id: str | None = None,
    draw_time: str | None = None, draw_index: int | None = None,
) -> Draw | None:
    """Use ID/time metadata first; positional fallback is for untimed history.

    Callers must provide draws from a single date in chronological order.
    Legacy entries without any target metadata use the last draw of the day.
    """
    if not draws:
        return None
    if draw_id:
        return next((d for d in reversed(draws)
                     if str(getattr(d, "draw_identifier", "")) == str(draw_id)), None)
    if draw_time:
        matches = [d for d in draws if getattr(d, "draw_time", None) == draw_time]
        if matches:
            return matches[-1]
        if any(getattr(d, "draw_time", None) for d in draws):
            return None
        if draw_index is None:
            return None
    if draw_index is not None:
        if isinstance(draw_index, bool) or not isinstance(draw_index, int):
            return None
        return draws[draw_index - 1] if 1 <= draw_index <= len(draws) else None
    return draws[-1]
