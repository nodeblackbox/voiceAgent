"""Startup banner: a big block-letter rendering of the user's name, shown once while the agent loads
(tui.py mounts it into the log before the models finish loading; ui.py prints it above the Live view).

Hand-rolled rather than a pyfiglet dependency — only the 7 letters "ANAS NASSEUR" needs are drawn, so a
tiny built-in font is simpler than pulling in a whole figlet font pack for one word.
"""
from __future__ import annotations

from rich.text import Text

_ROWS = 7
_GLYPH_W = 7

# Each glyph is 7 rows x 7 cols of "#" (filled) / " " (empty); only the letters "ANAS NASSEUR" needs.
_FONT: dict[str, list[str]] = {
    "A": [
        " ##### ",
        "##   ##",
        "##   ##",
        "#######",
        "##   ##",
        "##   ##",
        "##   ##",
    ],
    "N": [
        "##   ##",
        "###  ##",
        "###  ##",
        "## # ##",
        "##  ###",
        "##  ###",
        "##   ##",
    ],
    "S": [
        " ######",
        "##     ",
        "##     ",
        " ##### ",
        "     ##",
        "     ##",
        "###### ",
    ],
    "E": [
        "#######",
        "##     ",
        "##     ",
        "#####  ",
        "##     ",
        "##     ",
        "#######",
    ],
    "U": [
        "##   ##",
        "##   ##",
        "##   ##",
        "##   ##",
        "##   ##",
        "##   ##",
        " ##### ",
    ],
    "R": [
        "###### ",
        "##   ##",
        "##   ##",
        "###### ",
        "##  ## ",
        "##   ##",
        "##   ##",
    ],
    " ": [" " * 3] * _ROWS,
}

_BLOCK = "█"
_GAP = 1  # columns of blank space between glyphs

# gradient stops, left to right — the same accents the TUI already uses for "you" / input / "agent"
_STOPS = [(0x38, 0xbd, 0xf8), (0xa7, 0x8f, 0xfa), (0x4a, 0xde, 0x80)]


def _lerp(a: int, b: int, t: float) -> int:
    return round(a + (b - a) * t)


def _color_at(t: float) -> str:
    """t in [0, 1] across the whole banner width -> a hex color interpolated through _STOPS."""
    t = max(0.0, min(1.0, t))
    seg = t * (len(_STOPS) - 1)
    i = min(int(seg), len(_STOPS) - 2)
    lt = seg - i
    r0, g0, b0 = _STOPS[i]
    r1, g1, b1 = _STOPS[i + 1]
    return f"#{_lerp(r0, r1, lt):02x}{_lerp(g0, g1, lt):02x}{_lerp(b0, b1, lt):02x}"


def banner_width(text: str) -> int:
    return sum(_GLYPH_W + _GAP for _ in text) - _GAP


def render_banner(text: str = "ANAS NASSEUR", tagline: str = "voice agent") -> Text:
    """A multi-line, left-to-right gradient block-letter Text. Caller centers it if it wants to."""
    text = text.upper()
    width = banner_width(text)
    out = Text()
    for row in range(_ROWS):
        col = 0
        for ch in text:
            glyph = _FONT.get(ch, _FONT[" "])[row]
            for cell in glyph:
                if cell != " ":
                    out.append(_BLOCK, style=_color_at(col / max(1, width - 1)))
                else:
                    out.append(" ")
                col += 1
            out.append(" " * _GAP)
            col += _GAP
        out.append("\n")
    if tagline:
        pad = max(0, (width - len(tagline)) // 2)
        out.append(" " * pad + tagline, style="dim italic")
    return out


def small_banner(text: str = "Anas Nasseur", tagline: str = "voice agent") -> Text:
    """One-line gradient fallback for terminals too narrow for the block letters."""
    out = Text()
    for i, ch in enumerate(text):
        out.append(ch, style=f"bold {_color_at(i / max(1, len(text) - 1))}")
    if tagline:
        out.append("  ·  " + tagline, style="dim italic")
    return out


def startup_banner(width: int, text: str = "ANAS NASSEUR", tagline: str = "voice agent") -> Text:
    """The block-letter banner if it fits in `width` columns, else the one-line fallback."""
    if width >= banner_width(text) + 2:
        return render_banner(text, tagline)
    return small_banner(tagline=tagline)
