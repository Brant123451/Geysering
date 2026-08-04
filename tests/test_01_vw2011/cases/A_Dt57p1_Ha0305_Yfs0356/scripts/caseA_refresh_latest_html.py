"""Refresh a verified Case-A comparison HTML with a new 1D manifest.

Only the ``data1`` payload is replaced.  The UTF-8 user interface, original
2D frame manifest, and interactive controls are retained byte-for-byte from
the selected presentation template.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CASE = HERE.parent
TSTAR_PER_SECOND = 1.2269382978252207
TOWER_HEIGHT = 0.610


def load_frames(path: Path) -> list[dict]:
    frames = json.loads(path.read_text(encoding="utf-8"))
    for frame in frames:
        time = float(frame["time"])
        yint = float(frame["itop"])
        yfs = float(frame["wtop"])
        frame.update(
            Tstar=time * TSTAR_PER_SECOND,
            Yint=yint,
            Yfs=yfs,
            YintStar=yint / TOWER_HEIGHT,
            YfsStar=yfs / TOWER_HEIGHT,
        )
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--one-d-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    template = args.template if args.template.is_absolute() else CASE / args.template
    index = args.one_d_index if args.one_d_index.is_absolute() else CASE / args.one_d_index
    output = args.output if args.output.is_absolute() else CASE / args.output

    html = template.read_text(encoding="utf-8")
    start = html.index("const data1=") + len("const data1=")
    end = html.index(", data2=", start)
    payload = json.dumps(load_frames(index), ensure_ascii=False)
    output.write_text(html[:start] + payload + html[end:], encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
