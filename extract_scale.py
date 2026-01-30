import argparse
import json
import sys
from typing import Optional, Tuple

import cv2
import numpy as np


def wall_mask_from_image_rgb(img_rgb: np.ndarray) -> np.ndarray:
    # Walls are now BLACKish. Loosened range to handle non-exact black.
    return cv2.inRange(img_rgb, np.array([0, 0, 0]), np.array([80, 80, 80]))


def wall_bbox_from_mask(mask_wall: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Returns (xmin, ymin, xmax, ymax) in pixel coordinates for the union of wall pixels.
    """
    ys, xs = np.nonzero(mask_wall > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No wall pixels found; cannot compute wall bounding box.")
    xmin = int(xs.min())
    xmax = int(xs.max())
    ymin = int(ys.min())
    ymax = int(ys.max())
    return xmin, ymin, xmax, ymax


def _unit_to_meters_factor(unit_raw: str) -> float:
    unit = unit_raw.strip().lower()
    if unit in {"m", "meter", "meters", "metre", "metres"}:
        return 1.0
    if unit in {"cm", "centimeter", "centimeters", "centimetre", "centimetres"}:
        return 0.01
    if unit in {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}:
        return 0.001
    if unit in {"ft", "foot", "feet"}:
        return 0.3048
    if unit in {"in", "inch", "inches"}:
        return 0.0254
    raise ValueError(f"Unsupported unit: {unit_raw!r} (supported: m, cm, mm, ft, in).")


def _parse_dimension_to_meters(value: object, unit: object, *, label: str) -> float:
    if value is None or unit is None:
        raise ValueError(f"{label} must include both 'value' and 'unit'.")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{label}.value must be a number; got {value!r}.") from e
    if numeric_value <= 0:
        raise ValueError(f"{label}.value must be positive; got {numeric_value}.")
    if not isinstance(unit, str):
        raise ValueError(f"{label}.unit must be a string; got {unit!r}.")
    return numeric_value * _unit_to_meters_factor(unit)


def load_real_dims_from_json(path_or_dash: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Loads real-world dimensions from a JSON file (or stdin via '-').

    Expected structure (either key may be omitted):
      {
        "real-x": {"value": 12.3, "unit": "m"},
        "real-y": {"value": 8.7,  "unit": "m"}
      }

    Returns (real_width_m, real_height_m) as floats in meters.
    """
    if path_or_dash == "-":
        payload = json.load(sys.stdin)
    else:
        with open(path_or_dash, "r") as f:
            payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Real-dimensions JSON must be an object at the top level.")

    real_width_m = None
    real_height_m = None

    if "real-x" in payload and payload["real-x"] is not None:
        node = payload["real-x"]
        if not isinstance(node, dict):
            raise ValueError("'real-x' must be an object with 'value' and 'unit'.")
        real_width_m = _parse_dimension_to_meters(node.get("value"), node.get("unit"), label="real-x")

    if "real-y" in payload and payload["real-y"] is not None:
        node = payload["real-y"]
        if not isinstance(node, dict):
            raise ValueError("'real-y' must be an object with 'value' and 'unit'.")
        real_height_m = _parse_dimension_to_meters(node.get("value"), node.get("unit"), label="real-y")

    if real_width_m is None and real_height_m is None:
        raise ValueError("JSON must include at least one of 'real-x' or 'real-y'.")
    return real_width_m, real_height_m


def load_real_dims_from_payload(payload: object) -> Tuple[Optional[float], Optional[float]]:
    """
    Loads real-world dimensions from an in-memory JSON-like object.

    Expected structure (either key may be omitted):
      {
        "real-x": {"value": 12.3, "unit": "m"},
        "real-y": {"value": 8.7,  "unit": "m"}
      }

    Returns (real_width_m, real_height_m) as floats in meters.
    """
    if not isinstance(payload, dict):
        raise ValueError("Real-dimensions payload must be an object at the top level.")

    real_width_m = None
    real_height_m = None

    if "real-x" in payload and payload["real-x"] is not None:
        node = payload["real-x"]
        if not isinstance(node, dict):
            raise ValueError("'real-x' must be an object with 'value' and 'unit'.")
        real_width_m = _parse_dimension_to_meters(node.get("value"), node.get("unit"), label="real-x")

    if "real-y" in payload and payload["real-y"] is not None:
        node = payload["real-y"]
        if not isinstance(node, dict):
            raise ValueError("'real-y' must be an object with 'value' and 'unit'.")
        real_height_m = _parse_dimension_to_meters(node.get("value"), node.get("unit"), label="real-y")

    if real_width_m is None and real_height_m is None:
        raise ValueError("Payload must include at least one of 'real-x' or 'real-y'.")
    return real_width_m, real_height_m


def derive_isotropic_meters_per_pixel(
    *,
    wall_bbox_px: Tuple[int, int, int, int],
    real_width_m: Optional[float],
    real_height_m: Optional[float],
) -> Tuple[float, Optional[float], Optional[float], int, int]:
    xmin, ymin, xmax, ymax = wall_bbox_px
    px_w = int(xmax - xmin)
    px_h = int(ymax - ymin)
    if px_w <= 0 or px_h <= 0:
        raise ValueError(f"Invalid wall bbox for scale derivation: bbox={wall_bbox_px}.")

    if real_width_m is None and real_height_m is None:
        raise ValueError("Provide at least one of --real-width-m or --real-height-m.")

    mpp_x = None
    mpp_y = None
    if real_width_m is not None:
        if real_width_m <= 0:
            raise ValueError(f"--real-width-m must be positive; got {real_width_m}.")
        mpp_x = float(real_width_m) / float(px_w)
    if real_height_m is not None:
        if real_height_m <= 0:
            raise ValueError(f"--real-height-m must be positive; got {real_height_m}.")
        mpp_y = float(real_height_m) / float(px_h)

    if mpp_x is not None and mpp_y is not None:
        mpp = min(mpp_x, mpp_y)
    else:
        mpp = mpp_x if mpp_x is not None else mpp_y

    return mpp, mpp_x, mpp_y, px_w, px_h


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive meters-per-pixel from wall union bbox and real dimensions.")
    parser.add_argument("--input", required=True, help="RGB image input (same mask encoding as mask2dsl.py).")
    parser.add_argument("--real-x", type=float, default=None, help="Real-world X dimension (numeric). Use with --real-x-unit.")
    parser.add_argument("--real-x-unit", type=str, default="m", help="Unit for --real-x (m, cm, mm, ft, in).")
    parser.add_argument("--real-y", type=float, default=None, help="Real-world Y dimension (numeric). Use with --real-y-unit.")
    parser.add_argument("--real-y-unit", type=str, default="m", help="Unit for --real-y (m, cm, mm, ft, in).")
    parser.add_argument(
        "--real-width-m",
        type=float,
        default=None,
        help="DEPRECATED: use --real-x/--real-x-unit instead.",
    )
    parser.add_argument(
        "--real-height-m",
        type=float,
        default=None,
        help="DEPRECATED: use --real-y/--real-y-unit instead.",
    )
    parser.add_argument(
        "--real-json",
        default=None,
        help="Path to JSON with real dimensions using keys 'real-x'/'real-y' (or '-' for stdin).",
    )
    parser.add_argument(
        "--format",
        choices=("plain", "json"),
        default="plain",
        help="Output format. plain prints meters_per_pixel as a single number.",
    )
    parser.add_argument("--debug-dir", default=None, help="Optional directory to write debug artifacts.")
    args = parser.parse_args()

    real_width_m = args.real_width_m
    real_height_m = args.real_height_m

    if args.real_x is not None:
        real_width_m = _parse_dimension_to_meters(args.real_x, args.real_x_unit, label="real-x")
    if args.real_y is not None:
        real_height_m = _parse_dimension_to_meters(args.real_y, args.real_y_unit, label="real-y")

    if args.real_json is not None:
        json_width_m, json_height_m = load_real_dims_from_json(args.real_json)
        if real_width_m is None:
            real_width_m = json_width_m
        if real_height_m is None:
            real_height_m = json_height_m

    img = cv2.imread(args.input)
    if img is None:
        raise SystemExit(f"Could not load {args.input}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask_wall = wall_mask_from_image_rgb(img_rgb)
    wall_bbox_px = wall_bbox_from_mask(mask_wall)

    mpp, mpp_x, mpp_y, px_w, px_h = derive_isotropic_meters_per_pixel(
        wall_bbox_px=wall_bbox_px,
        real_width_m=real_width_m,
        real_height_m=real_height_m,
    )

    if args.debug_dir:
        import os

        os.makedirs(args.debug_dir, exist_ok=True)
        debug_img = cv2.cvtColor(mask_wall, cv2.COLOR_GRAY2BGR)
        xmin, ymin, xmax, ymax = wall_bbox_px
        cv2.rectangle(debug_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        cv2.imwrite(f"{args.debug_dir}/00_wall_mask.png", mask_wall)
        cv2.imwrite(f"{args.debug_dir}/00_wall_bbox.png", debug_img)
        with open(f"{args.debug_dir}/00_scale.json", "w") as f:
            json.dump(
                {
                    "wall_bbox_px": list(wall_bbox_px),
                    "px_w": px_w,
                    "px_h": px_h,
                    "real_width_m": real_width_m,
                    "real_height_m": real_height_m,
                    "mpp_x": mpp_x,
                    "mpp_y": mpp_y,
                    "meters_per_pixel": mpp,
                },
                f,
                indent=2,
                sort_keys=True,
            )

    if args.format == "plain":
        print(f"{mpp:.10f}")
    else:
        print(
            json.dumps(
                {
                    "meters_per_pixel": mpp,
                    "mpp_x": mpp_x,
                    "mpp_y": mpp_y,
                    "wall_bbox_px": list(wall_bbox_px),
                    "px_w": px_w,
                    "px_h": px_h,
                }
            )
        )


if __name__ == "__main__":
    main()
