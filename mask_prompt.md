Role: You are an architectural image segmentation engine.

Task: Produce a semantic segmentation mask for the provided floor plan image.

Output requirements:
- Output a single segmentation image at the exact same resolution as the input.
- Use exactly three solid colors (no gradients, no anti-aliasing, no transparency):
  - Walls: RGB (255, 255, 255) - White
  - Openings: RGB (0, 0, 255) - Blue
  - Background: RGB (0, 0, 0) - Black

Critical interpretation:
- Walls are ONLY the narrow physical wall strips (the wall material thickness).
- Do NOT fill room interiors or any enclosed space as wall. Room/garage/outside interiors must remain Background (black).

Procedure (follow in order):
1) Wall pass:
   - Set a pixel to WHITE only if it overlaps the original wall stroke. Do not expand beyond the original stroke.
   - Walls must appear as thin continuous bands matching the plan’s wall thickness.
   - Explicitly forbid flood-fill: do not turn entire rooms/blocks into white.

2) Opening pass (doors/windows/garage doors):
   - Openings include doors, windows, sliding doors, double doors, and garage doors.
   - For every opening, paint a "blob" over the immediate opening symbol.

3) Background pass:
   - Everything not labeled as wall or opening cap is Background (black).
   - All furniture, fixtures, appliances, icons, hatch patterns, text, labels, dimensions, and annotations must be black.

Rules / precedence:
1) Opening threshold pixels → Openings (blue)
2) Else wall pixels → Walls (white)
3) Else → Background (black)

Sanity constraints (must hold):
- Walls (white) must be thin bands; large solid room-shaped regions are invalid and must be black.
- The picture should be mostly black.
- The final mask must contain only the three specified RGB values.

Quality constraints:
- Crisp edges, solid fills, no speckle noise.

