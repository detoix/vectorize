You are given an architectural floor-plan image.

Task:
Extract the real-world OUTER dimensions of the building footprint:

- real-x: overall horizontal width of the building footprint
- real-y: overall vertical depth of the building footprint
  The footprint is the axis-aligned bounding box of the union of EXTERIOR walls (thick/dark wall strokes).
  IMPORTANT: Do NOT include the white margins or paper edges in the real-x/real-y dimensions.
  Exclude detached site elements (paths, landscaping, fences).

How to read:

1. Prefer the OUTERMOST overall dimension strings placed outside the plan that span the full footprint in X and in Y.
2. Ignore internal room dimensions and any dimensioning that refers to the plot boundary rather than the building.

Infer the unit (must be correct):
A) If the drawing explicitly labels units near dimensions (mm, cm, m) or in a title/legend, use that.
B) If units are not labeled, infer from building-scale plausibility and drafting conventions:

- Convert each candidate assuming mm, cm, m, ft, and in, and keep only interpretations where each axis is plausibly 3–100 meters for a residential footprint.
- Prefer the interpretation that best matches common plan conventions:
  - Unlabeled 3–4 digit overall values like 900–3000 are typically centimeters (e.g., 1300 => 13.00 m).
  - Unlabeled 4–5 digit overall values like 3000–60000 are typically millimeters (e.g., 13000 => 13.00 m).
  - Values already written with decimals or explicit “m” are typically meters.
- If more than one interpretation is plausible, choose the one consistent with the majority of other dimension annotations in the image (e.g., repeated 250/150 style values for openings often indicate cm).

Output requirements (strict):
Return ONLY valid JSON with exactly this structure (no extra keys, no commentary).
Omit an axis entirely if you cannot determine it confidently.

{
"real-x": { "value": <number>, "unit": "<mm|cm|m|ft|in>" },
"real-y": { "value": <number>, "unit": "<mm|cm|m|ft|in>" }
}

Notes:

- "value" must be numeric (float).
- "unit" must be exactly one of: "mm", "cm", "m", "ft", "in".
- If you convert during inference, output the chosen unit and the corresponding value in that unit (not forced to mm).
