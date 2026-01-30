# Furniture API Update

The `/furniture` endpoint has been updated to support real-world unit conversion (meters).

## Endpoint

`POST /furniture`

## Request

**Content-Type**: `multipart/form-data`

### Parameters

| Field             | Type   | Required | Description                        |
| :---------------- | :----- | :------- | :--------------------------------- |
| **`image`**       | File   | **Yes**  | The input image (PNG/JPG).         |
| **`dims`**        | JSON   | No       | Real-world dimensions for scaling. |
| **`real-x`**      | Number | No       | Real X dimension (e.g. `10.5`).    |
| **`real-y`**      | Number | No       | Real Y dimension.                  |
| **`real-x-unit`** | String | No       | Unit for X (default `m`).          |
| **`real-y-unit`** | String | No       | Unit for Y (default `m`).          |

### The `dims` Parameter

If provided, the API will calculate the scale (`meters_per_pixel`) and return all coordinate values in **meters**.
If omitted, the API returns raw **pixels**.

Format:

```json
{
  "real-x": { "value": 10.5, "unit": "m" }
}
```

_Note: You only need to provide one known dimension (`real-x` or `real-y`) to establish the scale._

## Response

### Case 1: With `dims` (Meters)

```json
{
  "units": "meters",
  "meters_per_pixel": 0.05,
  "furniture": [
    {
      "type": "bed",
      "x": 2.5, // meters
      "y": 3.1, // meters
      "length": 2.0, // meters
      "width": 1.6, // meters
      "rotation": 90
    }
  ]
}
```

### Case 2: Without `dims` (Pixels)

```json
{
  "units": "pixels",
  "meters_per_pixel": null,
  "furniture": [
    {
      "type": "bed",
      "x": 500, // pixels
      "y": 620, // pixels
      "length": 400, // pixels
      "width": 320, // pixels
      "rotation": 90
    }
  ]
}
```

## Example (JavaScript)

```javascript
const formData = new FormData();
formData.append("image", imageFile);

// OPTIONAL: Send dims to get results in METERS
formData.append(
  "dims",
  JSON.stringify({
    "real-x": { value: 12, unit: "m" },
  }),
);

const response = await fetch("http://localhost:8000/furniture", {
  method: "POST",
  body: formData,
});

const data = await response.json();
```
