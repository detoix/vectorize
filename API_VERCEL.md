## Vercel endpoint: mask → DSL

This repo can be deployed to Vercel with a single endpoint that runs `mask2dsl.py`.

### Endpoint

`POST /api/vectorize`

### Request (multipart/form-data)

- `mask` (file): PNG/JPEG mask image (3-color mask expected by `mask2dsl.py`)
- `scale` (text): meters-per-pixel (e.g. `0.01`)

Example:

```bash
curl -X POST "https://YOUR_APP.vercel.app/api/vectorize" \
  -F "mask=@mask.png" \
  -F "scale=0.01"
```

### Request (application/json)

```json
{
  "mask_b64": "<base64-encoded image bytes>",
  "scale": 0.01
}
```

### Response

- `200 text/plain`: DSL text
- `4xx/5xx application/json`: error payload

