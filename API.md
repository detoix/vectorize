# Vectorize HTTP API

This repo includes a dependency-free HTTP server (`api_server.py`) that exposes the same pipeline as `run_pipeline.py`.

## Run

Use the repo virtualenv:

```bash
./venv/bin/python api_server.py --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Run the pipeline

### Option A: Upload an image and get DSL back (recommended)

This does not require the client to know server filesystem paths.

```bash
curl -sS http://127.0.0.1:8000/run \
  -F "image=@aps-lua.png" \
  -F "apiKey=$GOOGLE_API_KEY"
```

If you already have a mask and want to skip generation:

```bash
curl -sS http://127.0.0.1:8000/run \
  -F "image=@aps-lua.png" \
  -F "mask=@mask.png" \
  -F "skipGen=true" \
  -F "apiKey=$GOOGLE_API_KEY"
```

By default the response body is the DSL text (HTTP 200).

If you prefer JSON:

```bash
curl -sS http://127.0.0.1:8000/run \
  -F "image=@aps-lua.png" \
  -F "apiKey=$GOOGLE_API_KEY" \
  -F "format=json"
```

Note: the server runs mask generation and real-dimensions extraction in parallel by default to reduce latency.

### Option B: JSON request using local server paths

Request body is JSON. `image` is a path on the server machine (relative to repo root, unless absolute).

```bash
curl -sS http://127.0.0.1:8000/run \
  -H 'Content-Type: application/json' \
  -d '{
    "image": "aps-lua.png",
    "skipGen": true,
    "mask": "mask.png",
    "output": "output.dsl",
    "debugDir": "debug_pipeline",
    "apiKey": "'"$GOOGLE_API_KEY"'"
  }'
```

Supported fields (optional unless noted):

- `image` (required)
- `prompt`, `promptFile` / `prompt_file`
- `mask`, `output`, `debugDir` / `debug_dir`
- `skipGen` / `skip_gen`
- `apiKey` / `api_key` (or set `GOOGLE_API_KEY` in the server environment)
- `format` / `responseFormat` (`text` default, `json` supported)
- `maxDslChars` / `max_dsl_chars` (truncate returned DSL, default `200000`)

## Mask → DSL (no AI)

`POST /` accepts a 3‑color mask image plus a `dims` JSON payload, derives meters-per-pixel internally, and returns the DSL.

### Multipart form

- File: `mask` (or `image`)
- Field: `dims` (JSON string) or file field `dims` (JSON file)

```bash
curl -sS http://127.0.0.1:8000/ \
  -F "mask=@mask.png" \
  -F 'dims={"real-x":{"value":860,"unit":"cm"},"real-y":{"value":1070,"unit":"cm"}}'
```

### JSON

```bash
b64=$(base64 -w 0 mask.png)
curl -sS http://127.0.0.1:8000/ \
  -H 'Content-Type: application/json' \
  -d "{\"mask_b64\":\"$b64\",\"dims\":{\"real-x\":{\"value\":860,\"unit\":\"cm\"},\"real-y\":{\"value\":1070,\"unit\":\"cm\"}}}"
```
