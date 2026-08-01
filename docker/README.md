# Askme Docker Surface

`docker/` contains the image and Compose assets for the AskMe + LiteLLM
default product stack. ZeroClaw v0.1.7 is retained only as an explicit
experimental profile.

## Files

| File | Purpose |
| --- | --- |
| `.env.example` | Application/scoped-key template; copy to `docker/.env`. |
| `litellm.env.example` | Provider/master-secret template; copy to `docker/.env.litellm`. |
| `litellm-config.yaml` | Product aliases, routing, retry, and fallback policy owned by LiteLLM. |
| `docker-compose.litellm.yml` | Standalone LiteLLM + PostgreSQL bootstrap stack. |
| `docker-compose.yml` | Default AskMe stack plus opt-in `experimental-zeroclaw` services. |
| `docker-compose.edge-linux.yml` | Required Linux edge hardware mapping for `/dev/snd` and the host audio GID. |
| `docker-compose.prod.yml` | Production port, resource, and logging overrides. |
| `Dockerfile.askme` | AskMe image with an explicit `/app/config.yaml` runtime contract. |
| `Dockerfile.zeroclaw` | Reserved experimental ZeroClaw gateway image. |
| `docker-entrypoint.sh` | AskMe container entrypoint. |

## Bootstrap contract

Create and fill both environment files from the repository root. Provider and
master secrets stay only in `docker/.env.litellm`; the AskMe virtual key
and `ASKME_CONTROL_API_KEY` stay in `docker/.env`. The helpers
restrict both live env files before use: `chmod 600` on Linux/macOS and
an inheritance-disabled `icacls` ACL on Windows. Permission-hardening
failure is fatal.

Start the control plane and wait for readiness, generate the AskMe scoped key,
then start the default product stack. The standalone stack first runs a
network-disabled `--control-plane-only` gate for master, salt, and database
secrets; PostgreSQL and LiteLLM both wait for that gate:

```bash
docker compose --env-file docker/.env.litellm -f docker/docker-compose.litellm.yml up -d --wait litellm
export ASKME_AUDIO_GID="$(getent group audio | cut -d: -f3)"
docker compose --env-file docker/.env --env-file docker/.env.litellm \
  -f docker/docker-compose.yml \
  -f docker/docker-compose.edge-linux.yml \
  up -d
```

The default product `litellm-key-policy` gate additionally checks the AskMe
virtual key. It requires every protected value to be at least 24 characters,
rejects template markers and low-diversity/repeated values, enforces `sk-` on
LiteLLM access keys, and prevents reuse across master, AskMe, salt, and database
roles. The database password is restricted to URL-safe unreserved characters
because Compose interpolates it into `DATABASE_URL`. The gate never logs
credential values. AskMe waits for that gate and LiteLLM readiness. The
container healthcheck calls `/ready`
so degraded LLM, memory, or other registered components remove it from service;
`/healthz` remains the lightweight liveness endpoint.

The image deliberately keeps `edge_robot` as its default runtime. Before the
runtime starts, the entrypoint validates config-derived ASR/VAD/KWS/TTS model
requirements plus a usable PortAudio input and output. Missing models, missing
`/dev/snd`, or a wrong `ASKME_AUDIO_GID` blocks startup with exit code 78 and
safe diagnostics. A clean clone without models/audio therefore cannot become
ready; this is intentional fail-closed behavior, not a liveness/readiness
substitution.

The product Compose fixes AskMe routing to `http://litellm:4000/v1`.
`LITELLM_BASE_URL=http://127.0.0.1:4000/v1` in the application env is
only for native quickstart. Optional external robot URLs are blank by default
and, when enabled, must resolve from `askme-net`; container
`localhost` points back to AskMe itself.

## ZeroClaw experimental profile

ZeroClaw is excluded from default `docker compose up` and production
acceptance. To inspect only its LiteLLM model route, first provision a separate
`robot-action` virtual key in
`ZEROCLAW_LITELLM_VIRTUAL_KEY`, then opt in:

```powershell
docker compose --env-file docker/.env --env-file docker/.env.litellm -f docker/docker-compose.yml -f docker/docker-compose.edge-linux.yml --profile experimental-zeroclaw up -d
```

The experimental profile has its own fail-closed
`litellm-zeroclaw-key-policy` gate, extending the same checks with a distinct
ZeroClaw virtual key. The pinned v0.1.7 schema has no
MCP connector field, so a running gateway is not AskMe MCP integration evidence.

## Container network and production proxy

AskMe binds `0.0.0.0:8765` inside the container and requires
`ASKME_CONTROL_API_KEY`. The production overlay removes its host port;
attach the proxy to `askme-litellm_askme-net` and use
`http://askme:8765` as upstream.

## Quickstart

`deploy/quickstart.sh` and `deploy/quickstart.bat` expose
`docker` and `local` for the default AskMe + LiteLLM path.
`docker-zeroclaw` and `local-zeroclaw` are explicit
experimental commands. The local experimental helper maps
`ZEROCLAW_LITELLM_VIRTUAL_KEY` to `ZEROCLAW_API_KEY` only in
the gateway child process; it does not persist or pass the value as an argv
argument.

Production overlay:

```powershell
docker compose --env-file docker/.env --env-file docker/.env.litellm -f docker/docker-compose.yml -f docker/docker-compose.edge-linux.yml -f docker/docker-compose.prod.yml up -d
```

## Verification

```powershell
python -m pytest tests/test_deploy_paths.py tests/test_litellm_deployment.py tests/test_litellm_key_policy.py -q
```
