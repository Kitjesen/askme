# Compatibility Facades

This package owns migration helpers for historical import paths.

## Rule

New code should not choose an import path from a legacy facade. Use the owner
package named by the facade registry instead.

The machine-readable registry is `askme.compat.legacy_facades.LEGACY_FACADES`.
Each entry records:

- `legacy_path`: old import path kept working.
- `canonical_path`: module that now owns the implementation.
- `new_code_import`: path or factory new code should use.
- `owner`: package responsible for the capability.
- `reason`: why the compatibility path exists.

## Current High-Risk Facades

| Legacy path | Canonical owner | New-code entrypoint |
| --- | --- | --- |
| `askme.voice.runtime_bridge` | `voice_gateway` / `providers` | `providers.build_voice_runtime_bridge()` plus `VoiceGatewayService` |
| `askme.voice.orchestration.runtime_bridge` | `voice_gateway` / `providers` | `providers.build_voice_runtime_bridge()` plus `VoiceGatewayService` |
| `askme.voice_gateway.runtime_bridge` | `providers.voice_runtime` | `providers.build_voice_runtime_bridge()` |
| `askme.voice.input.address_detector` | `robot_interaction` | `robot_interaction.AddressDetector` |
| `askme.voice.interaction.*` | `robot_interaction` | `robot_interaction` |
| `askme.interaction.*` | `robot_interaction` | `robot_interaction` |
| `askme.pipeline.reactions.state_led_bridge` | `providers.led` | `providers.build_status_led()` |
| `askme.robot.telemetry.ota_bridge` | `telemetry` | `telemetry.ota_bridge` |
| `askme.robot.telemetry.pubsub` | `interfaces.bus` | `interfaces.bus.BusBackend` |

## Test Guard

`tests/test_package_migration_compat.py` verifies that registered facades remain
importable and stay thin: they may re-export, but they must not grow new classes
or functions.
