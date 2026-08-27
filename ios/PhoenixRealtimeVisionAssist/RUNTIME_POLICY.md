# LiteView runtime policy

LiteView may ship with up to 1 GiB of offline resources, but runtime cost is intentionally bounded.

- Broadcast Extension: model-free, under 12 MiB packaged, no frame retention, low-rate Apple Vision only.
- Main app models: opt-in lazy load only, unloadable, and blocked under low-power or serious thermal conditions.
- Default realtime mode: visible-content analysis only. Legacy hidden-position/map extrapolation is disabled.
- Audio-level and screen-cue analysis are opt-in and disabled by default.
- Cross-process state: App Group when available, with an entitlement-free compact libnotify fallback.
- No video, screenshot, audio, or analysis-history persistence.

Package size is a storage budget, not a resident-memory target. Extra offline resources must remain dormant until explicitly requested.
