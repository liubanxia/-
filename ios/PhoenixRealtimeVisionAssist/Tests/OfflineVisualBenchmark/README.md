# LiteView Offline Visual Benchmark

This folder is an offline-only visual benchmark. It exists to answer one question before any runtime packaging: can the current person detector reliably find visible human figures in representative Delta Force Mobile gameplay imagery?

## Scope

- Offline evaluation only.
- No ReplayKit, PiP, live overlay, input automation, or gameplay integration.
- Labels are generic visible-human labels only. Do not encode team/enemy identity.
- Public image URLs are referenced; copyrighted images are not vendored into the repository.

## Scene buckets

1. near_indoor — close/medium human figure in indoor corridor or doorway.
2. medium_outdoor — medium-size human figure outdoors.
3. far_outdoor — small/distant human figure in open terrain.
4. optic_view — human figure visible through optic/scope.
5. partial_occlusion — partially occluded or edge-of-frame human figure.
6. negative_or_ambiguous — no clearly visible human target; used to measure false positives.

## Required offline report

For every sample record:

- model_loaded
- inference_succeeded
- decoded
- predicted_person_count
- highest_confidence
- manual_visible_person_count
- matched_visible_person_count
- false_positive_count
- notes

Aggregate:

- recall = matched_visible_person_count / manual_visible_person_count
- false positives per image
- recall by scene bucket
- median and p95 inference latency

Do not call the detector usable from build/packaging success alone. The detector only passes this first gate if the offline report shows useful recall across near, medium, far and optic-view buckets without excessive false positives.
