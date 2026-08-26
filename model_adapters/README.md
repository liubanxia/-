# Model adapters

Project Phoenix Core keeps model integration behind small adapters so the public pipeline can stay independent of any particular checkpoint, vendor, or deployment machine.

## Public adapter contract

`PhoenixExpertAdapter` in `expert_base.py` defines the minimal lifecycle used by specialist adapters:

- `validate_assets()` checks that required local assets exist.
- `load()` initializes the model/runtime.
- `run(case)` performs one inference pass for a Phoenix case.
- `unload()` releases the loaded model state.
- `describe()` exposes lightweight adapter metadata.

A new specialist adapter should normally subclass `PhoenixExpertAdapter` and define stable `model_id` and `task` values.

```python
from model_adapters.expert_base import PhoenixExpertAdapter


class ExampleAdapter(PhoenixExpertAdapter):
    model_id = "example_local_model"
    task = "example_task"

    def load(self):
        self.validate_assets()
        # Load a local runtime/checkpoint here.
        self.model = object()
        self.loaded = True
        return self

    def run(self, case):
        if not self.loaded:
            self.load()
        return {
            "lesions": [
                {
                    "label_name": "synthetic_finding",
                    "confidence": 0.90,
                    "series_uid": "synthetic-series",
                    "image_index": 0,
                    "box": [10, 10, 20, 20],
                }
            ]
        }
```

The example is intentionally runtime-agnostic. Real checkpoints are expected to remain outside the repository and be supplied through local paths.

## Result shape

Adapters that produce lesion candidates should return a dictionary containing a `lesions` list. `core.result_fusion.fuse_results()` accepts common fields including:

- label: `finding`, `label_name`, `name`, `type`, or textual `label`
- confidence: `confidence`, `score`, or `label_score`
- location: `series_uid`, `image_index`, `point`, `box`, or `box_3d`
- 3D geometry: `world_point_lps`, `geometry_mode`, and `voxel_count`

Unknown fields are preserved in lesion metadata where possible. This allows adapters to expose model-specific information without coupling the shared result contract to one model family.

## Repository boundary

Public adapters must not embed or commit:

- patient DICOM data or reports
- hospital PACS endpoints, credentials, or deployment configuration
- private model weights, API keys, or access tokens
- machine-specific absolute paths

Use synthetic or properly de-identified inputs for examples and tests. Keep model weights and clinical deployment configuration outside the repository.

## Adding an adapter

1. Implement the smallest adapter needed for the model task.
2. Keep model loading separate from case routing and result fusion.
3. Return structured results rather than model-specific UI objects.
4. Add a synthetic regression test for routing or output normalization when practical.
5. Document any optional runtime dependencies without committing the corresponding weights.

The goal is a stable public integration surface: local models may change, while Phoenix routing, result fusion, and clinician-reviewed output interfaces remain testable without private clinical assets.
