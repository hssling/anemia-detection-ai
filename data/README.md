# AnemiaScan Dataset

Unified dataset of conjunctival and nail-bed images for anemia screening.

**HuggingFace:** `hssling/anemia-conjunctiva-nailbed`

## Sources

See `dataset_registry.yaml` for all source datasets.

## Format

`metadata.csv` columns:
- `image_id` — unique image identifier
- `image_path` — local path to image
- `site` — `conjunctiva` or `nailbed`
- `hb_value` — hemoglobin (g/dL), reference standard
- `anemia_class` — WHO 2011: `normal`, `mild`, `moderate`, `severe`
- `age_group` — `adult` (>=15 yrs) or `child` (5-14 yrs)
- `source_dataset` — originating dataset ID
- `image_quality_score` — 0-1 (0 = rejected)
- `split` — `train`, `val`, or `test`

## Running the Pipeline

```bash
KAGGLE_USERNAME=... KAGGLE_KEY=... HF_TOKEN=... bash data/data_pipeline.sh
```

Or trigger via GitHub Actions -> Data Pipeline.
