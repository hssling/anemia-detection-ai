# AnemiaScan Dataset

Unified dataset of conjunctival and nail-bed images for anemia screening.

**HuggingFace:** `hssling/anemia-conjunctiva-nailbed`

## Sources

See `dataset_registry.yaml` for all source datasets.

Current source types:
- `kaggle`: direct Kaggle CLI download
- `mendeley`: public Mendeley Data zip download, including nested `.rar` extraction
- `manual`: contact/request dataset source; not auto-downloaded by the pipeline

Hb-regression-capable public source currently wired:
- `cp_anemic_ghana` (`m53vz6b7fx`): conjunctival images with measured Hb levels from Ghana

Binary-only public source currently wired:
- `ghana_fingernails` (`2xx4j3kjg2`): nail-bed images from Ghana, suitable for binary anemia classification or pretraining, not Hb regression

High-value contact/manual source:
- `imagehb_contact`: multi-site Hb-labeled pediatric dataset with conjunctiva, palm, tongue, and nailbed images; full dataset requires direct request via the project page

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
