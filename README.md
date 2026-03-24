# AnemiaScan

Research-oriented AI pipeline and deployment stack for non-invasive anemia screening from conjunctiva and nail-bed images.

## Model Summary

- Primary model: EfficientNet-B4 dual-head network for hemoglobin regression and four-class anemia severity classification
- Fine-tuning design: ImageNet-pretrained backbone, head warm-up followed by partial backbone unfreezing
- Inference features: MC-dropout uncertainty intervals and Grad-CAM visual explanations
- Deployment status: conjunctiva model is live first; nail-bed deployment depends on uploaded weights

## Data And Evaluation

- Dataset strategy: unified Hugging Face dataset combining public conjunctiva and nail-bed sources with planned ICMR field data from Tumakuru
- Canonical input: 380×380 RGB images
- Tracked metrics: MAE, RMSE, Pearson r, AUC, F1, sensitivity, specificity, Bland-Altman analysis
- Design targets from the project specification: MAE ≤ 1.0 g/dL, Pearson r ≥ 0.85, AUC ≥ 0.85, sensitivity/specificity ≥ 80%

## Attribution

Concept, design, build, training, deployment, testing by: Dr Siddalingaiah H S, Professor, Community Medicine, Shridevi Institute of Medical Sciences and Research Hospital, Tumkur, hssling@yahoo.com, 8941087719.

ORCID: 0000-0002-4771-8285
