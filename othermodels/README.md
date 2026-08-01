# Other Model Integrations

This directory contains DIVE implementations that do not belong to the
primary LLaVA-1.5/1.6 code path.

- `qwen2_vl/` contains the importable `Qwen2VL_DIVE` model implementation,
  its Qwen2-VL dependencies, and the lmms-eval adapter.
- `llava_ov/` contains the importable LLaVA-OneVision lmms-eval adapter. It
  reuses the primary DIVE LLaVA implementation and the shared Transformers
  dependency under `qwen2_vl/`.

Install the repository from its root before using either adapter:

```bash
pip install -e .
pip install -e lmms-eval
```

The regular lmms-eval model names remain `qwen2_vl` and `llava_onevision`.
