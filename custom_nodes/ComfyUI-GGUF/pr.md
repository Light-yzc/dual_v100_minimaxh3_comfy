# Summary

This adds support for Ideogram GGUF models.

# What Changed

- Added `ideogram` to the supported image GGUF architectures.
- Added Ideogram model detection to the converter.
- Added GGUF dtype handling needed by Ideogram inference.
- Fixed the Ideogram inference failure where a packed GGUF weight dtype caused a byte tensor to reach CUDA linear.
- Adjusted BF16 GGUF loading so Ideogram can start inference faster.

# Error Fixed

Before this change, Ideogram GGUF models could load but failed during sampling with:

```text
"addmm_cuda" not implemented for 'Byte'
```

# Tests

Ran Python compile checks:

```text
python -m py_compile dequant.py ops.py loader.py nodes.py tools\convert.py
```

Checked a synthetic GGUF linear forward:

- packed storage stayed `torch.uint8`
- reported dtype was `torch.bfloat16`
- input dtype was `torch.bfloat16`
- output dtype was `torch.bfloat16`

Checked BF16 loading behavior with finite values.

# Notes

Full ComfyUI inference was not run from this environment.

Other GGUF quant types still use the existing dequant paths.
