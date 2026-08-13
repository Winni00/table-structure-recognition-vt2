# Third-party software

Third-party source trees are not vendored because they are independent Git
repositories with their own licences. The project used the following
checkouts:

| Project | Repository | Commit used |
|---|---|---|
| TFLOP (public base) | <https://github.com/UpstageAI/TFLOP> | `d59ddf58a10fbaa22a7088b8f688bc6e694ae703` |
| Docling IBM Models | <https://github.com/docling-project/docling-ibm-models> | `9c5bcb4989aa8a5b657b315f9b371583276fbaec` |
| Docling | <https://github.com/docling-project/docling> | `53412ed43c9f5c77eb0c00a52b01560e9f9795fb` |

The experiments used the public TFLOP base commit shown above plus the
project-specific commit `584f6702591630b9180d0cff1ac447f11dc34343`. The latter
adds evaluation and inference diagnostics but is not part of the public
upstream repository. Its complete source patch is included at
`patches/tflop-evaluation-and-inference-diagnostics.patch`.

Clone the public dependencies separately:

```bash
mkdir -p external

git clone https://github.com/UpstageAI/TFLOP external/TFLOP
git -C external/TFLOP checkout d59ddf58a10fbaa22a7088b8f688bc6e694ae703
git -C external/TFLOP am \
  ../../patches/tflop-evaluation-and-inference-diagnostics.patch

git clone https://github.com/docling-project/docling-ibm-models external/docling-ibm-models
git -C external/docling-ibm-models checkout 9c5bcb4989aa8a5b657b315f9b371583276fbaec

git clone https://github.com/docling-project/docling external/docling
git -C external/docling checkout 53412ed43c9f5c77eb0c00a52b01560e9f9795fb
```

Review and comply with each project's licence before redistributing code or
model weights. Public datasets likewise remain governed by their own terms.
