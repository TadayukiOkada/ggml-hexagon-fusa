# ggml-hexagon-fusa

Out-of-tree ggml Hexagon backend for HTP v73 on FuSa firmware. Public repository.

- Never commit board-specific information: build-tree paths, host names, adb serials, firmware files or their symbol lists. Board scripts live outside this repo.
- Claude may write commit messages (end with `Assisted-by: Claude <model>`). Push only after the user confirms.
- Do not modify the `llama.cpp/` or `NeMo-Speech.cpp/` submodule; NeMo changes are `asr/patches/`. Upstream backend snapshots go to the `vendor` branch unmodified; changes go to `main`.
- Host library and skel must always come from the same build (build ID check).
- Keep other HTP generations building; do not remove `__HVX_ARCH__` branches.
