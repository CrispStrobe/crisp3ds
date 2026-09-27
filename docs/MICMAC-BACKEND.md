# MicMac CPU backend: shipping rejection

2026-09-27 policy clarification: the project now uses AGPL and allows GPL/AGPL
comparison experiments. The historical GPL exclusion below is not a blanket
ban on those experiments. The separate NEC noncommercial restriction still
requires resolution; no MicMac build or reconstruction has been run. See the
[current comparison plan](PIPELINE-COMPARISON-PLAN.md).

Inspection date: 2026-09-26. The inspected source is upstream `micmacIGN/micmac` commit `f8fe432101fb1852f8c4d8546504a5278f6ef57f`, cloned into ignored `.local-tools/micmac-src/`. No MicMac executable was built, run, linked, or added to the application.

The top-level [CeCILL-B license](https://github.com/micmacIGN/micmac/blob/f8fe432101fb1852f8c4d8546504a5278f6ef57f/LICENSE.md) does not cover every file on the normal `mm3d` build path under that license alone:

| Source at this commit | Finding | Build-path evidence |
| --- | --- | --- |
| [`include/api/cox_roy.h`](https://github.com/micmacIGN/micmac/blob/f8fe432101fb1852f8c4d8546504a5278f6ef57f/include/api/cox_roy.h) and [`src/optim/cox_roy.cpp`](https://github.com/micmacIGN/micmac/blob/f8fe432101fb1852f8c4d8546504a5278f6ef57f/src/optim/cox_roy.cpp) | NEC copyright permission says use, copy, modify and distribute **for non-commercial purposes**. | The header is included by `include/StdAfx.h`; `src/optim/Sources.cmake` includes `cox_roy.cpp`, and `src/CMakeLists.txt` includes that source group in `elise`, which `mm3d` links. |
| [`src/uti_phgrm/GraphCut/QPBO-v1.4/QPBO.cpp`](https://github.com/micmacIGN/micmac/blob/f8fe432101fb1852f8c4d8546504a5278f6ef57f/src/uti_phgrm/GraphCut/QPBO-v1.4/QPBO.cpp) and sibling files | Explicit GPL-3.0-or-later header and bundled `GPL.TXT`. | `QPBO-v1.4/Sources.cmake` lists four `.cpp` files; `src/uti_phgrm/Sources.cmake` appends them to `uti_phgrm_Src_Files`, which enters `elise`. |
| [`CodeExterne/ANN/License.txt`](https://github.com/micmacIGN/micmac/blob/f8fe432101fb1852f8c4d8546504a5278f6ef57f/CodeExterne/ANN/License.txt) | ANN is LGPL-2.1; project policy requires separate review. | Top-level CMake always adds the ANN subdirectory; `src/CBinaires/CMakeLists.txt` links `mm3d` with ANN. |
| [`include/api/win_regex.h`](https://github.com/micmacIGN/micmac/blob/f8fe432101fb1852f8c4d8546504a5278f6ef57f/include/api/win_regex.h) and `src/util/win_regex.c` | GPL-2.0-or-later text. | `src/CMakeLists.txt` adds `win_regex.c` for Windows; `include/api/el_regex.h` includes its header when `ELISE_windows` is set. It is not a macOS build input. |

This source revision fails the project's GPL and noncommercial exclusion policy on its ordinary CPU path. A headless CMake configuration or `BUILD_ONLY_ELISE_MM3D` does not remove the above `elise` sources. Any future MicMac attempt would need an exact-file alternative build graph, code provenance and rights review, plus a demonstrated photo-to-mesh run. It cannot be represented as an App Store-compatible shipping backend on the evidence here. The inspected clone occupies about 402 MiB of ignored local space; no global dependencies were installed.
