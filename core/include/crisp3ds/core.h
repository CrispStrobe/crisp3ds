#ifndef CRISP3DS_CORE_H
#define CRISP3DS_CORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CRISP3DS_ABI_VERSION 1u
#define CRISP3DS_PROJECT_SCHEMA_VERSION 1u

typedef enum crisp3ds_status {
  CRISP3DS_OK = 0,
  CRISP3DS_INVALID_ARGUMENT = 1,
  CRISP3DS_INVALID_PROJECT = 2,
  CRISP3DS_BUFFER_TOO_SMALL = 3,
  CRISP3DS_UNAVAILABLE = 4,
  CRISP3DS_INTERNAL_ERROR = 5
} crisp3ds_status;

/* All JSON results are UTF-8 and NUL-terminated. `required` includes the NUL.
 * Pass output=NULL, capacity=0 to query the required size. This ABI neither
 * owns caller memory nor retains input pointers. */
uint32_t crisp3ds_abi_version(void);
crisp3ds_status crisp3ds_capabilities_json(char *output, size_t capacity,
                                           size_t *required);
crisp3ds_status crisp3ds_validate_project_json(const char *input, size_t length,
                                                char *output, size_t capacity,
                                                size_t *required);
crisp3ds_status crisp3ds_geometry_diagnostic_json(char *output, size_t capacity,
                                                   size_t *required);
crisp3ds_status crisp3ds_reconstruct_json(const char *input, size_t length,
                                          char *output, size_t capacity,
                                          size_t *required);
/* Reads project_path and its referenced PNG/JPEG images. Result matrices are
 * row-major object-to-camera transforms; translations use millimetres.
 * Caller owns all buffers. No source file is modified. */
crisp3ds_status crisp3ds_estimate_poses_json(const char *project_path,
                                             char *output, size_t capacity,
                                             size_t *required);
/* Computes once and returns a heap-owned NUL-terminated report. Release with
 * crisp3ds_free; size includes the NUL. This avoids executing image work twice
 * when the caller cannot preallocate a result buffer. */
crisp3ds_status crisp3ds_estimate_poses_alloc(const char *project_path,
                                              char **output, size_t *size);
/* Re-estimates all image poses, then returns a masked sparse point report.
 * Available only in an OpenCV-enabled build; release output with crisp3ds_free. */
crisp3ds_status crisp3ds_reconstruct_sparse_alloc(const char *project_path,
                                                  char **output, size_t *size);
void crisp3ds_free(void *pointer);

#ifdef __cplusplus
}
#endif

#endif
