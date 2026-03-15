#include "tg.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int is_aligned(const void *ptr, size_t align)
{
    if (align == 0) {
        return 1;
    }
    return (((uintptr_t)ptr) & (uintptr_t)(align - 1)) == 0;
}

int main(void)
{
    const char *version;
    tg_tensor *t = NULL;
    tg_arena *arena = NULL;
    void *p1;
    void *p2;
    void *p3;
    void *q1;
    void *q2;
    size_t i;
    int rc = 1;

    version = tg_version_string();
    if (version == NULL) {
        fprintf(stderr, "gradcheck: tg_version_string returned NULL\n");
        goto cleanup;
    }

    t = tg_tensor_create(4);
    if (t == NULL) {
        fprintf(stderr, "gradcheck: tg_tensor_create failed\n");
        goto cleanup;
    }

    if (tg_tensor_size(t) != 4) {
        fprintf(stderr, "gradcheck: unexpected tensor size\n");
        goto cleanup;
    }

    if (tg_tensor_data(t) == NULL) {
        fprintf(stderr, "gradcheck: tg_tensor_data returned NULL\n");
        goto cleanup;
    }

    arena = tg_arena_create(64);
    if (arena == NULL) {
        fprintf(stderr, "gradcheck: tg_arena_create failed\n");
        goto cleanup;
    }

    p1 = tg_arena_alloc(arena, 8, 8);
    if (p1 == NULL || !is_aligned(p1, 8)) {
        fprintf(stderr, "gradcheck: first arena allocation failed/alignment error\n");
        goto cleanup;
    }
    memset(p1, 0x11, 8);

    p2 = tg_arena_alloc(arena, 24, 16);
    if (p2 == NULL || !is_aligned(p2, 16)) {
        fprintf(stderr, "gradcheck: second arena allocation failed/alignment error\n");
        goto cleanup;
    }
    memset(p2, 0x22, 24);

    p3 = tg_arena_alloc(arena, 200, 32);
    if (p3 == NULL || !is_aligned(p3, 32)) {
        fprintf(stderr, "gradcheck: growth allocation failed/alignment error\n");
        goto cleanup;
    }
    memset(p3, 0x33, 200);

    for (i = 0; i < 256; ++i) {
        int *tmp = (int *)tg_arena_alloc(arena, sizeof(int) * 3, _Alignof(int));
        if (tmp == NULL) {
            fprintf(stderr, "gradcheck: repeated arena allocation failed at %zu\n", i);
            goto cleanup;
        }
        tmp[0] = (int)i;
        tmp[1] = (int)i + 1;
        tmp[2] = (int)i + 2;
    }

    tg_arena_reset(arena);

    q1 = tg_arena_alloc(arena, 8, 8);
    if (q1 == NULL || !is_aligned(q1, 8)) {
        fprintf(stderr, "gradcheck: allocation after reset failed\n");
        goto cleanup;
    }

    if (q1 != p1) {
        fprintf(stderr, "gradcheck: reset did not reuse first chunk as expected\n");
        goto cleanup;
    }

    q2 = tg_arena_alloc(arena, 48, 16);
    if (q2 == NULL || !is_aligned(q2, 16)) {
        fprintf(stderr, "gradcheck: second allocation after reset failed\n");
        goto cleanup;
    }
    memset(q2, 0x44, 48);

    rc = 0;
    printf("gradcheck: ok\n");

cleanup:
    tg_arena_destroy(arena);
    tg_tensor_destroy(t);
    return rc;
}
