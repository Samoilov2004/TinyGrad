#include "tg.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TG_PARAM_IO_MAGIC     "TGPARAM1"
#define TG_PARAM_IO_MAGIC_LEN 8u
#define TG_PARAM_IO_VERSION   1u

static int tg_is_heap_param(const tg_tensor *t) {
    if (!t || !t->data) {
        return 0;
    }
    if (t->rows <= 0 || t->cols <= 0) {
        return 0;
    }
    if (!t->owns_self || !t->owns_data) {
        return 0;
    }
    return 1;
}

static tg_status tg_write_exact(FILE *f, const void *ptr, size_t size) {
    if (!f || (!ptr && size > 0)) {
        return TG_ERR_INVALID_ARGUMENT;
    }
    if (size == 0) {
        return TG_OK;
    }
    return (fwrite(ptr, 1, size, f) == size) ? TG_OK : TG_ERR_INVALID_ARGUMENT;
}

static tg_status tg_read_exact(FILE *f, void *ptr, size_t size) {
    if (!f || (!ptr && size > 0)) {
        return TG_ERR_INVALID_ARGUMENT;
    }
    if (size == 0) {
        return TG_OK;
    }
    return (fread(ptr, 1, size, f) == size) ? TG_OK : TG_ERR_INVALID_ARGUMENT;
}

static void tg_reset_loaded_param_state(tg_tensor *t) {
    size_t n;
    size_t bytes;

    if (!t) {
        return;
    }

    tg_zero_grad(t);

    n = tg_numel(t);
    if (n == 0) {
        return;
    }

    bytes = n * sizeof(float);

    if (t->opt1) {
        memset(t->opt1, 0, bytes);
    }
    if (t->opt2) {
        memset(t->opt2, 0, bytes);
    }
}

tg_status tg_params_save(const char *path, tg_tensor **params, int n) {
    FILE *f;
    uint32_t version;
    uint32_t count;
    int i;

    if (!path || !params || n < 0) {
        return TG_ERR_INVALID_ARGUMENT;
    }

    f = fopen(path, "wb");
    if (!f) {
        return TG_ERR_INVALID_ARGUMENT;
    }

    if (tg_write_exact(f, TG_PARAM_IO_MAGIC, TG_PARAM_IO_MAGIC_LEN) != TG_OK) {
        fclose(f);
        return TG_ERR_INVALID_ARGUMENT;
    }

    version = TG_PARAM_IO_VERSION;
    count = (uint32_t)n;

    if (tg_write_exact(f, &version, sizeof(version)) != TG_OK ||
        tg_write_exact(f, &count, sizeof(count)) != TG_OK) {
        fclose(f);
        return TG_ERR_INVALID_ARGUMENT;
    }

    for (i = 0; i < n; ++i) {
        tg_tensor *p = params[i];
        uint32_t rows;
        uint32_t cols;
        size_t numel;
        size_t bytes;

        if (!tg_is_heap_param(p)) {
            fclose(f);
            return TG_ERR_INVALID_ARGUMENT;
        }

        rows = (uint32_t)p->rows;
        cols = (uint32_t)p->cols;
        numel = tg_numel(p);
        bytes = numel * sizeof(float);

        if (tg_write_exact(f, &rows, sizeof(rows)) != TG_OK ||
            tg_write_exact(f, &cols, sizeof(cols)) != TG_OK ||
            tg_write_exact(f, p->data, bytes) != TG_OK) {
            fclose(f);
            return TG_ERR_INVALID_ARGUMENT;
        }
    }

    fclose(f);
    return TG_OK;
}

tg_status tg_params_load(const char *path, tg_tensor **params, int n) {
    FILE *f;
    char magic[TG_PARAM_IO_MAGIC_LEN];
    uint32_t version;
    uint32_t count;
    int i;

    if (!path || !params || n < 0) {
        return TG_ERR_INVALID_ARGUMENT;
    }

    f = fopen(path, "rb");
    if (!f) {
        return TG_ERR_INVALID_ARGUMENT;
    }

    if (tg_read_exact(f, magic, TG_PARAM_IO_MAGIC_LEN) != TG_OK) {
        fclose(f);
        return TG_ERR_INVALID_ARGUMENT;
    }

    if (memcmp(magic, TG_PARAM_IO_MAGIC, TG_PARAM_IO_MAGIC_LEN) != 0) {
        fclose(f);
        return TG_ERR_INVALID_ARGUMENT;
    }

    if (tg_read_exact(f, &version, sizeof(version)) != TG_OK ||
        tg_read_exact(f, &count, sizeof(count)) != TG_OK) {
        fclose(f);
        return TG_ERR_INVALID_ARGUMENT;
    }

    if (version != TG_PARAM_IO_VERSION || count != (uint32_t)n) {
        fclose(f);
        return TG_ERR_INVALID_ARGUMENT;
    }

    for (i = 0; i < n; ++i) {
        tg_tensor *p = params[i];
        uint32_t rows;
        uint32_t cols;
        size_t numel;
        size_t bytes;

        if (!tg_is_heap_param(p)) {
            fclose(f);
            return TG_ERR_INVALID_ARGUMENT;
        }

        if (tg_read_exact(f, &rows, sizeof(rows)) != TG_OK ||
            tg_read_exact(f, &cols, sizeof(cols)) != TG_OK) {
            fclose(f);
            return TG_ERR_INVALID_ARGUMENT;
        }

        if ((int)rows != p->rows || (int)cols != p->cols) {
            fclose(f);
            return TG_ERR_INVALID_ARGUMENT;
        }

        numel = tg_numel(p);
        bytes = numel * sizeof(float);

        if (tg_read_exact(f, p->data, bytes) != TG_OK) {
            fclose(f);
            return TG_ERR_INVALID_ARGUMENT;
        }

        tg_reset_loaded_param_state(p);
    }

    fclose(f);
    return TG_OK;
}
