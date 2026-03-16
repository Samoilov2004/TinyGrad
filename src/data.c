#include "tg.h"

#include <stdlib.h>
#include <string.h>

static unsigned int tg_xorshift32(unsigned int *state) {
    unsigned int x;

    if (!state || *state == 0u) {
        x = 2463534242u;
    } else {
        x = *state;
    }

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    *state = x;
    return x;
}

static void tg_shuffle_indices(int *indices, int n, unsigned int *state) {
    int i;

    if (!indices || n <= 1 || !state) {
        return;
    }

    for (i = n - 1; i > 0; --i) {
        unsigned int r = tg_xorshift32(state);
        int j = (int)(r % (unsigned int)(i + 1));

        {
            int tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }
    }
}

tg_status tg_dataloader_init(
    tg_dataloader *loader,
    const float *x,
    const float *y,
    int rows,
    int x_cols,
    int y_cols,
    int batch_size,
    bool shuffle
) {
    int i;

    if (!loader || !x) {
        return TG_ERR_INVALID_ARGUMENT;
    }
    if (rows <= 0 || x_cols <= 0 || batch_size <= 0) {
        return TG_ERR_INVALID_ARGUMENT;
    }
    if ((y == NULL && y_cols != 0) || (y != NULL && y_cols <= 0)) {
        return TG_ERR_INVALID_ARGUMENT;
    }

    memset(loader, 0, sizeof(*loader));

    loader->dataset.x = x;
    loader->dataset.y = y;
    loader->dataset.rows = rows;
    loader->dataset.x_cols = x_cols;
    loader->dataset.y_cols = y_cols;

    loader->batch_size = batch_size;
    loader->shuffle = shuffle;
    loader->cursor = 0;
    loader->rng_state = 1u;

    loader->indices = (int *)malloc((size_t)rows * sizeof(int));
    if (!loader->indices) {
        memset(loader, 0, sizeof(*loader));
        return TG_ERR_OOM;
    }

    for (i = 0; i < rows; ++i) {
        loader->indices[i] = i;
    }

    return TG_OK;
}

void tg_dataloader_reset(tg_dataloader *loader, unsigned int seed) {
    int i;

    if (!loader || !loader->indices || loader->dataset.rows <= 0) {
        return;
    }

    loader->cursor = 0;
    loader->rng_state = (seed != 0u) ? seed : 1u;

    for (i = 0; i < loader->dataset.rows; ++i) {
        loader->indices[i] = i;
    }

    if (loader->shuffle) {
        tg_shuffle_indices(loader->indices, loader->dataset.rows, &loader->rng_state);
    }
}

bool tg_dataloader_next(
    tg_dataloader *loader,
    tg_arena *arena,
    tg_tensor **out_x,
    tg_tensor **out_y
) {
    int start;
    int end;
    int batch_rows;
    tg_tensor *x_batch;
    tg_tensor *y_batch = NULL;
    int r;

    if (!loader || !arena || !out_x) {
        return false;
    }
    if (!loader->indices || loader->dataset.rows <= 0 || loader->dataset.x_cols <= 0) {
        return false;
    }
    if (loader->cursor >= loader->dataset.rows) {
        return false;
    }

    start = loader->cursor;
    end = start + loader->batch_size;
    if (end > loader->dataset.rows) {
        end = loader->dataset.rows;
    }
    batch_rows = end - start;

    x_batch = tg_tensor_tmp(arena, batch_rows, loader->dataset.x_cols, false);
    if (!x_batch) {
        return false;
    }

    if (loader->dataset.y && loader->dataset.y_cols > 0 && out_y) {
        y_batch = tg_tensor_tmp(arena, batch_rows, loader->dataset.y_cols, false);
        if (!y_batch) {
            return false;
        }
    }

    for (r = 0; r < batch_rows; ++r) {
        int src_row = loader->indices[start + r];
        const float *src_x = loader->dataset.x + ((size_t)src_row * (size_t)loader->dataset.x_cols);
        float *dst_x = x_batch->data + ((size_t)r * (size_t)loader->dataset.x_cols);

        memcpy(dst_x, src_x, (size_t)loader->dataset.x_cols * sizeof(float));

        if (y_batch) {
            const float *src_y = loader->dataset.y + ((size_t)src_row * (size_t)loader->dataset.y_cols);
            float *dst_y = y_batch->data + ((size_t)r * (size_t)loader->dataset.y_cols);

            memcpy(dst_y, src_y, (size_t)loader->dataset.y_cols * sizeof(float));
        }
    }

    loader->cursor = end;
    *out_x = x_batch;
    if (out_y) {
        *out_y = y_batch;
    }

    return true;
}

int tg_dataloader_num_batches(const tg_dataloader *loader) {
    if (!loader || loader->dataset.rows <= 0 || loader->batch_size <= 0) {
        return 0;
    }

    return (loader->dataset.rows + loader->batch_size - 1) / loader->batch_size;
}

void tg_dataloader_destroy(tg_dataloader *loader) {
    if (!loader) {
        return;
    }

    free(loader->indices);
    loader->indices = NULL;
    loader->cursor = 0;
    loader->rng_state = 0u;
    memset(&loader->dataset, 0, sizeof(loader->dataset));
    loader->batch_size = 0;
    loader->shuffle = false;
}
