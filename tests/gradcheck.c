#include "tg.h"

#include <stdio.h>

int main(void) {
    tg_tensor *t = tg_tensor_create(4U);
    if (t == NULL) {
        fprintf(stderr, "gradcheck: allocation failed\n");
        return 1;
    }

    if (tg_tensor_size(t) != 4U) {
        fprintf(stderr, "gradcheck: size mismatch\n");
        tg_tensor_destroy(t);
        return 1;
    }

    if (tg_version_string() == NULL) {
        fprintf(stderr, "gradcheck: version string is null\n");
        tg_tensor_destroy(t);
        return 1;
    }

    tg_tensor_destroy(t);

    printf("gradcheck: ok\n");
    return 0;
}
