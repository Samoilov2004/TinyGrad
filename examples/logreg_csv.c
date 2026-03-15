#include "tg.h"

#include <stdio.h>

int main(int argc, char **argv) {
    printf("logreg_csv example\n");
    printf("tinygradc: %s\n", tg_version_string());

    if (argc < 2) {
        printf("usage: %s <file.csv>\n", argv[0]);
        return 0;
    }

    printf("csv path: %s\n", argv[1]);
    printf("status: placeholder example\n");
    return 0;
}
