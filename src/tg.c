#include "tg.h"

const char *tg_version_string(void) {
    return "tinygradc 0.1.0";
}

tg_status tg_backward(tg_tensor *loss) {
    (void)loss;
    return TG_ERR_UNIMPLEMENTED;
}
