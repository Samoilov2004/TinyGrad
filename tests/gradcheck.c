#include "tg.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define TG_GC_ARENA_BYTES (64 * 1024)
#define TG_GC_EPS         1e-3f
#define TG_GC_REL_TOL     3e-2f
#define TG_GC_ABS_TOL     3e-3f
#define TG_GC_SAMPLES     5
#define TG_GC_SEED        1337u

typedef float (*tg_loss_eval_fn)(tg_arena *arena, void *ctx);

typedef struct tg_gc_param_ref {
    const char *name;
    tg_tensor *tensor;
} tg_gc_param_ref;

/* ----------------------------- RNG helpers ----------------------------- */

static float tg_rand_uniform(float lo, float hi) {
    float u = (float)rand() / (float)RAND_MAX;
    return lo + (hi - lo) * u;
}

static int tg_rand_int(int lo_inclusive, int hi_inclusive) {
    if (hi_inclusive <= lo_inclusive) {
        return lo_inclusive;
    }
    return lo_inclusive + (rand() % (hi_inclusive - lo_inclusive + 1));
}

static void tg_fill_uniform(tg_tensor *t, float lo, float hi) {
    size_t n;
    size_t i;

    if (!t || !t->data) {
        return;
    }

    n = tg_numel(t);
    for (i = 0; i < n; ++i) {
        t->data[i] = tg_rand_uniform(lo, hi);
    }
}

static void tg_fill_binary01(tg_tensor *t) {
    size_t n;
    size_t i;

    if (!t || !t->data) {
        return;
    }

    n = tg_numel(t);
    for (i = 0; i < n; ++i) {
        t->data[i] = (rand() & 1) ? 1.0f : 0.0f;
    }
}

static void tg_zero_grads(tg_tensor **params, int n) {
    int i;

    if (!params || n <= 0) {
        return;
    }

    for (i = 0; i < n; ++i) {
        tg_zero_grad(params[i]);
    }
}

/* --------------------------- Numeric helpers --------------------------- */

static float tg_max3f(float a, float b, float c) {
    float m = a;
    if (b > m) {
        m = b;
    }
    if (c > m) {
        m = c;
    }
    return m;
}

static float tg_rel_error(float analytic, float numeric) {
    float denom = tg_max3f(1.0f, fabsf(analytic), fabsf(numeric));
    return fabsf(analytic - numeric) / denom;
}

static bool tg_is_close(float analytic, float numeric, float abs_tol, float rel_tol) {
    float abs_err = fabsf(analytic - numeric);
    float rel_err = tg_rel_error(analytic, numeric);
    return (abs_err <= abs_tol) || (rel_err <= rel_tol);
}

static float tg_numeric_grad_at(
    tg_arena *arena,
    tg_loss_eval_fn eval_fn,
    void *ctx,
    tg_tensor *param,
    size_t idx,
    float eps
) {
    float orig;
    float loss_pos;
    float loss_neg;

    if (!arena || !eval_fn || !ctx || !param || !param->data) {
        return NAN;
    }

    orig = param->data[idx];

    param->data[idx] = orig + eps;
    loss_pos = eval_fn(arena, ctx);
    tg_arena_reset(arena);

    param->data[idx] = orig - eps;
    loss_neg = eval_fn(arena, ctx);
    tg_arena_reset(arena);

    param->data[idx] = orig;

    if (!isfinite(loss_pos) || !isfinite(loss_neg)) {
        return NAN;
    }

    return (loss_pos - loss_neg) / (2.0f * eps);
}

static bool tg_gradcheck_params(
    const char *test_name,
    tg_arena *arena,
    tg_loss_eval_fn eval_fn,
    void *ctx,
    const tg_gc_param_ref *params,
    int nparams,
    int samples_per_param,
    float eps,
    float abs_tol,
    float rel_tol
) {
    int i;
    bool ok = true;

    if (!test_name || !arena || !eval_fn || !ctx || !params || nparams <= 0) {
        return false;
    }

    for (i = 0; i < nparams; ++i) {
        const tg_gc_param_ref *pr = &params[i];
        tg_tensor *p = pr->tensor;
        size_t numel;
        int s;

        if (!p || !p->data || !p->grad) {
            printf("[FAIL] %s: param %s is invalid or has no grad buffer\n",
                   test_name, pr->name ? pr->name : "(unnamed)");
            return false;
        }

        numel = tg_numel(p);
        if (numel == 0) {
            printf("[FAIL] %s: param %s has zero elements\n",
                   test_name, pr->name ? pr->name : "(unnamed)");
            return false;
        }

        for (s = 0; s < samples_per_param; ++s) {
            size_t idx = (size_t)tg_rand_int(0, (int)numel - 1);
            float analytic = p->grad[idx];
            float numeric = tg_numeric_grad_at(arena, eval_fn, ctx, p, idx, eps);
            float abs_err = fabsf(analytic - numeric);
            float rel_err = tg_rel_error(analytic, numeric);

            if (!isfinite(analytic) || !isfinite(numeric)) {
                printf("[FAIL] %s: %s[%zu] produced non-finite gradient "
                       "(analytic=%g, numeric=%g)\n",
                       test_name,
                       pr->name ? pr->name : "(unnamed)",
                       idx,
                       (double)analytic,
                       (double)numeric);
                ok = false;
                continue;
            }

            if (!tg_is_close(analytic, numeric, abs_tol, rel_tol)) {
                printf("[FAIL] %s: %s[%zu] analytic=% .7f numeric=% .7f "
                       "abs_err=% .7f rel_err=% .7f\n",
                       test_name,
                       pr->name ? pr->name : "(unnamed)",
                       idx,
                       analytic,
                       numeric,
                       abs_err,
                       rel_err);
                ok = false;
            } else {
                printf("[PASS] %s: %s[%zu] analytic=% .7f numeric=% .7f "
                       "abs_err=% .7f rel_err=% .7f\n",
                       test_name,
                       pr->name ? pr->name : "(unnamed)",
                       idx,
                       analytic,
                       numeric,
                       abs_err,
                       rel_err);
            }
        }
    }

    return ok;
}

/* ------------------------------ Test 1 ---------------------------------
   mean(mul(add(A,B), C))
   Shapes:
     A, B, C: [2 x 3]
------------------------------------------------------------------------- */

typedef struct tg_gc_case1 {
    tg_tensor *A;
    tg_tensor *B;
    tg_tensor *C;
} tg_gc_case1;

static float tg_eval_case1(tg_arena *arena, void *ctx) {
    tg_gc_case1 *p = (tg_gc_case1 *)ctx;
    tg_tensor *sum_ab;
    tg_tensor *prod;
    tg_tensor *loss;

    sum_ab = tg_add(arena, p->A, p->B);
    if (!sum_ab) {
        return NAN;
    }

    prod = tg_mul(arena, sum_ab, p->C);
    if (!prod) {
        return NAN;
    }

    loss = tg_mean(arena, prod);
    if (!loss || !loss->data) {
        return NAN;
    }

    return loss->data[0];
}

static bool tg_run_case1(void) {
    const char *test_name = "mean(mul(add(A,B),C))";
    tg_arena *arena = NULL;
    tg_gc_case1 ctx;
    tg_tensor *params_to_zero[3];
    tg_gc_param_ref refs[3];
    tg_tensor *sum_ab;
    tg_tensor *prod;
    tg_tensor *loss;
    tg_status st;
    bool ok = false;

    arena = tg_arena_create(TG_GC_ARENA_BYTES);
    if (!arena) {
        printf("[FAIL] %s: tg_arena_create failed\n", test_name);
        return false;
    }

    ctx.A = tg_param_create(2, 3, true);
    ctx.B = tg_param_create(2, 3, true);
    ctx.C = tg_param_create(2, 3, true);
    if (!ctx.A || !ctx.B || !ctx.C) {
        printf("[FAIL] %s: tensor allocation failed\n", test_name);
        goto cleanup;
    }

    tg_fill_uniform(ctx.A, -1.0f, 1.0f);
    tg_fill_uniform(ctx.B, -1.0f, 1.0f);
    tg_fill_uniform(ctx.C, -1.0f, 1.0f);

    params_to_zero[0] = ctx.A;
    params_to_zero[1] = ctx.B;
    params_to_zero[2] = ctx.C;
    tg_zero_grads(params_to_zero, 3);

    sum_ab = tg_add(arena, ctx.A, ctx.B);
    prod = tg_mul(arena, sum_ab, ctx.C);
    loss = tg_mean(arena, prod);
    if (!sum_ab || !prod || !loss) {
        printf("[FAIL] %s: forward graph construction failed\n", test_name);
        goto cleanup;
    }

    st = tg_backward(loss);
    if (st != TG_OK) {
        printf("[FAIL] %s: tg_backward returned %d\n", test_name, (int)st);
        goto cleanup;
    }

    refs[0].name = "A"; refs[0].tensor = ctx.A;
    refs[1].name = "B"; refs[1].tensor = ctx.B;
    refs[2].name = "C"; refs[2].tensor = ctx.C;

    tg_arena_reset(arena);

    ok = tg_gradcheck_params(
        test_name,
        arena,
        tg_eval_case1,
        &ctx,
        refs,
        3,
        TG_GC_SAMPLES,
        TG_GC_EPS,
        TG_GC_ABS_TOL,
        TG_GC_REL_TOL
    );

cleanup:
    tg_tensor_destroy(ctx.A);
    tg_tensor_destroy(ctx.B);
    tg_tensor_destroy(ctx.C);
    tg_arena_destroy(arena);
    return ok;
}

/* ------------------------------ Test 2 ---------------------------------
   mse(matmul(X,W), Y)
   Shapes:
     X: [2 x 3]
     W: [3 x 4]
     Y: [2 x 4]
------------------------------------------------------------------------- */

typedef struct tg_gc_case2 {
    tg_tensor *X;
    tg_tensor *W;
    tg_tensor *Y;
} tg_gc_case2;

static float tg_eval_case2(tg_arena *arena, void *ctx) {
    tg_gc_case2 *p = (tg_gc_case2 *)ctx;
    tg_tensor *pred;
    tg_tensor *loss;

    pred = tg_matmul(arena, p->X, p->W);
    if (!pred) {
        return NAN;
    }

    loss = tg_mse(arena, pred, p->Y);
    if (!loss || !loss->data) {
        return NAN;
    }

    return loss->data[0];
}

static bool tg_run_case2(void) {
    const char *test_name = "mse(matmul(X,W),Y)";
    tg_arena *arena = NULL;
    tg_gc_case2 ctx;
    tg_tensor *params_to_zero[2];
    tg_gc_param_ref refs[2];
    tg_tensor *pred;
    tg_tensor *loss;
    tg_status st;
    bool ok = false;

    arena = tg_arena_create(TG_GC_ARENA_BYTES);
    if (!arena) {
        printf("[FAIL] %s: tg_arena_create failed\n", test_name);
        return false;
    }

    ctx.X = tg_param_create(2, 3, true);
    ctx.W = tg_param_create(3, 4, true);
    ctx.Y = tg_param_create(2, 4, false);
    if (!ctx.X || !ctx.W || !ctx.Y) {
        printf("[FAIL] %s: tensor allocation failed\n", test_name);
        goto cleanup;
    }

    tg_fill_uniform(ctx.X, -0.8f, 0.8f);
    tg_fill_uniform(ctx.W, -0.8f, 0.8f);
    tg_fill_uniform(ctx.Y, -0.8f, 0.8f);

    params_to_zero[0] = ctx.X;
    params_to_zero[1] = ctx.W;
    tg_zero_grads(params_to_zero, 2);

    pred = tg_matmul(arena, ctx.X, ctx.W);
    loss = tg_mse(arena, pred, ctx.Y);
    if (!pred || !loss) {
        printf("[FAIL] %s: forward graph construction failed\n", test_name);
        goto cleanup;
    }

    st = tg_backward(loss);
    if (st != TG_OK) {
        printf("[FAIL] %s: tg_backward returned %d\n", test_name, (int)st);
        goto cleanup;
    }

    refs[0].name = "X"; refs[0].tensor = ctx.X;
    refs[1].name = "W"; refs[1].tensor = ctx.W;

    tg_arena_reset(arena);

    ok = tg_gradcheck_params(
        test_name,
        arena,
        tg_eval_case2,
        &ctx,
        refs,
        2,
        TG_GC_SAMPLES,
        TG_GC_EPS,
        TG_GC_ABS_TOL,
        TG_GC_REL_TOL
    );

cleanup:
    tg_tensor_destroy(ctx.X);
    tg_tensor_destroy(ctx.W);
    tg_tensor_destroy(ctx.Y);
    tg_arena_destroy(arena);
    return ok;
}

/* ------------------------------ Test 3 ---------------------------------
   bce_with_logits(matmul(X,W)+b, y)
   Shapes:
     X: [2 x 3]
     W: [3 x 2]
     b: [1 x 2]   (tests bias broadcast in tg_add)
     y: [2 x 2]
------------------------------------------------------------------------- */

typedef struct tg_gc_case3 {
    tg_tensor *X;
    tg_tensor *W;
    tg_tensor *b;
    tg_tensor *y;
} tg_gc_case3;

static float tg_eval_case3(tg_arena *arena, void *ctx) {
    tg_gc_case3 *p = (tg_gc_case3 *)ctx;
    tg_tensor *mm;
    tg_tensor *logits;
    tg_tensor *loss;

    mm = tg_matmul(arena, p->X, p->W);
    if (!mm) {
        return NAN;
    }

    logits = tg_add(arena, mm, p->b);
    if (!logits) {
        return NAN;
    }

    loss = tg_bce_with_logits(arena, logits, p->y);
    if (!loss || !loss->data) {
        return NAN;
    }

    return loss->data[0];
}

static bool tg_run_case3(void) {
    const char *test_name = "bce_with_logits(matmul(X,W)+b,y)";
    tg_arena *arena = NULL;
    tg_gc_case3 ctx;
    tg_tensor *params_to_zero[3];
    tg_gc_param_ref refs[3];
    tg_tensor *mm;
    tg_tensor *logits;
    tg_tensor *loss;
    tg_status st;
    bool ok = false;

    arena = tg_arena_create(TG_GC_ARENA_BYTES);
    if (!arena) {
        printf("[FAIL] %s: tg_arena_create failed\n", test_name);
        return false;
    }

    ctx.X = tg_param_create(2, 3, true);
    ctx.W = tg_param_create(3, 2, true);
    ctx.b = tg_param_create(1, 2, true);
    ctx.y = tg_param_create(2, 2, false);
    if (!ctx.X || !ctx.W || !ctx.b || !ctx.y) {
        printf("[FAIL] %s: tensor allocation failed\n", test_name);
        goto cleanup;
    }

    /*
    Для BCE лучше избегать слишком больших логитов в gradcheck,
    поэтому держим входы в умеренном диапазоне.
    */
    tg_fill_uniform(ctx.X, -0.5f, 0.5f);
    tg_fill_uniform(ctx.W, -0.5f, 0.5f);
    tg_fill_uniform(ctx.b, -0.3f, 0.3f);
    tg_fill_binary01(ctx.y);

    params_to_zero[0] = ctx.X;
    params_to_zero[1] = ctx.W;
    params_to_zero[2] = ctx.b;
    tg_zero_grads(params_to_zero, 3);

    mm = tg_matmul(arena, ctx.X, ctx.W);
    logits = tg_add(arena, mm, ctx.b);
    loss = tg_bce_with_logits(arena, logits, ctx.y);
    if (!mm || !logits || !loss) {
        printf("[FAIL] %s: forward graph construction failed\n", test_name);
        goto cleanup;
    }

    st = tg_backward(loss);
    if (st != TG_OK) {
        printf("[FAIL] %s: tg_backward returned %d\n", test_name, (int)st);
        goto cleanup;
    }

    refs[0].name = "X"; refs[0].tensor = ctx.X;
    refs[1].name = "W"; refs[1].tensor = ctx.W;
    refs[2].name = "b"; refs[2].tensor = ctx.b;

    tg_arena_reset(arena);

    ok = tg_gradcheck_params(
        test_name,
        arena,
        tg_eval_case3,
        &ctx,
        refs,
        3,
        TG_GC_SAMPLES,
        TG_GC_EPS,
        TG_GC_ABS_TOL,
        TG_GC_REL_TOL
    );

cleanup:
    tg_tensor_destroy(ctx.X);
    tg_tensor_destroy(ctx.W);
    tg_tensor_destroy(ctx.b);
    tg_tensor_destroy(ctx.y);
    tg_arena_destroy(arena);
    return ok;
}

/* -------------------------------- main -------------------------------- */

int main(void) {
    bool ok1;
    bool ok2;
    bool ok3;
    int failed = 0;

    srand((unsigned int)TG_GC_SEED);

    printf("=== tinygradc gradcheck ===\n");
    printf("seed=%u eps=%g rel_tol=%g abs_tol=%g samples=%d\n\n",
           (unsigned int)TG_GC_SEED,
           (double)TG_GC_EPS,
           (double)TG_GC_REL_TOL,
           (double)TG_GC_ABS_TOL,
           TG_GC_SAMPLES);

    ok1 = tg_run_case1();
    printf("\n");

    ok2 = tg_run_case2();
    printf("\n");

    ok3 = tg_run_case3();
    printf("\n");

    if (!ok1) {
        failed++;
    }
    if (!ok2) {
        failed++;
    }
    if (!ok3) {
        failed++;
    }

    if (failed == 0) {
        printf("GRADCHECK RESULT: PASS\n");
        return 0;
    }

    printf("GRADCHECK RESULT: FAIL (%d test group(s) failed)\n", failed);
    return 1;
}
