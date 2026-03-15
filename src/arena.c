#include "tg.h"

#include <stdint.h>
#include <stdlib.h>

#define TG_ARENA_DEFAULT_INITIAL_BYTES ((size_t)4096)

typedef struct tg_arena_chunk {
    struct tg_arena_chunk *next;
    size_t capacity;
    size_t used;
    unsigned char data[];
} tg_arena_chunk;

struct tg_arena {
    tg_arena_chunk *head;
    tg_arena_chunk *current;
    size_t initial_chunk_size;
};

static int tg_is_power_of_two(size_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

static int tg_size_add_overflow(size_t a, size_t b, size_t *out)
{
    if (a > SIZE_MAX - b) {
        return 1;
    }
    *out = a + b;
    return 0;
}

static uintptr_t tg_align_up_uintptr(uintptr_t value, size_t align)
{
    uintptr_t mask = (uintptr_t)(align - 1);

    if (value > UINTPTR_MAX - mask) {
        return 0;
    }

    return (value + mask) & ~mask;
}

static tg_arena_chunk *tg_arena_chunk_create(size_t capacity)
{
    tg_arena_chunk *chunk;

    if (capacity > SIZE_MAX - sizeof(*chunk)) {
        return NULL;
    }

    chunk = (tg_arena_chunk *)malloc(sizeof(*chunk) + capacity);
    if (chunk == NULL) {
        return NULL;
    }

    chunk->next = NULL;
    chunk->capacity = capacity;
    chunk->used = 0;
    return chunk;
}

static void *tg_arena_chunk_try_alloc(tg_arena_chunk *chunk, size_t size, size_t align)
{
    uintptr_t chunk_base;
    uintptr_t raw;
    uintptr_t aligned;
    size_t offset;

    if (chunk == NULL) {
        return NULL;
    }

    chunk_base = (uintptr_t)chunk->data;

    if ((uintptr_t)chunk->used > UINTPTR_MAX - chunk_base) {
        return NULL;
    }

    raw = chunk_base + (uintptr_t)chunk->used;
    aligned = tg_align_up_uintptr(raw, align);
    if (aligned == 0) {
        return NULL;
    }

    offset = (size_t)(aligned - chunk_base);
    if (offset > chunk->capacity) {
        return NULL;
    }

    if (size > chunk->capacity - offset) {
        return NULL;
    }

    chunk->used = offset + size;
    return (void *)aligned;
}

static size_t tg_arena_choose_chunk_capacity(const tg_arena *arena,
                                             const tg_arena_chunk *current,
                                             size_t min_capacity)
{
    size_t capacity;

    if (arena == NULL || current == NULL) {
        return 0;
    }

    capacity = current->capacity;
    if (capacity < arena->initial_chunk_size) {
        capacity = arena->initial_chunk_size;
    }

    if (capacity >= min_capacity) {
        if (capacity <= SIZE_MAX / 2) {
            capacity *= 2;
        }
    } else {
        while (capacity < min_capacity) {
            if (capacity > SIZE_MAX / 2) {
                capacity = min_capacity;
                break;
            }
            capacity *= 2;
        }
    }

    if (capacity < min_capacity) {
        return 0;
    }

    return capacity;
}

tg_arena *tg_arena_create(size_t initial_bytes)
{
    tg_arena *arena;
    tg_arena_chunk *head;

    if (initial_bytes == 0) {
        initial_bytes = TG_ARENA_DEFAULT_INITIAL_BYTES;
    }

    head = tg_arena_chunk_create(initial_bytes);
    if (head == NULL) {
        return NULL;
    }

    arena = (tg_arena *)malloc(sizeof(*arena));
    if (arena == NULL) {
        free(head);
        return NULL;
    }

    arena->head = head;
    arena->current = head;
    arena->initial_chunk_size = initial_bytes;
    return arena;
}

void tg_arena_destroy(tg_arena *arena)
{
    tg_arena_chunk *chunk;
    tg_arena_chunk *next;

    if (arena == NULL) {
        return;
    }

    chunk = arena->head;
    while (chunk != NULL) {
        next = chunk->next;
        free(chunk);
        chunk = next;
    }

    free(arena);
}

void tg_arena_reset(tg_arena *arena)
{
    if (arena == NULL || arena->head == NULL) {
        return;
    }

    arena->current = arena->head;
    arena->head->used = 0;
}

void *tg_arena_alloc(tg_arena *arena, size_t size, size_t align)
{
    tg_arena_chunk *chunk;
    void *ptr;
    size_t min_capacity;

    if (arena == NULL || arena->current == NULL) {
        return NULL;
    }

    if (size == 0) {
        size = 1;
    }

    if (align == 0) {
        align = _Alignof(max_align_t);
    }

    if (!tg_is_power_of_two(align)) {
        return NULL;
    }

    if (tg_size_add_overflow(size, align - 1, &min_capacity)) {
        return NULL;
    }

    chunk = arena->current;

    for (;;) {
        ptr = tg_arena_chunk_try_alloc(chunk, size, align);
        if (ptr != NULL) {
            arena->current = chunk;
            return ptr;
        }

        if (chunk->next != NULL) {
            chunk = chunk->next;
            chunk->used = 0;
            continue;
        }

        {
            size_t new_capacity;
            tg_arena_chunk *new_chunk;

            new_capacity = tg_arena_choose_chunk_capacity(arena, chunk, min_capacity);
            if (new_capacity == 0) {
                return NULL;
            }

            new_chunk = tg_arena_chunk_create(new_capacity);
            if (new_chunk == NULL) {
                return NULL;
            }

            chunk->next = new_chunk;
            chunk = new_chunk;
        }
    }
}
