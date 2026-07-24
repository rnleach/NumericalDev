#ifndef _PACKRAT_H_
#define _PACKRAT_H_

/*---------------------------------------------------------------------------------------------------------------------------
 *                                                     String Interner
 *---------------------------------------------------------------------------------------------------------------------------
 *
 * size_exp - The interner is backed by a hash table with a capacity that is a power of 2. The size_exp is that power of two.
 *     This value is only used initially, if the table needs to expand, it will, so it's OK to start with small values here.
 *     However, if you know it will grow larger, it's better to start larger! For most reasonable use cases, it really
 *     probably shouldn't be smaller than 5, but no checks are done for this.
 *
 * NOTE: A cstring is a null terminated string of unknown length.
 *
 * NOTE: The interner copies any successfully interned string, so once it has been interned you can reclaim the memory that 
 * was holding it before.
 */
typedef struct /* Internal only */
{
    u64 hash;
    ElkStr str;
} PakStringInternerHandle;

typedef struct 
{
    MagAllocator *storage;            /* This is where to store the strings.           */

    PakStringInternerHandle *handles; /* The hash table - handles index into storage.  */
    u32 num_handles;                  /* The number of handles.                        */
    i8 size_exp;                      /* Used to keep track of the length of *handles. */
} PakStringInterner;


static inline PakStringInterner pak_string_interner_create(i8 size_exp, MagAllocator *storage);
static inline void pak_string_interner_destroy(PakStringInterner *interner);
static inline ElkStr pak_string_interner_intern_cstring(PakStringInterner *interner, char *string);
static inline ElkStr pak_string_interner_intern(PakStringInterner *interner, ElkStr str);

/*---------------------------------------------------------------------------------------------------------------------------
 *
 *                                         
 *                                                       Collections
 *
 *
 *-------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------
 *                                         
 *                                                   Ordered Collections
 *
 *---------------------------------------------------------------------------------------------------------------------------
 *
 *  Ordered collections are things like arrays (sometimes called vectors), queues, dequeues, and heaps. I'm taking a 
 *  different approach to how I implement them. They will be implemented in two parts.
 *
 *  The first part is the ledger, which just does all the bookkeeping about how full the collection is, which elements are
 *  valid, and the order of elements. A user can use these with their own supplied buffer for storing objects. The ledger
 *  types only track indexes.
 *
 *  One advantage of the ledger approach is that the user can manage their own memory, allowing them to use the most
 *  efficient allocation strategy. 
 *
 *  Another advantage of this is that if you have several parallel collections (e.g. parallel arrays), you can use a single
 *  instance of a bookkeeping type (e.g. PakQueueLedger) to track the state of all the arrays that back it. Further,
 *  different collections in the parallel collections can have different sized objects.
 *
 *  Complicated memory management can be a disadvantage of the ledger approach. For instance, implementing growable
 *  collections can require shuffling objects around in the backing buffer during the resize operation. A queue, for 
 *  instance, is non-trivial to expand because you have to account for wrap-around (assuming it is backed by a circular
 *  buffer). So not all the ledger types APIs will directly support resizing, and even those that do will further require 
 *  the user to make sure that as they allocate more memory in the backing buffer, they also update the ledger. Finally, if
 *  you want to pass the collection as a whole to a function, you'll have to pass the ledger and buffer(s) separately or
 *  create your own composite type to package them together in.
 *
 *  The second part is an implementation that manages memory for the user. This gives the user less control over how memory
 *  is allocated / freed, but it also burdens them less with the responsibility of doing so. These types can reliably be
 *  expanded on demand. These types will internally use the ledger types.
 *
 *  The more full featured types that manage memory also require the use of void pointers for adding and retrieving data 
 *  from the collections. So they are less type safe, whereas when using just the ledger types they only deal in indexes, so
 *  you can make the backing buffer have any type you want.
 *
 *  Adjacent just means that the objects are stored adjacently in memory. All of the collection ledger types must be backed
 *  by an array or a block of contiguous memory.
 */

static size const PAK_COLLECTION_EMPTY = -1;
static size const PAK_COLLECTION_FULL = -2;

/*---------------------------------------------------------------------------------------------------------------------------
 *                                                      Queue Ledger
 *---------------------------------------------------------------------------------------------------------------------------
 *
 * Mind the PAK_COLLECTION_FULL and PAK_COLLECTION_EMPTY return values.
 */

typedef struct 
{
    size capacity;
    size length;
    size front;
    size back;
} PakQueueLedger;

static inline PakQueueLedger pak_queue_ledger_create(size capacity);
static inline b32 pak_queue_ledger_full(PakQueueLedger *queue);
static inline b32 pak_queue_ledger_empty(PakQueueLedger *queue);
static inline size pak_queue_ledger_push_back_index(PakQueueLedger *queue);  /* index of next location to put an object   */
static inline size pak_queue_ledger_pop_front_index(PakQueueLedger *queue);  /* index of next location to take object     */
static inline size pak_queue_ledger_peek_front_index(PakQueueLedger *queue); /* index of next object, but not incremented */
static inline size pak_queue_ledger_len(PakQueueLedger const *queue);

/*---------------------------------------------------------------------------------------------------------------------------
 *                                                      Array Ledger
 *---------------------------------------------------------------------------------------------------------------------------
 *
 * Mind the PAK_COLLECTION_FULL and PAK_COLLECTION_EMPTY return values.
 */
typedef struct 
{
    size capacity;
    size length;
} PakArrayLedger;

static inline PakArrayLedger pak_array_ledger_create(size capacity);
static inline b32 pak_array_ledger_full(PakArrayLedger *array);
static inline b32 pak_array_ledger_empty(PakArrayLedger *array);
static inline size pak_array_ledger_push_back_index(PakArrayLedger *array);
static inline size pak_array_ledger_pop_back_index(PakArrayLedger *array);
static inline size pak_array_ledger_len(PakArrayLedger const *array);
static inline void pak_array_ledger_reset(PakArrayLedger *array);
static inline void pak_array_ledger_set_capacity(PakArrayLedger *array, size capacity);

/*---------------------------------------------------------------------------------------------------------------------------
 *                                                      Stack Ledger
 *---------------------------------------------------------------------------------------------------------------------------
 *
 * Mind the PAK_COLLECTION_FULL and PAK_COLLECTION_EMPTY return values.
 */
typedef struct 
{
    size capacity;
    size length;
} PakStackLedger;

static inline PakStackLedger pak_stack_ledger_create(size capacity);
static inline b32 pak_stack_ledger_full(PakStackLedger *stack);
static inline b32 pak_stack_ledger_empty(PakStackLedger *stack);
static inline size pak_stack_ledger_push_index(PakStackLedger *stack);
static inline size pak_stack_ledger_pop_index(PakStackLedger *stack);
static inline size pak_stack_ledger_len(PakStackLedger const *stack);
static inline void pak_stack_ledger_reset(PakStackLedger *stack);
static inline void pak_stack_ledger_set_capacity(PakStackLedger *stack, size capacity);

/*---------------------------------------------------------------------------------------------------------------------------
 *                                         
 *                                                  Unordered Collections
 *
 *-------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------
 *                                                    Hash Map (Table)
 *---------------------------------------------------------------------------------------------------------------------------
 * The table size must be a power of two, so size_exp is used to calcualte the size of the table. If it needs to grow in 
 * size, it will grow the table, so this is only a starting point.
 *
 * The ELkHashMap does NOT copy any objects, so it only stores pointers. The user has to manage the memory for their own
 * objects.
 */
typedef struct /* Internal Only */
{
    u64 hash;
    void *key;
    void *value;
} PakHashMapHandle;

typedef struct 
{
    MagAllocator *alloc;
    PakHashMapHandle *handles;
    size num_handles;
    ElkSimpleHash hasher;
    ElkEqFunction eq;
    i8 size_exp;
} PakHashMap;

typedef size PakHashMapKeyIter;

static inline PakHashMap pak_hash_map_create(i8 size_exp, ElkSimpleHash key_hash, ElkEqFunction key_eq, MagAllocator *alloc);
static inline void pak_hash_map_destroy(PakHashMap *map);
static inline void *pak_hash_map_insert(PakHashMap *map, void *key, void *value); /* if return != value, key was already in the map  */
static inline void *pak_hash_map_lookup(PakHashMap *map, void *key); /* return NULL if not in map, otherwise return pointer to value */
static inline size pak_hash_map_len(PakHashMap *map);
static inline PakHashMapKeyIter pak_hash_map_key_iter(PakHashMap *map);

static inline void *pak_hash_map_key_iter_next(PakHashMap *map, PakHashMapKeyIter *iter);

/*--------------------------------------------------------------------------------------------------------------------------
 *                                            Hash Map (Table, ElkStr as keys)
 *---------------------------------------------------------------------------------------------------------------------------
 * Designed for use when keys are strings.
 *
 * Values are not copied, they are stored as pointers, so the user must manage memory.
 *
 * Uses fnv1a hash.
 */
typedef struct
{
    u64 hash;
    ElkStr key;
    void *value;
} PakStrMapHandle;

typedef struct 
{
    MagAllocator *alloc;
    PakStrMapHandle *handles;
    size num_handles;
    i8 size_exp;
} PakStrMap;

typedef size PakStrMapKeyIter;
typedef size PakStrMapHandleIter;

static inline PakStrMap pak_str_map_create(i8 size_exp, MagAllocator *alloc);
static inline void pak_str_map_destroy(PakStrMap *map);
static inline void *pak_str_map_insert(PakStrMap *map, ElkStr key, void *value); /* if return != value, key was already in the map                           */
static inline void *pak_str_map_lookup(PakStrMap *map, ElkStr key); /* return NULL if not in map, otherwise return pointer to value                          */
static inline PakStrMapHandle const *pak_str_map_lookup_handle(PakStrMap *map, ElkStr key); /* return NULL if not in map, otherwise return pointer to handle */
static inline size pak_str_map_len(PakStrMap *map);
static inline PakStrMapKeyIter pak_str_map_key_iter(PakStrMap *map);
static inline PakStrMapHandleIter pak_str_map_handle_iter(PakStrMap *map);

static inline ElkStr pak_str_map_key_iter_next(PakStrMap *map, PakStrMapKeyIter *iter);
static inline PakStrMapHandle pak_str_map_handle_iter_next(PakStrMap *map, PakStrMapHandleIter *iter);

/*---------------------------------------------------------------------------------------------------------------------------
 *                                                        Hash Set
 *---------------------------------------------------------------------------------------------------------------------------
 * The table size must be a power of two, so size_exp is used to calcualte the size of the table. If it needs to grow in 
 * size, it will grow the table, so this is only a starting point.
 *
 * The ELkHashSet does NOT copy any objects, so it only stores pointers. The user has to manage the memory for their own
 * objects.
 */
typedef struct /* Internal only */
{
    u64 hash;
    void *value;
} PakHashSetHandle;

typedef struct 
{
    MagAllocator *alloc;
    PakHashSetHandle *handles;
    size num_handles;
    ElkSimpleHash hasher;
    ElkEqFunction eq;
    i8 size_exp;
} PakHashSet;

typedef size PakHashSetIter;

static inline PakHashSet pak_hash_set_create(i8 size_exp, ElkSimpleHash val_hash, ElkEqFunction val_eq, MagAllocator *alloc);
static inline void pak_hash_set_destroy(PakHashSet *set);
static inline void *pak_hash_set_insert(PakHashSet *set, void *value); /* if return != value, value was already in the set    */
static inline void *pak_hash_set_lookup(PakHashSet *set, void *value); /* return NULL if not in set, else return ptr to value */
static inline size pak_hash_set_len(PakHashSet *set);
static inline PakHashSetIter pak_hash_set_value_iter(PakHashSet *set);

static inline void *pak_hash_set_value_iter_next(PakHashSet *set, PakHashSetIter *iter);

/*---------------------------------------------------------------------------------------------------------------------------
 *                                            Generic Macros for Collections
 *---------------------------------------------------------------------------------------------------------------------------
 * These macros take any collection and return a result.
 */
#define pak_len(x) _Generic((x),                                                                                            \
        PakQueueLedger *: pak_queue_ledger_len,                                                                             \
        PakArrayLedger *: pak_array_ledger_len,                                                                             \
        PakHashMap *: pak_hash_map_len,                                                                                     \
        PakStrMap *: pak_str_map_len,                                                                                       \
        PakHashSet *: pak_hash_set_len)(x)

/*---------------------------------------------------------------------------------------------------------------------------
 *
 *                                         
 *                                                        Sorting
 *
 *
 *-------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------
 *                                                       Radix Sort 
 *---------------------------------------------------------------------------------------------------------------------------
 * I've only implemented radix sort for fixed size signed intengers, unsigned integers, and floating point types thus far.
 *
 * The sort requires extra memory that depends on the size of the list being sorted, so the user must provide that. In order
 * to ensure proper alignment and size, the user must provide the scratch buffer. Probably using an arena with 
 * eco_arena_nmalloc() to create a buffer and then freeing it after the sort would be the most efficient way to handle that.
 *
 * The API is set up for sorting an array of structures. The 'offset' argument is the offset in bytes into the structure 
 * where the sort key can be found. The 'stride' argument is just the size of the structures so we know how to iterate the
 * array. The sort order (ascending / descending) and sort key type (signed integer, unsigned integer, floating) are passed
 * as arguments. The sort key type includes the size of the sort key in bits.
 */

typedef enum
{
    PAK_RADIX_SORT_UINT8,
    PAK_RADIX_SORT_INT8,
    PAK_RADIX_SORT_UINT16,
    PAK_RADIX_SORT_INT16,
    PAK_RADIX_SORT_UINT32,
    PAK_RADIX_SORT_INT32,
    PAK_RADIX_SORT_F32,
    PAK_RADIX_SORT_UINT64,
    PAK_RADIX_SORT_INT64,
    PAK_RADIX_SORT_F64,
} PakRadixSortByType;

typedef enum { PAK_SORT_ASCENDING, PAK_SORT_DESCENDING } PakSortOrder;

static inline void pak_radix_sort(
        void *buffer, 
        size num, 
        size offset, 
        size stride, 
        void *scratch, 
        PakRadixSortByType sort_type, 
        PakSortOrder order);

/*---------------------------------------------------------------------------------------------------------------------------
 *
 *
 *
 *                                          Implementation of `inline` functions.
 *
 *
 *
 *-------------------------------------------------------------------------------------------------------------------------*/
static inline PakStringInterner 
pak_string_interner_create(i8 size_exp, MagAllocator *storage)
{
    Assert(size_exp > 0 && size_exp <= 31); /* Come on, 2^31 is HUGE */

    usize const handles_len = (usize)(1 << size_exp);
    PakStringInternerHandle *handles = eco_arena_nmalloc(storage, handles_len, PakStringInternerHandle);
    PanicIf(!handles);

    return (PakStringInterner)
    {
        .storage = storage,
        .handles = handles,
        .num_handles = 0, 
        .size_exp = size_exp
    };
}

static inline void
pak_string_interner_destroy(PakStringInterner *interner)
{
    return;
}

static inline b32
pak_hash_table_large_enough(usize num_handles, i8 size_exp)
{
    /* Shoot for no more than 75% of slots filled. */
    return num_handles < 3 * (1 << size_exp) / 4;
}

static inline u32
pak_hash_lookup(u64 hash, i8 exp, u32 idx)
{
    /* Copied from https://nullprogram.com/blog/2022/08/08
     * All code & writing on this blog is in the public domain.
     */
    u32 mask = (UINT32_C(1) << exp) - 1;
    u32 step = (u32)((hash >> (64 - exp)) | 1);    /* the | 1 forces an odd number */
    return (idx + step) & mask;
}

static inline void
pak_string_interner_expand_table(PakStringInterner *interner)
{
    i8 const size_exp = interner->size_exp;
    i8 const new_size_exp = size_exp + 1;

    usize const handles_len = (usize)(1 << size_exp);
    usize const new_handles_len = (usize)(1 << new_size_exp);

    PakStringInternerHandle *new_handles = eco_arena_nmalloc(interner->storage, new_handles_len, PakStringInternerHandle);
    PanicIf(!new_handles);

    for (u32 i = 0; i < handles_len; i++) 
    {
        PakStringInternerHandle *handle = &interner->handles[i];

        if (handle->str.start == NULL) { continue; } /* Skip if it's empty */

        /* Find the position in the new table and update it. */
        u64 const hash = handle->hash;
        u32 j = hash & 0xffffffff; /* truncate */
        while (true) 
        {
            j = pak_hash_lookup(hash, new_size_exp, j);
            PakStringInternerHandle *new_handle = &new_handles[j];

            if (!new_handle->str.start)
            {
                /* empty - put it here. Don't need to check for room because we just expanded
                 * the hash table of handles, and we're not copying anything new into storage,
                 * it's already there!
                 */
                *new_handle = *handle;
                break;
            }
        }
    }

    interner->handles = new_handles;
    interner->size_exp = new_size_exp;

    return;
}

static inline ElkStr
pak_string_interner_intern_cstring(PakStringInterner *interner, char *string)
{
    ElkStr str = elk_str_from_cstring(string);
    return pak_string_interner_intern(interner, str);
}

static inline ElkStr
pak_string_interner_intern(PakStringInterner *interner, ElkStr str)
{
    /* Inspired by https://nullprogram.com/blog/2022/08/08
     * All code & writing on this blog is in the public domain.
     */

    u64 const hash = elk_fnv1a_hash_str(str);
    u32 i = hash & 0xffffffff; /* truncate */
    while (true)
    {
        i = pak_hash_lookup(hash, interner->size_exp, i);
        PakStringInternerHandle *handle = &interner->handles[i];

        if (!handle->str.start)
        {
            /* empty, insert here if room in the table of handles. Check for room first! */
            if (pak_hash_table_large_enough(interner->num_handles, interner->size_exp))
            {
                char *dest = eco_arena_nmalloc(interner->storage, str.len + 1, char);
                PanicIf(!dest);
                ElkStr interned_str = elk_str_copy(str.len + 1, dest, str);

                *handle = (PakStringInternerHandle){.hash = hash, .str = interned_str};
                interner->num_handles += 1;

                return handle->str;
            }
            else 
            {
                /* Grow the table so we have room */
                pak_string_interner_expand_table(interner);

                /* Recurse because all the state needed by the *_lookup function was just crushed
                 * by the expansion of the table.
                 */
                return pak_string_interner_intern(interner, str);
            }
        }
        else if (handle->hash == hash && elk_str_eq(str, handle->str)) 
        {
            /* found it! */
            return handle->str;
        }
    }
}

static inline PakQueueLedger
pak_queue_ledger_create(size capacity)
{
    return (PakQueueLedger)
    {
        .capacity = capacity, 
        .length = 0,
        .front = 0, 
        .back = 0
    };
}

static inline b32 
pak_queue_ledger_full(PakQueueLedger *queue)
{ 
    return queue->length == queue->capacity;
}

static inline b32
pak_queue_ledger_empty(PakQueueLedger *queue)
{ 
    return queue->length == 0;
}

static inline size
pak_queue_ledger_push_back_index(PakQueueLedger *queue)
{
    if(pak_queue_ledger_full(queue)) { return PAK_COLLECTION_FULL; }

    size idx = queue->back % queue->capacity;
    queue->back += 1;
    queue->length += 1;
    return idx;
}

static inline size
pak_queue_ledger_pop_front_index(PakQueueLedger *queue)
{
    if(pak_queue_ledger_empty(queue)) { return PAK_COLLECTION_EMPTY; }

    size idx = queue->front % queue->capacity;
    queue->front += 1;
    queue->length -= 1;
    return idx;
}

static inline size
pak_queue_ledger_peek_front_index(PakQueueLedger *queue)
{
    if(queue->length == 0) { return PAK_COLLECTION_EMPTY; }
    return queue->front % queue->capacity;
}

static inline size
pak_queue_ledger_len(PakQueueLedger const *queue)
{
    return queue->length;
}

static inline PakArrayLedger
pak_array_ledger_create(size capacity)
{
    Assert(capacity > 0);
    return (PakArrayLedger)
    {
        .capacity = capacity,
        .length = 0
    };
}

static inline b32 
pak_array_ledger_full(PakArrayLedger *array)
{ 
    return array->length == array->capacity;
}

static inline b32
pak_array_ledger_empty(PakArrayLedger *array)
{ 
    return array->length == 0;
}

static inline size
pak_array_ledger_push_back_index(PakArrayLedger *array)
{
    if(pak_array_ledger_full(array)) { return PAK_COLLECTION_FULL; }

    size idx = array->length;
    array->length += 1;
    return idx;
}

static inline size
pak_array_ledger_pop_back_index(PakArrayLedger *array)
{
    if(pak_array_ledger_empty(array)) { return PAK_COLLECTION_EMPTY; }

    return --array->length;
}

static inline size
pak_array_ledger_len(PakArrayLedger const *array)
{
    return array->length;
}

static inline void
pak_array_ledger_reset(PakArrayLedger *array)
{
    array->length = 0;
}

static inline void
pak_array_ledger_set_capacity(PakArrayLedger *array, size capacity)
{
    Assert(capacity > 0);
    array->capacity = capacity;
}

static inline PakStackLedger 
pak_stack_ledger_create(size capacity)
{
    Assert(capacity > 0);
    return (PakStackLedger)
    {
        .capacity = capacity,
        .length = 0
    };
}

static inline b32 
pak_stack_ledger_full(PakStackLedger *stack)
{
    return stack->length == stack->capacity;
}

static inline b32 
pak_stack_ledger_empty(PakStackLedger *stack)
{
    return stack->length == 0;
}

static inline size 
pak_stack_ledger_push_index(PakStackLedger *stack)
{
    if(pak_stack_ledger_full(stack)) { return PAK_COLLECTION_FULL; }

    size idx = stack->length;
    stack->length += 1;
    return idx;
}

static inline size 
pak_stack_ledger_pop_index(PakStackLedger *stack)
{
    if(pak_stack_ledger_empty(stack)) { return PAK_COLLECTION_EMPTY; }

    return --stack->length;
}

static inline size 
pak_stack_ledger_len(PakStackLedger const *stack)
{
    return stack->length;
}

static inline void 
pak_stack_ledger_reset(PakStackLedger *stack)
{
    stack->length = 0;
}

static inline void 
pak_stack_ledger_set_capacity(PakStackLedger *stack, size capacity)
{
    Assert(capacity > 0);
    stack->capacity = capacity;
}

static PakHashMap 
pak_hash_map_create(i8 size_exp, ElkSimpleHash key_hash, ElkEqFunction key_eq, MagAllocator *alloc)
{
    Assert(size_exp > 0 && size_exp <= 31); /* Come on, 31 is HUGE */

    size const handles_len = (size)(1 << size_exp);
    PakHashMapHandle *handles = eco_arena_nmalloc(alloc, handles_len, PakHashMapHandle);
    PanicIf(!handles);

    return (PakHashMap)
    {
        .alloc = alloc,
        .handles = handles,
        .num_handles = 0, 
        .hasher = key_hash,
        .eq = key_eq,
        .size_exp = size_exp
    };
}

static inline void 
pak_hash_map_destroy(PakHashMap *map)
{
    return;
}

static inline void
pak_hash_table_expand(PakHashMap *map)
{
    i8 const size_exp = map->size_exp;
    i8 const new_size_exp = size_exp + 1;

    size const handles_len = (size)(1 << size_exp);
    size const new_handles_len = (size)(1 << new_size_exp);

    PakHashMapHandle *new_handles = eco_arena_nmalloc(map->alloc, new_handles_len, PakHashMapHandle);
    PanicIf(!new_handles);

    for (u32 i = 0; i < handles_len; i++) 
    {
        PakHashMapHandle *handle = &map->handles[i];

        if (handle->key == NULL) { continue; } /* Skip if it's empty */

        /* Find the position in the new table and update it. */
        u64 const hash = handle->hash;
        u32 j = hash & 0xffffffff; /* truncate */
        while (true) 
        {
            j = pak_hash_lookup(hash, new_size_exp, j);
            PakHashMapHandle *new_handle = &new_handles[j];

            if (!new_handle->key)
            {
                /* empty - put it here. Don't need to check for room because we just expanded
                /  the hash table of handles, and we're not copying anything new into storage,
                 * it's already there!
                 */
                *new_handle = *handle;
                break;
            }
        }
    }

    map->handles = new_handles;
    map->size_exp = new_size_exp;

    return;
}

static inline void *
pak_hash_map_insert(PakHashMap *map, void *key, void *value)
{
    /* Inspired by https://nullprogram.com/blog/2022/08/08
     * All code & writing on this blog is in the public domain. 
     */

    u64 const hash = map->hasher(key);
    u32 i = hash & 0xffffffff; /* truncate */
    while (true)
    {
        i = pak_hash_lookup(hash, map->size_exp, i);
        PakHashMapHandle *handle = &map->handles[i];

        if (!handle->key)
        {
            /* empty, insert here if room in the table of handles. Check for room first! */
            if (pak_hash_table_large_enough(map->num_handles, map->size_exp))
            {

                *handle = (PakHashMapHandle){.hash = hash, .key=key, .value=value};
                map->num_handles += 1;

                return handle->value;
            }
            else 
            {
                /* Grow the table so we have room */
                pak_hash_table_expand(map);

                /* Recurse because all the state needed by the *_lookup function was just crushed
                 * by the expansion of the table.
                 */
                return pak_hash_map_insert(map, key, value);
            }
        }
        else if (handle->hash == hash && map->eq(handle->key, key)) 
        {
            /* found it! */
            return handle->value;
        }
    }

    return NULL;
}

static inline void *
pak_hash_map_lookup(PakHashMap *map, void *key)
{
    /* Inspired by https://nullprogram.com/blog/2022/08/08
     * All code & writing on this blog is in the public domain.
     */

    u64 const hash = map->hasher(key);
    u32 i = hash & 0xffffffff; /* truncate */
    while (true)
    {
        i = pak_hash_lookup(hash, map->size_exp, i);
        PakHashMapHandle *handle = &map->handles[i];

        if (!handle->key)
        {
            return NULL;
            
        }
        else if (handle->hash == hash && map->eq(handle->key, key)) 
        {
            /* found it! */
            return handle->value;
        }
    }

    return NULL;
}

static inline size 
pak_hash_map_len(PakHashMap *map)
{
    return map->num_handles;
}

static inline PakHashMapKeyIter 
pak_hash_map_key_iter(PakHashMap *map)
{
    return 0;
}

static inline void *
pak_hash_map_key_iter_next(PakHashMap *map, PakHashMapKeyIter *iter)
{
    size const max_iter = (size)(1 << map->size_exp);
    void *next_key = NULL;
    if(*iter >= max_iter) { return next_key; }

    do
    {
        next_key = map->handles[*iter].key;
        *iter += 1;

    } while(next_key == NULL && *iter < max_iter);

    return next_key;
}

static inline PakStrMap 
pak_str_map_create(i8 size_exp, MagAllocator *alloc)
{
    Assert(size_exp > 0 && size_exp <= 31); /* Come on, 31 is HUGE */

    size const handles_len = (size)(1 << size_exp);
    PakStrMapHandle *handles = eco_arena_nmalloc(alloc, handles_len, PakStrMapHandle);
    PanicIf(!handles);

    return (PakStrMap)
    {
        .alloc = alloc,
        .handles = handles,
        .num_handles = 0, 
        .size_exp = size_exp,
    };
}

static inline void
pak_str_map_destroy(PakStrMap *map)
{
    return;
}

static inline void
pak_str_table_expand(PakStrMap *map)
{
    i8 const size_exp = map->size_exp;
    i8 const new_size_exp = size_exp + 1;

    size const handles_len = (size)(1 << size_exp);
    size const new_handles_len = (size)(1 << new_size_exp);

    PakStrMapHandle *new_handles = eco_arena_nmalloc(map->alloc, new_handles_len, PakStrMapHandle);
    PanicIf(!new_handles);

    for (u32 i = 0; i < handles_len; i++) 
    {
        PakStrMapHandle *handle = &map->handles[i];

        if (handle->key.start == NULL) { continue; } /* Skip if it's empty */

        /* Find the position in the new table and update it. */
        u64 const hash = handle->hash;
        u32 j = hash & 0xffffffff; /* truncate */
        while (true) 
        {
            j = pak_hash_lookup(hash, new_size_exp, j);
            PakStrMapHandle *new_handle = &new_handles[j];

            if (!new_handle->key.start)
            {
                /* empty - put it here. Don't need to check for room because we just expanded
                 * the hash table of handles, and we're not copying anything new into storage,
                 * it's already there!
                 */
                *new_handle = *handle;
                break;
            }
        }
    }

    map->handles = new_handles;
    map->size_exp = new_size_exp;

    return;
}

static inline void *
pak_str_map_insert(PakStrMap *map, ElkStr key, void *value)
{
    /* Inspired by https://nullprogram.com/blog/2022/08/08
     * All code & writing on this blog is in the public domain.
     */

    u64 const hash = elk_fnv1a_hash_str(key);
    u32 i = hash & 0xffffffff; /* truncate */
    while (true)
    {
        i = pak_hash_lookup(hash, map->size_exp, i);
        PakStrMapHandle *handle = &map->handles[i];

        if (!handle->key.start)
        {
            /* empty, insert here if room in the table of handles. Check for room first! */
            if (pak_hash_table_large_enough(map->num_handles, map->size_exp))
            {

                *handle = (PakStrMapHandle){.hash = hash, .key=key, .value=value};
                map->num_handles += 1;

                return handle->value;
            }
            else 
            {
                /* Grow the table so we have room */
                pak_str_table_expand(map);

                /* Recurse because all the state needed by the *_lookup function was just crushed
                 * by the expansion of the table.
                 */
                return pak_str_map_insert(map, key, value);
            }
        }
        else if (handle->hash == hash && elk_str_eq(key, handle->key)) 
        {
            /* found it! Replace value */
            void *tmp = handle->value;
            handle->value = value;

            return tmp;
        }
    }
}

static inline void *
pak_str_map_lookup(PakStrMap *map, ElkStr key)
{
    /* Inspired by https://nullprogram.com/blog/2022/08/08
     * All code & writing on this blog is in the public domain.
     */

    u64 const hash = elk_fnv1a_hash_str(key);
    u32 i = hash & 0xffffffff; /* truncate */
    while (true)
    {
        i = pak_hash_lookup(hash, map->size_exp, i);
        PakStrMapHandle *handle = &map->handles[i];

        if (!handle->key.start) { return NULL; }
        else if (handle->hash == hash && elk_str_eq(key, handle->key)) 
        {
            /* found it! */
            return handle->value;
        }
    }

    return NULL;
}

static inline PakStrMapHandle const *
pak_str_map_lookup_handle(PakStrMap *map, ElkStr key)
{
    /* Inspired by https://nullprogram.com/blog/2022/08/08
     * All code & writing on this blog is in the public domain.
     */

    u64 const hash = elk_fnv1a_hash_str(key);
    u32 i = hash & 0xffffffff; /* truncate */
    while (true)
    {
        i = pak_hash_lookup(hash, map->size_exp, i);
        PakStrMapHandle *handle = &map->handles[i];

        if (!handle->key.start) { return NULL; }
        else if (handle->hash == hash && elk_str_eq(key, handle->key)) 
        {
            /* found it! */
            return handle;
        }
    }

    return NULL;
}

static inline size
pak_str_map_len(PakStrMap *map)
{
    return map->num_handles;
}

static inline PakHashMapKeyIter 
pak_str_map_key_iter(PakStrMap *map)
{
    return 0;
}

static inline PakStrMapHandleIter 
pak_str_map_handle_iter(PakStrMap *map)
{
    return 0;
}

static inline ElkStr 
pak_str_map_key_iter_next(PakStrMap *map, PakStrMapKeyIter *iter)
{
    size const max_iter = (size)(1 << map->size_exp);
    if(*iter < max_iter) 
    {
        ElkStr next_key = map->handles[*iter].key;
        *iter += 1;
        while(next_key.start == NULL && *iter < max_iter)
        {
            next_key = map->handles[*iter].key;
            *iter += 1;
        }

        return next_key;
    }
    else
    {
        return (ElkStr){.start=NULL, .len=0};
    }

}

static inline PakStrMapHandle 
pak_str_map_handle_iter_next(PakStrMap *map, PakStrMapHandleIter *iter)
{
    size const max_iter = (size)(1 << map->size_exp);
    if(*iter < max_iter) 
    {
        PakStrMapHandle next = map->handles[*iter];
        *iter += 1;
        while(next.key.start == NULL && *iter < max_iter)
        {
            next = map->handles[*iter];
            *iter += 1;
        }

        return next;
    }
    else
    {
        return (PakStrMapHandle){ .key = (ElkStr){.start=NULL, .len=0}, .hash = 0, .value = NULL };
    }
}

static inline PakHashSet 
pak_hash_set_create(i8 size_exp, ElkSimpleHash val_hash, ElkEqFunction val_eq, MagAllocator *alloc)
{
    Assert(size_exp > 0 && size_exp <= 31); /* Come on, 31 is HUGE */

    size const handles_len = (size)(1 << size_exp);
    PakHashSetHandle *handles = eco_arena_nmalloc(alloc, handles_len, PakHashSetHandle);
    PanicIf(!handles);

    return (PakHashSet)
    {
        .alloc = alloc,
        .handles = handles,
        .num_handles = 0, 
        .hasher = val_hash,
        .eq = val_eq,
        .size_exp = size_exp
    };
}

static inline void 
pak_hash_set_destroy(PakHashSet *set)
{
    return;
}

static void
pak_hash_set_expand(PakHashSet *set)
{
    i8 const size_exp = set->size_exp;
    i8 const new_size_exp = size_exp + 1;

    size const handles_len = (size)(1 << size_exp);
    size const new_handles_len = (size)(1 << new_size_exp);

    PakHashSetHandle *new_handles = eco_arena_nmalloc(set->alloc, new_handles_len, PakHashSetHandle);
    PanicIf(!new_handles);

    for (u32 i = 0; i < handles_len; i++) 
    {
        PakHashSetHandle *handle = &set->handles[i];

        if (handle->value == NULL) { continue; } /* Skip if it's empty */

        /* Find the position in the new table and update it. */
        u64 const hash = handle->hash;
        u32 j = hash & 0xffffffff; /* truncate */
        while (true) 
        {
            j = pak_hash_lookup(hash, new_size_exp, j);
            PakHashSetHandle *new_handle = &new_handles[j];

            if (!new_handle->value)
            {
                /* empty - put it here. Don't need to check for room because we just expanded
                 * the hash table of handles, and we're not copying anything new into storage,
                 * it's already there!
                 */
                *new_handle = *handle;
                break;
            }
        }
    }

    set->handles = new_handles;
    set->size_exp = new_size_exp;

    return;
}

static inline void *
pak_hash_set_insert(PakHashSet *set, void *value)
{
    /* Inspired by https://nullprogram.com/blog/2022/08/08
     * All code & writing on this blog is in the public domain.
     */

    u64 const hash = set->hasher(value);
    u32 i = hash & 0xffffffff; /* truncate */
    while (true)
    {
        i = pak_hash_lookup(hash, set->size_exp, i);
        PakHashSetHandle *handle = &set->handles[i];

        if (!handle->value)
        {
            /* empty, insert here if room in the table of handles. Check for room first! */
            if (pak_hash_table_large_enough(set->num_handles, set->size_exp))
            {

                *handle = (PakHashSetHandle){.hash = hash, .value=value};
                set->num_handles += 1;

                return handle->value;
            }
            else 
            {
                /* Grow the table so we have room */
                pak_hash_set_expand(set);

                /* Recurse because all the state needed by the *_lookup function was just crushed
                 * by the expansion of the set. 
                 */
                return pak_hash_set_insert(set, value);
            }
        }
        else if (handle->hash == hash && set->eq(handle->value, value)) 
        {
            /* found it! */
            return handle->value;
        }
    }

    return NULL;
}

static inline void *
pak_hash_set_lookup(PakHashSet *set, void *value)
{
    /* Inspired by https://nullprogram.com/blog/2022/08/08
     * All code & writing on this blog is in the public domain.
     */

    u64 const hash = set->hasher(value);
    u32 i = hash & 0xffffffff; /* truncate */
    while (true)
    {
        i = pak_hash_lookup(hash, set->size_exp, i);
        PakHashSetHandle *handle = &set->handles[i];

        if (!handle->value)
        {
            return NULL;
        }
        else if (handle->hash == hash && set->eq(handle->value, value)) 
        {
            /* found it! */
            return handle->value;
        }
    }

    return NULL;
}

static inline size 
pak_hash_set_len(PakHashSet *set)
{
    return set->num_handles;
}

static inline PakHashSetIter
pak_hash_set_value_iter(PakHashSet *set)
{
    return 0;
}

static inline void *
pak_hash_set_value_iter_next(PakHashSet *set, PakHashSetIter *iter)
{
    size const max_iter = (size)(1 << set->size_exp);
    void *next_value = NULL;
    if(*iter >= max_iter) { return next_value; }

    do
    {
        next_value = set->handles[*iter].value;
        *iter += 1;

    } while(next_value == NULL && *iter < max_iter);

    return next_value;
}

#define PAK_I8_FLIP(x) ((x) ^ UINT8_C(0x80))
#define PAK_I8_FLIP_BACK(x) PAK_I8_FLIP(x)

#define PAK_I16_FLIP(x) ((x) ^ UINT16_C(0x8000))
#define PAK_I16_FLIP_BACK(x) PAK_I16_FLIP(x)

#define PAK_I32_FLIP(x) ((x) ^ UINT32_C(0x80000000))
#define PAK_I32_FLIP_BACK(x) PAK_I32_FLIP(x)

#define PAK_I64_FLIP(x) ((x) ^ UINT64_C(0x8000000000000000))
#define PAK_I64_FLIP_BACK(x) PAK_I64_FLIP(x)

#define PAK_F32_FLIP(x) ((x) ^ (-(i32)(((u32)(x)) >> 31) | UINT32_C(0x80000000)))
#define PAK_F32_FLIP_BACK(x) ((x) ^ (((((u32)(x)) >> 31) - 1) | UINT32_C(0x80000000)))

#define PAK_F64_FLIP(x) ((x) ^ (-(i64)(((u64)(x)) >> 63) | UINT64_C(0x8000000000000000)))
#define PAK_F64_FLIP_BACK(x) ((x) ^ (((((u64)(x)) >> 63) - 1) | UINT64_C(0x8000000000000000)))

static inline void
pak_radix_pre_sort_transform(void *buffer, size num, size offset, size stride, PakRadixSortByType sort_type)
{
    Assert(
           sort_type == PAK_RADIX_SORT_UINT64 || sort_type == PAK_RADIX_SORT_INT64 || sort_type == PAK_RADIX_SORT_F64
        || sort_type == PAK_RADIX_SORT_UINT32 || sort_type == PAK_RADIX_SORT_INT32 || sort_type == PAK_RADIX_SORT_F32
        || sort_type == PAK_RADIX_SORT_UINT16 || sort_type == PAK_RADIX_SORT_INT16
        || sort_type == PAK_RADIX_SORT_UINT8  || sort_type == PAK_RADIX_SORT_INT8
    );

    for(size i = 0; i < num; ++i)
    {
        switch(sort_type)
        {
            case PAK_RADIX_SORT_F64:
            {
                /* Flip bits so doubles sort correctly. */
                u64 *v = (u64 *)((byte *)buffer + offset + i * stride);
                *v = PAK_F64_FLIP(*v);
            } break;

            case PAK_RADIX_SORT_INT64:
            {
                /* Flip bits so two's complement signed integers sort correctly. */
                u64 *v = (u64 *)((byte *)buffer + offset + i * stride);
                *v = PAK_I64_FLIP(*v);
            } break;

            case PAK_RADIX_SORT_F32:
            {
                /* Flip bits so doubles sort correctly. */
                u32 *v = (u32 *)((byte *)buffer + offset + i * stride);
                *v = PAK_F32_FLIP(*v);
            } break;

            case PAK_RADIX_SORT_INT32:
            {
                /* Flip bits so two's complement signed integers sort correctly. */
                u32 *v = (u32 *)((byte *)buffer + offset + i * stride);
                *v = PAK_I32_FLIP(*v);
            } break;

            case PAK_RADIX_SORT_INT16:
            {
                /* Flip bits so two's complement signed integers sort correctly. */
                u16 *v = (u16 *)((byte *)buffer + offset + i * stride);
                *v = PAK_I16_FLIP(*v);
            } break;

            case PAK_RADIX_SORT_INT8:
            {
                /* Flip bits so two's complement signed integers sort correctly. */
                u8 *v = (u8 *)((byte *)buffer + offset + i * stride);
                *v = PAK_I8_FLIP(*v);
            } break;

            /* Fall through without doing anything. */
            case PAK_RADIX_SORT_UINT64:
            case PAK_RADIX_SORT_UINT32:
            case PAK_RADIX_SORT_UINT16:
            case PAK_RADIX_SORT_UINT8: {}
        }
    }
}

static inline void
pak_radix_post_sort_transform(void *buffer, size num, size offset, size stride, PakRadixSortByType sort_type)
{
    Assert(
           sort_type == PAK_RADIX_SORT_UINT64 || sort_type == PAK_RADIX_SORT_INT64 || sort_type == PAK_RADIX_SORT_F64
        || sort_type == PAK_RADIX_SORT_UINT32 || sort_type == PAK_RADIX_SORT_INT32 || sort_type == PAK_RADIX_SORT_F32
        || sort_type == PAK_RADIX_SORT_UINT16 || sort_type == PAK_RADIX_SORT_INT16
        || sort_type == PAK_RADIX_SORT_UINT8  || sort_type == PAK_RADIX_SORT_INT8
    );

    for(size i = 0; i < num; ++i)
    {
        switch(sort_type)
        {
            case PAK_RADIX_SORT_F64:
            {
                /* Flip bits so doubles sort correctly. */
                u64 *v = (u64 *)((byte *)buffer + offset + i * stride);
                *v = PAK_F64_FLIP_BACK(*v);
            } break;

            case PAK_RADIX_SORT_INT64:
            {
                /* Flip bits so two's complement signed integers sort correctly. */
                u64 *v = (u64 *)((byte *)buffer + offset + i * stride);
                *v = PAK_I64_FLIP_BACK(*v);
            } break;

            case PAK_RADIX_SORT_F32:
            {
                /* Flip bits so doubles sort correctly. */
                u32 *v = (u32 *)((byte *)buffer + offset + i * stride);
                *v = PAK_F32_FLIP_BACK(*v);
            } break;

            case PAK_RADIX_SORT_INT32:
            {
                /* Flip bits so two's complement signed integers sort correctly. */
                u32 *v = (u32 *)((byte *)buffer + offset + i * stride);
                *v = PAK_I32_FLIP_BACK(*v);
            } break;

            case PAK_RADIX_SORT_INT16:
            {
                /* Flip bits so two's complement signed integers sort correctly. */
                u16 *v = (u16 *)((byte *)buffer + offset + i * stride);
                *v = PAK_I16_FLIP_BACK(*v);
            } break;

            case PAK_RADIX_SORT_INT8:
            {
                /* Flip bits so two's complement signed integers sort correctly. */
                u8 *v = (u8 *)((byte *)buffer + offset + i * stride);
                *v = PAK_I8_FLIP_BACK(*v);
            } break;

            /* Fall through with no-op */
            case PAK_RADIX_SORT_UINT64:
            case PAK_RADIX_SORT_UINT32:
            case PAK_RADIX_SORT_UINT16:
            case PAK_RADIX_SORT_UINT8: {}
        }
    }
}

static inline void
pak_radix_sort_8(void *buffer, size num, size offset, size stride, void *scratch, PakSortOrder order)
{
    i64 counts[256] = {0};

    /* Build the histograms. */
    byte *position = (byte *)buffer + offset;

    for(size i = 0; i < num; ++i)
    {
        u8 b0 = *(u8 *)position;
        counts[b0] += 1;
        position += stride;
    }

    /* Sum the histograms. */
    if(order == PAK_SORT_ASCENDING)
    {
        for(size i = 1; i < 256; ++i)
        {
            counts[i] += counts[i - 1];
        }
    }
    else
    {
        for(size i = 254; i >= 0; --i)
        {
            counts[i] += counts[i + 1];
        }
    }

    /* Build the output array. */
    void *dest = scratch;
    void *source = buffer;
    for(size i = num - 1; i >= 0; --i)
    {
        void *val_src = (byte *)source + i * stride;
        u8 cnts_idx = *((u8 *)val_src + offset);
        void *val_dest = (byte *)dest + (--counts[cnts_idx]) * stride;

        memcpy(val_dest, val_src, stride);
    }

    /* Copy back to the original buffer. */
    memcpy(buffer, scratch, stride * num);
}

static inline void
pak_radix_sort_16(void *buffer, size num, size offset, size stride, void *scratch, PakSortOrder order)
{
    i64 counts[256][2] = {0};
    i32 skips[2] = {1, 1};

    /* Build the histograms. */
    byte *position = (byte *)buffer + offset;

    u16 val = *(u16 *)position;
    u8 b0 = UINT16_C(0xFF) & (val >>  0);
    u8 b1 = UINT16_C(0xFF) & (val >>  8);

    counts[b0][0] += 1;
    counts[b1][1] += 1;

    position += stride;

    u8 initial_values[2] = {b0, b1};

    for(size i = 1; i < num; ++i)
    {
        val = *(u16 *)position;
        b0 = UINT16_C(0xFF) & (val >>  0);
        b1 = UINT16_C(0xFF) & (val >>  8);

        counts[b0][0] += 1;
        counts[b1][1] += 1;

        skips[0] &= initial_values[0] == b0;
        skips[1] &= initial_values[1] == b1;

        position += stride;
    }

    /* Sum the histograms. */
    if(order == PAK_SORT_ASCENDING)
    {
        for(size i = 1; i < 256; ++i)
        {
            counts[i][0] += counts[i - 1][0];
            counts[i][1] += counts[i - 1][1];
        }
    }
    else
    {
        for(size i = 254; i >= 0; --i)
        {
            counts[i][0] += counts[i + 1][0];
            counts[i][1] += counts[i + 1][1];
        }
    }

    /* Build the output array. */
    void *dest = scratch;
    void *source = buffer;
    size num_skips = 0;
    for(size b = 0; b < 2; ++b)
    {
        if(!skips[b])
        {
            for(size i = num - 1; i >= 0; --i)
            {
                void *val_src = (byte *)source + i * stride;
                u16 val = *(u16 *)((byte *)val_src + offset);
                u8 cnts_idx = UINT16_C(0xFF) & (val >> (b * 8));
                void *val_dest = (byte *)dest + (--counts[cnts_idx][b]) * stride;

                memcpy(val_dest, val_src, stride);
            }

            /* Flip the source & destination buffers. */
            dest = dest == scratch ? buffer : scratch;
            source = source == scratch ? buffer : scratch;
        }
        else
        {
            num_skips++;
        }
    }

    /* Swap back into the original buffer. */
    if(num_skips % 2)
    {
        memcpy(buffer, scratch, num * stride);
    }
}

static inline void
pak_radix_sort_32(void *buffer, size num, size offset, size stride, void *scratch, PakSortOrder order)
{
    i64 counts[256][4] = {0};
    i32 skips[4] = {1, 1, 1, 1};

    /* Build the histograms. */
    byte *position = (byte *)buffer + offset;

    u32 val = *(u32 *)position;
    u8 b0 = UINT32_C(0xFF) & (val >>  0);
    u8 b1 = UINT32_C(0xFF) & (val >>  8);
    u8 b2 = UINT32_C(0xFF) & (val >> 16);
    u8 b3 = UINT32_C(0xFF) & (val >> 24);

    counts[b0][0] += 1;
    counts[b1][1] += 1;
    counts[b2][2] += 1;
    counts[b3][3] += 1;

    position += stride;

    u8 initial_values[4] = {b0, b1, b2, b3};

    for(size i = 1; i < num; ++i)
    {
        val = *(u32 *)position;
        b0 = UINT32_C(0xFF) & (val >>  0);
        b1 = UINT32_C(0xFF) & (val >>  8);
        b2 = UINT32_C(0xFF) & (val >> 16);
        b3 = UINT32_C(0xFF) & (val >> 24);

        counts[b0][0] += 1;
        counts[b1][1] += 1;
        counts[b2][2] += 1;
        counts[b3][3] += 1;

        skips[0] &= initial_values[0] == b0;
        skips[1] &= initial_values[1] == b1;
        skips[2] &= initial_values[2] == b2;
        skips[3] &= initial_values[3] == b3;

        position += stride;
    }

    /* Sum the histograms. */
    if(order == PAK_SORT_ASCENDING)
    {
        for(size i = 1; i < 256; ++i)
        {
            counts[i][0] += counts[i - 1][0];
            counts[i][1] += counts[i - 1][1];
            counts[i][2] += counts[i - 1][2];
            counts[i][3] += counts[i - 1][3];
        }
    }
    else
    {
        for(size i = 254; i >= 0; --i)
        {
            counts[i][0] += counts[i + 1][0];
            counts[i][1] += counts[i + 1][1];
            counts[i][2] += counts[i + 1][2];
            counts[i][3] += counts[i + 1][3];
        }
    }

    /* Build the output array. */
    void *dest = scratch;
    void *source = buffer;
    size num_skips = 0;
    for(size b = 0; b < 4; ++b)
    {
        if(!skips[b])
        {
            for(size i = num - 1; i >= 0; --i)
            {
                void *val_src = (byte *)source + i * stride;
                u32 val = *(u32 *)((byte *)val_src + offset);
                u8 cnts_idx = UINT32_C(0xFF) & (val >> (b * 8));
                void *val_dest = (byte *)dest + (--counts[cnts_idx][b]) * stride;

                memcpy(val_dest, val_src, stride);
            }

            /* Flip the source & destination buffers. */
            dest = dest == scratch ? buffer : scratch;
            source = source == scratch ? buffer : scratch;
        }
        else
        {
            num_skips++;
        }
    }

    /* Swap back into the original buffer. */
    if(num_skips % 2)
    {
        memcpy(buffer, scratch, num * stride);
    }
}

static inline void
pak_radix_sort_64(void *buffer, size num, size offset, size stride, void *scratch, PakSortOrder order)
{
    i64 counts[256][8] = {0};
    i32 skips[8] = {1, 1, 1, 1, 1, 1, 1, 1};

    /* Build the histograms. */
    byte *position = (byte *)buffer + offset;

    u64 val = *(u64 *)position;
    u8 b0 = UINT64_C(0xFF) & (val >>  0);
    u8 b1 = UINT64_C(0xFF) & (val >>  8);
    u8 b2 = UINT64_C(0xFF) & (val >> 16);
    u8 b3 = UINT64_C(0xFF) & (val >> 24);
    u8 b4 = UINT64_C(0xFF) & (val >> 32);
    u8 b5 = UINT64_C(0xFF) & (val >> 40);
    u8 b6 = UINT64_C(0xFF) & (val >> 48);
    u8 b7 = UINT64_C(0xFF) & (val >> 56);

    counts[b0][0] += 1;
    counts[b1][1] += 1;
    counts[b2][2] += 1;
    counts[b3][3] += 1;
    counts[b4][4] += 1;
    counts[b5][5] += 1;
    counts[b6][6] += 1;
    counts[b7][7] += 1;

    position += stride;

    u8 initial_values[8] = {b0, b1, b2, b3, b4, b5, b6, b7};

    for(size i = 1; i < num; ++i)
    {
        val = *(u64 *)position;
        b0 = UINT64_C(0xFF) & (val >>  0);
        b1 = UINT64_C(0xFF) & (val >>  8);
        b2 = UINT64_C(0xFF) & (val >> 16);
        b3 = UINT64_C(0xFF) & (val >> 24);
        b4 = UINT64_C(0xFF) & (val >> 32);
        b5 = UINT64_C(0xFF) & (val >> 40);
        b6 = UINT64_C(0xFF) & (val >> 48);
        b7 = UINT64_C(0xFF) & (val >> 56);

        counts[b0][0] += 1;
        counts[b1][1] += 1;
        counts[b2][2] += 1;
        counts[b3][3] += 1;
        counts[b4][4] += 1;
        counts[b5][5] += 1;
        counts[b6][6] += 1;
        counts[b7][7] += 1;

        skips[0] &= initial_values[0] == b0;
        skips[1] &= initial_values[1] == b1;
        skips[2] &= initial_values[2] == b2;
        skips[3] &= initial_values[3] == b3;
        skips[4] &= initial_values[4] == b4;
        skips[5] &= initial_values[5] == b5;
        skips[6] &= initial_values[6] == b6;
        skips[7] &= initial_values[7] == b7;

        position += stride;
    }

    /* Sum the histograms. */
    if(order == PAK_SORT_ASCENDING)
    {
        for(size i = 1; i < 256; ++i)
        {
            counts[i][0] += counts[i - 1][0];
            counts[i][1] += counts[i - 1][1];
            counts[i][2] += counts[i - 1][2];
            counts[i][3] += counts[i - 1][3];
            counts[i][4] += counts[i - 1][4];
            counts[i][5] += counts[i - 1][5];
            counts[i][6] += counts[i - 1][6];
            counts[i][7] += counts[i - 1][7];
        }
    }
    else
    {
        for(size i = 254; i >= 0; --i)
        {
            counts[i][0] += counts[i + 1][0];
            counts[i][1] += counts[i + 1][1];
            counts[i][2] += counts[i + 1][2];
            counts[i][3] += counts[i + 1][3];
            counts[i][4] += counts[i + 1][4];
            counts[i][5] += counts[i + 1][5];
            counts[i][6] += counts[i + 1][6];
            counts[i][7] += counts[i + 1][7];
        }
    }

    /* Build the output array. */
    void *dest = scratch;
    void *source = buffer;
    size num_skips = 0;
    for(size b = 0; b < 8; ++b)
    {
        if(!skips[b])
        {
            for(size i = num - 1; i >= 0; --i)
            {
                void *val_src = (byte *)source + i * stride;
                u64 val = *(u64 *)((byte *)val_src + offset);
                u8 cnts_idx = UINT64_C(0xFF) & (val >> (b * 8));
                void *val_dest = (byte *)dest + (--counts[cnts_idx][b]) * stride;

                memcpy(val_dest, val_src, stride);
            }

            /* Flip the source & destination buffers. */
            dest = dest == scratch ? buffer : scratch;
            source = source == scratch ? buffer : scratch;
        }
        else
        {
            num_skips++;
        }
    }

    /* Swap back into the original buffer. */
    if(num_skips % 2)
    {
        memcpy(buffer, scratch, num * stride);
    }
}

static inline void
pak_radix_sort(void *buffer, size num, size offset, size stride, void *scratch, PakRadixSortByType sort_type, PakSortOrder order)
{
    pak_radix_pre_sort_transform(buffer, num, offset, stride, sort_type);

    switch(sort_type)
    {
        case PAK_RADIX_SORT_UINT64:
        case PAK_RADIX_SORT_INT64:
        case PAK_RADIX_SORT_F64:
        {
            pak_radix_sort_64(buffer, num, offset, stride, scratch, order);
        } break;

        case PAK_RADIX_SORT_UINT32:
        case PAK_RADIX_SORT_INT32:
        case PAK_RADIX_SORT_F32:
        {
            pak_radix_sort_32(buffer, num, offset, stride, scratch, order);
        } break;

        case PAK_RADIX_SORT_UINT16:
        case PAK_RADIX_SORT_INT16:
        {
            pak_radix_sort_16(buffer, num, offset, stride, scratch, order);
        } break;

        case PAK_RADIX_SORT_UINT8:
        case PAK_RADIX_SORT_INT8:
        {
            pak_radix_sort_8(buffer, num, offset, stride, scratch, order);
        } break;

        default: Panic();
    }
    
    pak_radix_post_sort_transform(buffer, num, offset, stride, sort_type);
}

#endif

