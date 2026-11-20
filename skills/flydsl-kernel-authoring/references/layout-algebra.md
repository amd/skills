## 2. Layout System (Core Abstraction)

### Core Types
| Type | Description | Example |
|------|-------------|---------|
| `!fly.int_tuple` | Integer tuple (can be nested) | `(8, 16)`, `(8, (4, 2))` |
| `!fly.layout` | (Shape, Stride) pair | `(8, 16):(1, 8)` (col-major) |
| `!fly.memref` | Memory reference with layout | Typed pointer + layout info |

### Construction
```python
import flydsl.expr as fx

shape = fx.make_shape(8, 16)              # IntTuple (8, 16)
stride = fx.make_stride(1, 8)             # IntTuple (1, 8)
layout = fx.make_layout(shape, stride)    # Layout (8,16):(1,8)

# Shorthand with Python tuples
layout = fx.make_layout((8, 16), (1, 8))

# Coordinates
coord = fx.make_coord(i, j)

# Nested shapes for hierarchical tiling
shape_nested = fx.make_shape(9, (4, 8))   # (9, (4, 8))

# Identity layout
identity = fx.make_identity_layout((M, N))
```

### Coordinate Mapping
The fundamental operation maps logical coordinates to physical memory indices.

**Formula**: `Index = sum(coord_i * stride_i)`

```python
idx = fx.crd2idx(coord, layout)    # Coordinate -> linear index
coord = fx.idx2crd(idx, layout)    # Linear index -> coordinate
s = fx.size(layout)                # Total element count (product of shape)
```

**Example**: For layout `(8, 16):(1, 8)` (8x16, column-major):
- `crd2idx((3, 5), layout)` = `3*1 + 5*8` = 43
- `idx2crd(43, layout)` = `(43 % 8, 43 / 8)` = `(3, 5)`

### Query Operations
```python
fx.size(layout)           # Total element count
fx.get_shape(layout)      # Extract shape IntTuple
fx.get_stride(layout)     # Extract stride IntTuple
fx.get(int_tuple, i)      # Get i-th element
fx.rank(int_tuple)        # Number of top-level modes
```

### Layout Algebra Operations

#### Composition: `fx.composition(A, B)`
Compose two layouts: `result(x) = A(B(x))`. Used to apply permutations or tile coordinate mappings.

#### Complement: `fx.complement(tiler, target_size)`
Compute remaining modes not covered by tiler, up to target_size. Internal building block for divides.

#### Coalesce: `fx.coalesce(layout)`
Simplify layout by merging adjacent modes. Preserves mapping but flattens structure.

#### Right Inverse: `fx.right_inverse(layout)`
Compute right inverse of layout mapping.

#### Recast: `fx.recast_layout(layout, old_bits, new_bits)`
Adjust layout for type width change (e.g., FP16->FP8).

### Product Operations (Combine Layouts)
Products combine two layouts to create a larger layout:

```python
fx.logical_product(layout, tiler)   # Basic mode-wise concatenation
fx.raked_product(thr, val)          # Interleaved access pattern (see make_layout_tv for TV layouts)
fx.blocked_product(layout, tiler)   # Blocked access pattern
fx.zipped_product(layout, tiler)    # Zipped modes
fx.tiled_product(layout, tiler)     # Hierarchical tiled structure
fx.flat_product(layout, tiler)      # Flattened result
```

### Divide Operations (Partition Layouts)
Divides split a layout by a divisor, creating tile + rest dimensions:

```python
fx.logical_divide(layout, divisor)  # Basic partitioning (uses complement internally)
fx.zipped_divide(layout, divisor)   # Zipped division
fx.tiled_divide(layout, divisor)    # Hierarchical tiled division
fx.flat_divide(layout, divisor)     # Flattened division
```

### Structural Operations
```python
fx.select(int_tuple, indices=[0, 2])      # Pick specific modes
fx.group(int_tuple, begin=1, end=3)        # Group modes into nested tuple
fx.append(base, elem)                      # Append mode
fx.prepend(base, elem)                     # Prepend mode
fx.slice(src, coord)                       # Slice at coordinate (None = keep mode)
```
