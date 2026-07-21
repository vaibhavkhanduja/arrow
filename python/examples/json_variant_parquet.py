"""
Example: Writing JSON as Parquet Variant Type using PyArrow

This example demonstrates how to:
1. Store JSON data as a Parquet variant type
2. Write JSON data to Parquet files with variant encoding
3. Read back and verify the round-trip

The variant type in Parquet is designed to store flexible, semi-structured data.
By storing JSON as a variant, you can efficiently represent heterogeneous data
while maintaining compatibility with the Parquet format.
"""

import json as json_module
import pyarrow as pa
import pyarrow.parquet as pq

from variant_regen_pyarrow import encode_variant


VARIANT_STORAGE_TYPE = pa.struct([
    pa.field('metadata', pa.binary(), nullable=False),
    pa.field('value', pa.binary(), nullable=False)
])


def json_variant_type():
    """Return the canonical type written as a Parquet ``VARIANT`` group."""
    variant = getattr(pa, 'variant', None)
    if variant is None:
        raise RuntimeError(
            "This example requires a PyArrow build with canonical Variant "
            "support (pa.variant). Rebuild PyArrow from the python directory "
            "with PYARROW_WITH_PARQUET=1, then rerun the example."
        )
    return variant(VARIANT_STORAGE_TYPE)


def create_json_variant_array(json_strings):
    """Convert a list of JSON strings to a variant array.
    
    Parameters
    ----------
    json_strings : List[str or None]
        List of JSON strings (can include None for null values)
    
    Returns
    -------
    ExtensionArray
        Array with JsonVariantExtensionType
    
    Examples
    --------
    >>> json_data = ['{"name": "Alice"}', '{"name": "Bob"}', None]
    >>> arr = create_json_variant_array(json_data)
    """
    metadata_list = []
    value_list = []
    null_mask = []

    for json_str in json_strings:
        if json_str is None:
            metadata_list.append(b'')
            value_list.append(b'')
            null_mask.append(True)
        else:
            # Parse the JSON and encode both buffers according to the Parquet
            # Variant binary encoding. The metadata buffer is required even
            # when the value has no object keys.
            try:
                parsed = json_module.loads(json_str)
            except (json_module.JSONDecodeError, TypeError):
                parsed = str(json_str)

            metadata, value = encode_variant(parsed)
            metadata_list.append(metadata)
            value_list.append(value)
            null_mask.append(False)

    # Create struct array with metadata and value fields
    struct_array = pa.StructArray.from_arrays(
        [
            pa.array(metadata_list, type=pa.binary()),
            pa.array(value_list, type=pa.binary())
        ],
        fields=[
            pa.field('metadata', pa.binary(), nullable=False),
            pa.field('value', pa.binary(), nullable=False)
        ],
        mask=pa.array(null_mask, type=pa.bool_())
    )
    
    # Wrap as extension array
    # The canonical extension type causes the enclosing Parquet group to be
    # annotated with VARIANT before its metadata and value child fields.
    ext_type = json_variant_type()
    return ext_type.wrap_array(struct_array)

def json_variant_to_storage(array):
    """Return Variant ``(metadata, value)`` buffers for comparison."""
    if isinstance(array, pa.ChunkedArray):
        out = []
        for chunk in array.chunks:
            out.extend(json_variant_to_storage(chunk))
        return out

    storage = array.storage if isinstance(array, pa.ExtensionArray) else array
    metadata = storage.field("metadata").to_pylist()
    values = storage.field("value").to_pylist()
    valid = storage.is_valid().to_pylist()

    return [
        (metadata_value, value) if is_valid else None
        for metadata_value, value, is_valid in zip(metadata, values, valid)
    ]


def validate_json_variant_column(original_table, result_table, column_name):
    """Validate a variant column by comparing its encoded buffers."""
    original_storage = json_variant_to_storage(original_table.column(column_name))
    result_storage = json_variant_to_storage(result_table.column(column_name))
    if result_storage != original_storage:
        raise AssertionError(
            f"Variant storage mismatch for column {column_name!r}:\n"
            f"original={original_storage!r}\n"
            f"result={result_storage!r}"
        )
    return True


def example_basic_json_variant():
    """Example 1: Basic JSON to Parquet variant."""
    print("=" * 70)
    print("Example 1: Basic JSON to Parquet Variant")
    print("=" * 70)
    
    # Sample JSON data
    json_data = [
        '{"product": "laptop", "price": 999.99}',
        '{"product": "phone", "price": 499.99}',
        '{"product": "tablet", "price": 299.99}',
        None  # Null value
    ]
    
    # Create variant array
    variant_array = create_json_variant_array(json_data)
    
    # Create Arrow table
    table = pa.table({"product_info": variant_array})
    print(f"\nArrow Table Schema:\n{table.schema}\n")
    print(f"Table:\n{table}\n")
    
    # Write to Parquet
    output_file = "/tmp/example_json_variant.parquet"
    pq.write_table(table, output_file, store_schema=True)
    print(f"✓ Written to {output_file}\n")
    
    # Read back
    result_table = pq.read_table(output_file, arrow_extensions_enabled=True)
    print(f"Read back schema:\n{result_table.schema}\n")
    print(
        "Read back JSON text matches: "
        f"{validate_json_variant_column(table, result_table, 'product_info')}\n"
    )
    
    return result_table


def example_multiple_json_columns():
    """Example 2: Multiple JSON variant columns."""
    print("=" * 70)
    print("Example 2: Multiple JSON Variant Columns")
    print("=" * 70)
    
    # Different JSON structures in different columns
    users_data = [
        '{"id": 1, "name": "Alice", "active": true}',
        '{"id": 2, "name": "Bob", "active": false}',
        None
    ]
    
    events_data = [
        '{"event": "login", "timestamp": "2024-01-01T10:00:00"}',
        '{"event": "logout", "timestamp": "2024-01-01T11:00:00"}',
        None
    ]
    
    # Create variant arrays
    users_array = create_json_variant_array(users_data)
    events_array = create_json_variant_array(events_data)
    
    # Create table with multiple columns
    table = pa.table({
        "user_info": users_array,
        "event_log": events_array
    })
    
    print(f"\nSchema:\n{table.schema}\n")
    
    # Write and read back
    output_file = "/tmp/example_multiple_variants.parquet"
    pq.write_table(table, output_file, store_schema=True)
    result = pq.read_table(output_file, arrow_extensions_enabled=True)
    users_match = validate_json_variant_column(table, result, "user_info")
    events_match = validate_json_variant_column(table, result, "event_log")
    
    print(f"✓ Wrote {table.num_rows} rows, {table.num_columns} columns")
    print(f"✓ JSON text roundtrip successful: {users_match and events_match}\n")
    
    return result


def example_nested_json():
    """Example 3: Deeply nested JSON structures."""
    print("=" * 70)
    print("Example 3: Nested JSON Structures in Variant")
    print("=" * 70)
    
    nested_data = [
        '''{
            "user": {
                "id": 1,
                "profile": {
                    "name": "Alice",
                    "contact": {
                        "email": "alice@example.com",
                        "phone": "555-0001"
                    }
                }
            }
        }''',
        '''{
            "user": {
                "id": 2,
                "profile": {
                    "name": "Bob",
                    "contact": {
                        "email": "bob@example.com"
                    }
                }
            }
        }''',
        None
    ]
    
    variant_array = create_json_variant_array(nested_data)
    table = pa.table({"nested_data": variant_array})
    
    # Write to Parquet
    output_file = "/tmp/example_nested_variant.parquet"
    pq.write_table(table, output_file, store_schema=True)
    result = pq.read_table(output_file, arrow_extensions_enabled=True)
    
    print(f"Original:\n{table}\n")
    print(
        "After roundtrip JSON text matches: "
        f"{validate_json_variant_column(table, result, 'nested_data')}\n"
    )
    
    return result


def example_json_arrays_in_variant():
    """Example 4: JSON arrays as variant."""
    print("=" * 70)
    print("Example 4: JSON Arrays in Variant")
    print("=" * 70)
    
    json_arrays = [
        '[1, 2, 3, 4, 5]',
        '["apple", "banana", "cherry"]',
        '[{"id": 1}, {"id": 2}]',
        None
    ]
    
    variant_array = create_json_variant_array(json_arrays)
    table = pa.table({"arrays": variant_array})
    
    # Roundtrip
    output_file = "/tmp/example_json_arrays.parquet"
    pq.write_table(table, output_file, store_schema=True)
    result = pq.read_table(output_file, arrow_extensions_enabled=True)
    arrays_match = validate_json_variant_column(table, result, "arrays")
    
    print(f"Data types: {[type(val) for val in result.column('arrays')]}\n")
    print(f"✓ JSON text roundtrip successful: {arrays_match}\n")
    
    return result


def example_mixed_null_handling():
    """Example 5: Handling null values."""
    print("=" * 70)
    print("Example 5: Null Value Handling")
    print("=" * 70)
    
    mixed_data = [
        '{"value": 1}',
        None,
        '{"value": 2}',
        None,
        '{"value": 3}'
    ]
    
    variant_array = create_json_variant_array(mixed_data)
    table = pa.table({"data": variant_array})
    
    # Check null counts
    print(f"Array: {table.column('data').to_pylist()}\n")
    print(f"Null count: {table.column('data').null_count}\n")
    
    # Roundtrip
    output_file = "/tmp/example_nulls.parquet"
    pq.write_table(table, output_file, store_schema=True)
    result = pq.read_table(output_file, arrow_extensions_enabled=True)
    nulls_match = validate_json_variant_column(table, result, "data")
    
    print(f"✓ JSON text roundtrip successful: {nulls_match}\n")
    
    return result


def example_performance_large_dataset():
    """Example 6: Writing large dataset as variant."""
    print("=" * 70)
    print("Example 6: Large Dataset Performance")
    print("=" * 70)
    
    import time
    
    # Generate large dataset
    num_rows = 100000
    json_data = [
        json_module.dumps({
            "id": i,
            "value": i * 2,
            "name": f"item_{i}",
            "tags": ["tag1", "tag2"] if i % 2 == 0 else ["tag3"]
        })
        for i in range(num_rows)
    ]
    
    start = time.time()
    variant_array = create_json_variant_array(json_data)
    table = pa.table({"data": variant_array})
    create_time = time.time() - start
    
    print(f"Created table with {num_rows} rows in {create_time:.2f}s")
    
    # Write
    output_file = "/tmp/example_large.parquet"
    start = time.time()
    pq.write_table(table, output_file, store_schema=True)
    write_time = time.time() - start
    
    print(f"Wrote to Parquet in {write_time:.2f}s")
    
    # Read
    start = time.time()
    result = pq.read_table(output_file, arrow_extensions_enabled=True)
    read_time = time.time() - start
    
    print(f"Read from Parquet in {read_time:.2f}s")
    print(
        "✓ JSON text integrity check passed: "
        f"{validate_json_variant_column(table, result, 'data')}\n"
    )


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  PyArrow: Writing JSON as Parquet Variant Type".ljust(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")
    
    # Run examples
    example_basic_json_variant()
    example_multiple_json_columns()
    example_nested_json()
    example_json_arrays_in_variant()
    example_mixed_null_handling()
    example_performance_large_dataset()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
