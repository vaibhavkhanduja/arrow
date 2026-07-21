#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Download, regenerate, and validate parquet-testing Variant fixtures.

This is a PyArrow counterpart of:
https://github.com/apache/parquet-testing/blob/master/variant/regen.py

The default workflow is deliberately direct:

1. Download ``data_dictionary.json`` from apache/parquet-testing.
2. Encode every JSON value and write its ``.metadata`` and ``.value`` files.
3. Create a PyArrow table, persist it as Parquet, and compare every generated
   binary file with its counterpart on GitHub.

JSON does not preserve integer widths, decimal scale, or object field order.
The built-in ``examples()`` data therefore acts as the physical encoding
template, while the downloaded dictionary is the logical-value source of truth.

Run from an empty output directory::

    python variant_regen_pyarrow.py --output-dir /tmp/variant-fixtures

Use ``--local-examples`` to generate without network validation.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import struct
import ssl
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin
from urllib.request import urlopen

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_GITHUB_VARIANT_URL = (
    "https://raw.githubusercontent.com/apache/parquet-testing/master/variant/"
)


@dataclass(frozen=True)
class Primitive:
    """A value with an explicitly selected Variant primitive physical type."""

    type_id: int
    value: Any


def parquet_variant_type() -> pa.DataType:
    """Return the canonical Variant type that writes ``VARIANT(1)``."""
    storage_type = pa.struct([
        pa.field("metadata", pa.binary(), nullable=False),
        pa.field("value", pa.binary(), nullable=False),
    ])
    return pa.variant(storage_type)


def _uint(value: int, width: int) -> bytes:
    return value.to_bytes(width, "little", signed=False)


def _sint(value: int, width: int) -> bytes:
    return value.to_bytes(width, "little", signed=True)


def _width(value: int) -> int:
    if value < 1 << 8:
        return 1
    if value < 1 << 16:
        return 2
    if value < 1 << 24:
        return 3
    if value < 1 << 32:
        return 4
    raise ValueError(f"Variant offset or dictionary ID is too large: {value}")


def _header(basic_type: int, value_header: int) -> bytes:
    return bytes([basic_type | (value_header << 2)])


def _decimal_bytes(value: Decimal, width: int) -> bytes:
    exponent = value.as_tuple().exponent
    scale = max(0, -exponent)
    if not 0 <= scale <= 38:
        raise ValueError(f"Decimal scale must be in [0, 38], got {scale}")
    unscaled = int(value.scaleb(scale))
    return bytes([scale]) + _sint(unscaled, width)


def _epoch_micros(value: datetime) -> int:
    if value.tzinfo is not None:
        return int(value.timestamp() * 1_000_000)
    epoch = datetime(1970, 1, 1)
    delta = value - epoch
    return ((delta.days * 86400 + delta.seconds) * 1_000_000
            + delta.microseconds)


def _primitive(type_id: int, value: Any) -> bytes:
    prefix = _header(0, type_id)
    if type_id in (0, 1, 2):
        return prefix
    if type_id == 3:
        return prefix + _sint(int(value), 1)
    if type_id == 4:
        return prefix + _sint(int(value), 2)
    if type_id == 5:
        return prefix + _sint(int(value), 4)
    if type_id == 6:
        return prefix + _sint(int(value), 8)
    if type_id == 7:
        return prefix + struct.pack("<d", float(value))
    if type_id == 8:
        return prefix + _decimal_bytes(Decimal(value), 4)
    if type_id == 9:
        return prefix + _decimal_bytes(Decimal(value), 8)
    if type_id == 10:
        return prefix + _decimal_bytes(Decimal(value), 16)
    if type_id == 11:
        days = (value - date(1970, 1, 1)).days
        return prefix + _sint(days, 4)
    if type_id in (12, 13):
        return prefix + _sint(_epoch_micros(value), 8)
    if type_id == 14:
        return prefix + struct.pack("<f", float(value))
    if type_id in (15, 16):
        data = value if isinstance(value, bytes) else value.encode("utf-8")
        return prefix + _uint(len(data), 4) + data
    if type_id == 17:
        micros = ((value.hour * 60 + value.minute) * 60 + value.second) * 1_000_000
        micros += value.microsecond
        return prefix + _sint(micros, 8)
    if type_id in (18, 19):
        # An integer is accepted to preserve nanoseconds that datetime cannot.
        nanos = value if isinstance(value, int) else _epoch_micros(value) * 1000
        return prefix + _sint(nanos, 8)
    if type_id == 20:
        value = value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
        return prefix + value.bytes
    raise ValueError(f"Unsupported Variant primitive type ID: {type_id}")


def _collect_keys(value: Any) -> list[str]:
    if isinstance(value, Primitive):
        return []
    if isinstance(value, dict):
        keys: list[str] = []
        for name, child in value.items():
            if name not in keys:
                keys.append(name)
            for child_name in _collect_keys(child):
                if child_name not in keys:
                    keys.append(child_name)
        return keys
    if isinstance(value, (list, tuple)):
        keys = []
        for child in value:
            for child_name in _collect_keys(child):
                if child_name not in keys:
                    keys.append(child_name)
        return keys
    return []


def encode_metadata(dictionary: Iterable[str]) -> tuple[bytes, dict[str, int]]:
    # Preserve first-seen order to match the original Spark-generated fixtures.
    # The metadata sorted_strings flag is therefore zero.
    strings = list(dict.fromkeys(dictionary))
    encoded = [item.encode("utf-8") for item in strings]
    offsets = [0]
    for item in encoded:
        offsets.append(offsets[-1] + len(item))
    width = _width(max(len(strings), offsets[-1]))
    header = 1 | ((width - 1) << 6)
    metadata = bytes([header]) + _uint(len(strings), width)
    metadata += b"".join(_uint(offset, width) for offset in offsets)
    metadata += b"".join(encoded)
    return metadata, {item: index for index, item in enumerate(strings)}


def _encode_array(value: list[Any], dictionary: dict[str, int]) -> bytes:
    children = [encode_value(item, dictionary) for item in value]
    offsets = [0]
    for child in children:
        offsets.append(offsets[-1] + len(child))
    offset_width = _width(offsets[-1])
    is_large = len(children) > 255
    value_header = ((1 if is_large else 0) << 2) | (offset_width - 1)
    count_width = 4 if is_large else 1
    return (_header(3, value_header) + _uint(len(children), count_width)
            + b"".join(_uint(item, offset_width) for item in offsets)
            + b"".join(children))


def _encode_object(value: dict[str, Any], dictionary: dict[str, int]) -> bytes:
    fields = list(value.items())
    children = [encode_value(child, dictionary) for _, child in fields]
    offsets = [0]
    for child in children:
        offsets.append(offsets[-1] + len(child))
    value_offsets = {name: offsets[index] for index, (name, _) in enumerate(fields)}
    sorted_names = sorted(value, key=lambda name: name.encode("utf-8"))
    field_ids = [dictionary[name] for name in sorted_names]
    field_offsets = [value_offsets[name] for name in sorted_names] + [offsets[-1]]
    id_width = _width(max(field_ids, default=0))
    offset_width = _width(offsets[-1])
    is_large = len(fields) > 255
    value_header = ((1 if is_large else 0) << 4) | ((id_width - 1) << 2)
    value_header |= offset_width - 1
    count_width = 4 if is_large else 1
    return (_header(2, value_header) + _uint(len(fields), count_width)
            + b"".join(_uint(item, id_width) for item in field_ids)
            + b"".join(_uint(item, offset_width) for item in field_offsets)
            + b"".join(children))


def encode_value(value: Any, dictionary: dict[str, int]) -> bytes:
    if isinstance(value, Primitive):
        return _primitive(value.type_id, value.value)
    if value is None:
        return _primitive(0, None)
    if value is True:
        return _primitive(1, True)
    if value is False:
        return _primitive(2, False)
    if isinstance(value, int):
        for type_id, width in ((3, 1), (4, 2), (5, 4), (6, 8)):
            if -(1 << (width * 8 - 1)) <= value < (1 << (width * 8 - 1)):
                return _primitive(type_id, value)
        raise ValueError(f"Integer is outside the Variant int64 range: {value}")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON Variant examples require finite floats")
        return _primitive(7, value)
    if isinstance(value, Decimal):
        precision = len(value.as_tuple().digits)
        type_id, width = ((8, 4) if precision <= 9 else
                          (9, 8) if precision <= 18 else (10, 16))
        return _header(0, type_id) + _decimal_bytes(value, width)
    if isinstance(value, str):
        data = value.encode("utf-8")
        if len(data) <= 63:
            return _header(1, len(data)) + data
        return _primitive(16, value)
    if isinstance(value, bytes):
        return _primitive(15, value)
    if isinstance(value, list):
        return _encode_array(value, dictionary)
    if isinstance(value, dict):
        return _encode_object(value, dictionary)
    raise TypeError(f"Cannot encode {type(value).__name__} as Variant")


def encode_variant(value: Any) -> tuple[bytes, bytes]:
    metadata, dictionary = encode_metadata(_collect_keys(value))
    return metadata, encode_value(value, dictionary)


def _ssl_context() -> ssl.SSLContext:
    """Return a certificate-verifying context, including common macOS paths."""
    paths = ssl.get_default_verify_paths()
    if paths.cafile and Path(paths.cafile).is_file():
        return ssl.create_default_context()
    for ca_file in (
        Path("/etc/ssl/cert.pem"),
        Path("/opt/homebrew/etc/ca-certificates/cert.pem"),
    ):
        if ca_file.is_file():
            return ssl.create_default_context(cafile=str(ca_file))
    return ssl.create_default_context()


def download_github_file(base_url: str, filename: str) -> bytes:
    """Download one file from a raw GitHub Variant fixture directory."""
    if not base_url.endswith("/"):
        base_url += "/"
    with urlopen(
        urljoin(base_url, filename), timeout=30, context=_ssl_context()
    ) as response:
        return response.read()


def parse_data_dictionary(raw: bytes) -> dict[str, Any]:
    """Parse data_dictionary.json, tolerating its historical trailing comma."""
    text = raw.decode("utf-8")
    try:
        result = json.loads(text)
    except json.JSONDecodeError as error:
        closing_brace = text.rfind("}")
        prefix = text[:closing_brace].rstrip()
        if closing_brace < 0 or not prefix.endswith(","):
            raise error
        result = json.loads(prefix[:-1] + text[len(prefix):])
    if not isinstance(result, dict):
        raise ValueError("data_dictionary.json must contain a JSON object")
    return result


def read_github_dictionary(base_url: str) -> dict[str, Any]:
    """Download and parse data_dictionary.json from ``base_url``."""
    return parse_data_dictionary(
        download_github_file(base_url, "data_dictionary.json")
    )


def _typed_github_value(name: str, value: Any) -> Any:
    """Restore physical type information that JSON cannot represent.

    Most values can be encoded directly. The upstream ``primitive_*`` fixture
    names are the type hints for values whose JSON representation is ambiguous.
    """
    primitive_types = {
        "primitive_null": 0,
        "primitive_boolean_true": 1,
        "primitive_boolean_false": 2,
        "primitive_int8": 3,
        "primitive_int16": 4,
        "primitive_int32": 5,
        "primitive_int64": 6,
        "primitive_double": 7,
        "primitive_decimal4": 8,
        "primitive_decimal8": 9,
        "primitive_decimal16": 10,
        "primitive_float": 14,
        "primitive_binary": 15,
        "primitive_string": 16,
        "primitive_uuid": 20,
    }
    if name in primitive_types:
        type_id = primitive_types[name]
        if type_id in (8, 9, 10):
            # JSON numbers do not preserve decimal scale (for example 12.340
            # and 12.34 are identical JSON), so the fixture contract supplies
            # the exact decimal spellings.
            value = Decimal({
                "primitive_decimal4": "12.34",
                "primitive_decimal8": "12345678.90",
                "primitive_decimal16": "12345678912345678.90",
            }[name])
        elif type_id == 15:
            value = base64.b64decode(value)
        elif type_id == 20:
            value = uuid.UUID(value)
        return Primitive(type_id, value)
    if name == "primitive_date":
        return Primitive(11, date.fromisoformat(value))
    if name == "primitive_timestamp":
        return Primitive(12, datetime.fromisoformat(value))
    if name == "primitive_timestampntz":
        return Primitive(13, datetime.fromisoformat(value))
    if name == "primitive_time":
        return Primitive(
            17, time.fromisoformat(value.replace("12:33:54:", "12:33:54."))
        )
    if name in ("primitive_timestamp_nanos", "primitive_timestampntz_nanos"):
        # Python datetime is only precise to microseconds. Preserve the exact
        # nanosecond value used by the upstream fixture.
        type_id = 18 if name == "primitive_timestamp_nanos" else 19
        return Primitive(type_id, 1_730_982_834_123_456_789)
    if name == "object_primitive":
        value = dict(value)
        value["double_field"] = Primitive(
            8, Decimal(str(value["double_field"]))
        )
    return value


def examples() -> list[tuple[str, Any, Any]]:
    """Return (fixture name, typed Variant value, JSON-compatible value)."""
    long_string = (
        "This string is longer than 64 bytes and therefore does not fit in a "
        "short_string and it also includes several non ascii characters such "
        "as 🐢, 💖, ♥️, 🎣 and 🤦!!"
    )
    binary = bytes.fromhex("031337deadbeefcafe")
    aware_timestamp = datetime.fromisoformat("2025-04-16T12:34:56.780000-04:00")
    naive_timestamp = datetime.fromisoformat("2025-04-16T12:34:56.780000")
    # datetime has microsecond precision, so spell the nanosecond epoch values out.
    nanos = 1_730_982_834_123_456_789
    return [
        ("primitive_null", Primitive(0, None), None),
        ("primitive_boolean_true", Primitive(1, True), True),
        ("primitive_boolean_false", Primitive(2, False), False),
        ("primitive_int8", Primitive(3, 42), 42),
        ("primitive_int16", Primitive(4, 1234), 1234),
        ("primitive_int32", Primitive(5, 123456), 123456),
        ("primitive_int64", Primitive(6, 1234567890123456789), 1234567890123456789),
        ("primitive_double", Primitive(7, 1234567890.1234), 1234567890.1234),
        ("primitive_decimal4", Primitive(8, Decimal("12.34")), 12.34),
        ("primitive_decimal8", Primitive(9, Decimal("12345678.90")), 12345678.9),
        ("primitive_decimal16", Primitive(10, Decimal("12345678912345678.90")), 1.2345678912345678e16),
        ("primitive_date", Primitive(11, date(2025, 4, 16)), "2025-04-16"),
        ("primitive_timestamp", Primitive(12, aware_timestamp), "2025-04-16 12:34:56.78-04:00"),
        ("primitive_timestampntz", Primitive(13, naive_timestamp), "2025-04-16 12:34:56.78"),
        ("primitive_float", Primitive(14, 1234567890.1234), 1234567940.0),
        ("primitive_binary", Primitive(15, binary), base64.b64encode(binary).decode()),
        ("primitive_string", Primitive(16, long_string), long_string),
        ("primitive_time", Primitive(17, time(12, 33, 54, 123456)), "12:33:54:123456"),
        ("primitive_timestamp_nanos", Primitive(18, nanos), "2024-11-07T12:33:54.123456789+00:00"),
        ("primitive_timestampntz_nanos", Primitive(19, nanos), "2024-11-07T12:33:54.123456789"),
        ("primitive_uuid", Primitive(20, uuid.UUID("f24f9b64-81fa-49d1-b74e-8c09a6e31c56")), "f24f9b64-81fa-49d1-b74e-8c09a6e31c56"),
        ("short_string", "Less than 64 bytes (❤️ with utf8)", "Less than 64 bytes (❤️ with utf8)"),
        ("object_empty", {}, {}),
        ("object_primitive", {
            "int_field": 1,
            "double_field": Primitive(8, Decimal("1.23456789")),
            "boolean_true_field": True,
            "boolean_false_field": False,
            "string_field": "Apache Parquet",
            "null_field": None,
            "timestamp_field": "2025-04-16T12:34:56.78",
        }, {
            "int_field": 1,
            "double_field": 1.23456789,
            "boolean_true_field": True,
            "boolean_false_field": False,
            "string_field": "Apache Parquet",
            "null_field": None,
            "timestamp_field": "2025-04-16T12:34:56.78",
        }),
        ("object_nested", {"id": 1, "species": {"name": "lava monster", "population": 6789}, "observation": {"time": "12:34:56", "location": "In the Volcano", "value": {"temperature": 123, "humidity": 456}}}, None),
        ("array_empty", [], []),
        ("array_primitive", [2, 1, 5, 9], [2, 1, 5, 9]),
        ("array_nested", [{"id": 1, "thing": {"names": ["Contrarian", "Spider"]}}, None, {"id": 2, "type": "if", "names": ["Apple", "Ray", None]}], None),
    ]


def _normalize_examples(rows: list[tuple[str, Any, Any]]) -> list[tuple[str, Any, Any]]:
    return [(name, value, value if json_value is None and isinstance(value, (dict, list)) else json_value)
            for name, value, json_value in rows]


def write_examples(
    rows: list[tuple[str, Any, Any]], output_dir: Path, parquet_name: str
) -> None:
    """Write named typed values as raw Variant fixtures and a Parquet table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _normalize_examples(rows)
    encoded = [(name, *encode_variant(value), json_value)
               for name, value, json_value in rows]

    for name, metadata, value, _ in encoded:
        (output_dir / f"{name}.metadata").write_bytes(metadata)
        (output_dir / f"{name}.value").write_bytes(value)

    data_dictionary = {name: json_value for name, _, _, json_value in encoded}
    (output_dir / "data_dictionary.json").write_text(
        json.dumps(data_dictionary, sort_keys=True, indent=4) + "\n",
        encoding="utf-8",
    )

    storage = pa.StructArray.from_arrays(
        [pa.array([row[1] for row in encoded], type=pa.binary()),
         pa.array([row[2] for row in encoded], type=pa.binary())],
        fields=list(parquet_variant_type().storage_type),
    )
    variants = pa.ExtensionArray.from_storage(parquet_variant_type(), storage)
    table = pa.table({
        "name": pa.array([row[0] for row in encoded]),
        "variant_col": variants,
        "json_col": pa.array([
            json.dumps(row[3], ensure_ascii=False, separators=(",", ":"))
            for row in encoded
        ]),
    })
    pq.write_table(table, output_dir / parquet_name, store_schema=True)


def github_rows(dictionary: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """Convert downloaded JSON into rows with explicit physical type hints.

    JSON loses decimal scale, integer width, and object field order. For known
    parquet-testing fixtures, ``examples()`` is therefore the physical encoding
    template. The downloaded JSON remains the source of truth for the logical
    value and must agree with that template before any bytes are generated.
    """
    templates = {
        name: (typed_value, json_value)
        for name, typed_value, json_value in _normalize_examples(examples())
    }
    rows = []
    for name, github_value in dictionary.items():
        if name in templates:
            typed_value, template_json = templates[name]
            if github_value != template_json:
                raise ValueError(
                    f"GitHub JSON for {name!r} differs from the encoding template"
                )
        else:
            typed_value = _typed_github_value(name, github_value)
        rows.append((name, typed_value, github_value))
    return rows


def describe_bytes_difference(generated: bytes, expected: bytes) -> str | None:
    """Describe the first byte difference, or return ``None`` when equal."""
    if generated == expected:
        return None

    common_length = min(len(generated), len(expected))
    offset = next(
        (
            index
            for index in range(common_length)
            if generated[index] != expected[index]
        ),
        common_length,
    )
    context_start = max(0, offset - 4)
    context_end = offset + 5
    return (
        f"first difference at byte {offset}; "
        f"generated length={len(generated)}, expected length={len(expected)}; "
        f"generated[{context_start}:{context_end}]="
        f"{generated[context_start:context_end].hex()}, "
        f"expected[{context_start}:{context_end}]="
        f"{expected[context_start:context_end].hex()}"
    )


def generate_and_validate_github_fixtures(
    base_url: str, output_dir: Path, parquet_name: str
) -> pa.Table:
    """Download JSON, generate fixtures and a table, and validate raw bytes.

    The returned table is read back from the generated Parquet file so callers
    can inspect exactly what was persisted.
    """
    dictionary = read_github_dictionary(base_url)
    write_examples(github_rows(dictionary), output_dir, parquet_name)

    mismatches: list[str] = []
    for name in dictionary:
        for suffix in ("metadata", "value"):
            filename = f"{name}.{suffix}"
            generated = (output_dir / filename).read_bytes()
            expected = download_github_file(base_url, filename)
            difference = describe_bytes_difference(generated, expected)
            if difference is not None:
                mismatches.append(f"{filename}: {difference}")

    if mismatches:
        joined = "\n  ".join(mismatches)
        raise ValueError(f"Generated fixtures differ from GitHub:\n  {joined}")
    return pq.read_table(output_dir / parquet_name)


def write_fixtures(output_dir: Path, parquet_name: str) -> None:
    write_examples(examples(), output_dir, parquet_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--parquet-name", default="variant.parquet")
    parser.add_argument(
        "--github-base-url",
        default=DEFAULT_GITHUB_VARIANT_URL,
        help="raw GitHub directory containing data_dictionary.json and fixtures",
    )
    parser.add_argument(
        "--local-examples",
        action="store_true",
        help="generate built-in examples instead of downloading and validating GitHub",
    )
    args = parser.parse_args()
    if args.local_examples:
        write_fixtures(args.output_dir, args.parquet_name)
        print(f"Wrote local Variant fixtures to {args.output_dir}")
    else:
        table = generate_and_validate_github_fixtures(
            args.github_base_url, args.output_dir, args.parquet_name
        )
        print(
            f"Validated {table.num_rows} GitHub fixtures; wrote "
            f"{args.parquet_name} to {args.output_dir}"
        )


if __name__ == "__main__":
    main()
