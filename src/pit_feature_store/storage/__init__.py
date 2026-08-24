"""Parquet-on-disk storage layer: partition grammar, schemas, writer, reader.

The storage backend is deliberately boring and portable: Hive-partitioned
Parquet files read back through DuckDB. Nothing here is domain-specific.
"""
