"""Database connectors: `vector` for similarity search, `text` for documents, `tabular` for SQL.

Every connector in a module shares one interface, and each imports its driver lazily inside
`_initialize_connection`, so using one engine never requires installing the rest.
"""
