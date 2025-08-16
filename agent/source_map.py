from dataclasses import dataclass


@dataclass
class SourceMapEntry:
    original: str
    transformed: str
    line_no: int
    offset: int
    type: str
