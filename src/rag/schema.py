"""Pydantic schema for LLM-based metadata filter extraction."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class FiscalQuarter(str, Enum):
    Q1 = "q1"
    Q2 = "q2"
    Q3 = "q3"
    Q4 = "q4"


class ChunkMetadata(BaseModel):
    company: Optional[str] = Field(
        default=None,
        description=(
            "IBEX35 company name in uppercase. "
            "Examples: 'IBERDROLA', 'SANTANDER', 'INDITEX', 'BBVA', 'TELEFONICA', "
            "'ACCIONA', 'ACERINOX', 'ACS', 'AENA', 'AMADEUS', 'ARCELORMITTAL', "
            "'CELLNEX', 'ENAGAS', 'ENDESA', 'FERROVIAL', 'FLUIDRA', 'GRIFOLS', "
            "'IAG', 'INDRA', 'LOGISTA', 'MAPFRE', 'MERLINPROPERTIES', 'NATURGY', "
            "'PUIG', 'REDEIA', 'ROVI', 'SACYR', 'SOLARIA', 'UNICAJA'"
        ),
    )
    doc_type: Optional[str] = Field(
        default=None,
        description="Document type if mentioned: 'financial_results', '10-k', '10-q', etc.",
    )
    fiscal_year: Optional[str] = Field(
        default=None,
        description="Fiscal year if mentioned, e.g. '2024', '2023'",
    )
    fiscal_quarter: Optional[FiscalQuarter] = Field(
        default=None,
        description="Fiscal quarter if mentioned: q1, q2, q3, q4",
    )

    model_config = {"use_enum_values": True}
