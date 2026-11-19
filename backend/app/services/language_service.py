"""
Language & Country Expansion Service for Faceify Labs.

Source of Truth: Languages.pdf
Includes:
- Target languages and countries
- Priority tiers (High, Medium, Selective, Supported)
- Rollout phases (Phase 1, Phase 2, Phase 3, Current)
- Economic status and clinical/aesthetic market rationale
- Standardized country metadata (ISO-2, ISO-3, region group)
- Multi-language to multi-country associations
- PDF & JSON report generation preserving Languages.pdf content
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Source of Truth: Languages.pdf Data                                         #
# --------------------------------------------------------------------------- #

LANGUAGE_STRATEGY: Dict[str, Dict[str, Any]] = {
    # Phase 1 — High Priority
    "Spanish": {
        "priority": "High",
        "phase": "Phase 1",
        "phase_num": 1,
        "target_countries": ["Spain", "Mexico", "Colombia"],
        "economic_status": "High to upper-middle income",
        "why_consider_it": "Cosmetic procedures and medical tourism in Spain",
        "market_categories": [],
    },
    "Arabic": {
        "priority": "High",
        "phase": "Phase 1",
        "phase_num": 1,
        "target_countries": ["UAE", "Saudi Arabia", "Qatar"],
        "economic_status": "High income",
        "why_consider_it": "Premium aesthetic services (UAE); aesthetic clinics and cosmetic procedures (Saudi Arabia)",
        "market_categories": [],
    },
    "German": {
        "priority": "High",
        "phase": "Phase 1",
        "phase_num": 1,
        "target_countries": ["Germany", "Austria", "Switzerland"],
        "economic_status": "High income",
        "why_consider_it": "Affluent consumers and clinics; strong purchasing power in Switzerland",
        "market_categories": [],
    },
    "French": {
        "priority": "High",
        "phase": "Phase 1",
        "phase_num": 1,
        "target_countries": ["France", "Belgium", "Switzerland"],
        "economic_status": "High income",
        "why_consider_it": "Established aesthetic medicine market in France",
        "market_categories": [],
    },
    "Turkish": {
        "priority": "High",
        "phase": "Phase 1",
        "phase_num": 1,
        "target_countries": ["Turkey"],
        "economic_status": "Upper-middle income",
        "why_consider_it": "Cosmetic surgery and medical tourism",
        "market_categories": [],
    },

    # Phase 2 — Medium Priority
    "Italian": {
        "priority": "Medium",
        "phase": "Phase 2",
        "phase_num": 2,
        "target_countries": ["Italy", "Switzerland"],
        "economic_status": "High income",
        "why_consider_it": "High purchasing power; extends reach in Switzerland",
        "market_categories": [],
    },
    "Mandarin Chinese": {
        "priority": "Medium",
        "phase": "Phase 2",
        "phase_num": 2,
        "target_countries": ["China", "Taiwan"],
        "economic_status": "Upper-middle to high income",
        "why_consider_it": "Large potential customer base; assess entry requirements",
        "market_categories": [],
    },
    "Dutch": {
        "priority": "Medium",
        "phase": "Phase 2",
        "phase_num": 2,
        "target_countries": ["Netherlands", "Belgium"],
        "economic_status": "High income",
        "why_consider_it": "High-income markets",
        "market_categories": [],
    },
    "Malay": {
        "priority": "Medium",
        "phase": "Phase 2",
        "phase_num": 2,
        "target_countries": ["Malaysia", "Brunei"],
        "economic_status": "Upper-middle to high income",
        "why_consider_it": "Regional aesthetic market",
        "market_categories": [],
    },
    "Tamil": {
        "priority": "Medium",
        "phase": "Phase 2",
        "phase_num": 2,
        "target_countries": ["Singapore", "India", "Sri Lanka"],
        "economic_status": "Varies by market",
        "why_consider_it": "Singapore is the priority market: premium clinics and international clientele",
        "market_categories": [],
    },

    # Phase 3 — Selective Priority
    "Hindi": {
        "priority": "Selective",
        "phase": "Phase 3",
        "phase_num": 3,
        "target_countries": ["India"],
        "economic_status": "Lower-middle income",
        "why_consider_it": "Affluent urban patients and clinics; broad reach",
        "market_categories": [],
    },
    "Indonesian": {
        "priority": "Selective",
        "phase": "Phase 3",
        "phase_num": 3,
        "target_countries": ["Indonesia"],
        "economic_status": "Upper-middle income",
        "why_consider_it": "Developing aesthetic market; validate demand",
        "market_categories": [],
    },
    "Russian": {
        "priority": "Selective",
        "phase": "Phase 3",
        "phase_num": 3,
        "target_countries": ["Russia"],
        "economic_status": "Varies by market",
        "why_consider_it": "Expand only if demand is measured",
        "market_categories": ["Russian-speaking markets"],
    },
    "Malayalam": {
        "priority": "Selective",
        "phase": "Phase 3",
        "phase_num": 3,
        "target_countries": ["India"],
        "economic_status": "Varies by market",
        "why_consider_it": "Validate demand among clinics and overseas communities",
        "market_categories": ["Overseas communities"],
    },
    "Telugu": {
        "priority": "Selective",
        "phase": "Phase 3",
        "phase_num": 3,
        "target_countries": ["India"],
        "economic_status": "Varies by market",
        "why_consider_it": "Validate demand in major urban markets",
        "market_categories": [],
    },

    # Currently Supported Languages (from Languages.pdf header)
    "English": {
        "priority": "Supported",
        "phase": "Current",
        "phase_num": 0,
        "target_countries": ["United States", "United Kingdom", "Australia", "Singapore"],
        "economic_status": "High income",
        "why_consider_it": "Core international operating language",
        "market_categories": [],
    },
    "Japanese": {
        "priority": "Supported",
        "phase": "Current",
        "phase_num": 0,
        "target_countries": ["Japan"],
        "economic_status": "High income",
        "why_consider_it": "Core APAC aesthetic market",
        "market_categories": [],
    },
    "Portuguese": {
        "priority": "Supported",
        "phase": "Current",
        "phase_num": 0,
        "target_countries": ["Brazil", "Portugal"],
        "economic_status": "Upper-middle to high income",
        "why_consider_it": "Leading global plastic surgery and aesthetic volume",
        "market_categories": [],
    },
    "Thai": {
        "priority": "Supported",
        "phase": "Current",
        "phase_num": 0,
        "target_countries": ["Thailand"],
        "economic_status": "Upper-middle income",
        "why_consider_it": "Major Asian medical tourism and cosmetic surgery hub",
        "market_categories": [],
    },
    "Korean": {
        "priority": "Supported",
        "phase": "Current",
        "phase_num": 0,
        "target_countries": ["Korea"],
        "economic_status": "High income",
        "why_consider_it": "Global capital of facial aesthetics and rhinoplasty innovation",
        "market_categories": [],
    },
}

ROLLOUT_PHASES: List[Dict[str, Any]] = [
    {
        "phase": "Phase 1",
        "priority": "High",
        "languages": ["Spanish", "Arabic", "German", "French", "Turkish"],
        "objective": "Target the most promising international aesthetic markets",
    },
    {
        "phase": "Phase 2",
        "priority": "Medium",
        "languages": ["Italian", "Mandarin Chinese", "Dutch", "Malay", "Tamil"],
        "objective": "Expand further and support Singapore",
    },
    {
        "phase": "Phase 3",
        "priority": "Selective",
        "languages": ["Hindi", "Indonesian", "Russian", "Malayalam", "Telugu"],
        "objective": "Expand selectively, based on measured demand",
    },
]

# Standardized Country Master List (ISO 3166-1 alpha-2 and alpha-3)
STANDARDIZED_COUNTRIES: Dict[str, Dict[str, str]] = {
    # Phase 1 Target Countries
    "Spain": {"iso2": "ES", "iso3": "ESP", "region_group": "Europe"},
    "Mexico": {"iso2": "MX", "iso3": "MEX", "region_group": "Americas"},
    "Colombia": {"iso2": "CO", "iso3": "COL", "region_group": "Americas"},
    "UAE": {"iso2": "AE", "iso3": "ARE", "region_group": "Middle East"},
    "Saudi Arabia": {"iso2": "SA", "iso3": "SAU", "region_group": "Middle East"},
    "Qatar": {"iso2": "QA", "iso3": "QAT", "region_group": "Middle East"},
    "Germany": {"iso2": "DE", "iso3": "DEU", "region_group": "Europe"},
    "Austria": {"iso2": "AT", "iso3": "AUT", "region_group": "Europe"},
    "Switzerland": {"iso2": "CH", "iso3": "CHE", "region_group": "Europe"},
    "France": {"iso2": "FR", "iso3": "FRA", "region_group": "Europe"},
    "Belgium": {"iso2": "BE", "iso3": "BEL", "region_group": "Europe"},
    "Turkey": {"iso2": "TR", "iso3": "TUR", "region_group": "Eurasia"},

    # Phase 2 Target Countries
    "Italy": {"iso2": "IT", "iso3": "ITA", "region_group": "Europe"},
    "China": {"iso2": "CN", "iso3": "CHN", "region_group": "East Asia"},
    "Taiwan": {"iso2": "TW", "iso3": "TWN", "region_group": "East Asia"},
    "Netherlands": {"iso2": "NL", "iso3": "NLD", "region_group": "Europe"},
    "Malaysia": {"iso2": "MY", "iso3": "MYS", "region_group": "Southeast Asia"},
    "Brunei": {"iso2": "BN", "iso3": "BRN", "region_group": "Southeast Asia"},
    "Singapore": {"iso2": "SG", "iso3": "SGP", "region_group": "Southeast Asia"},
    "India": {"iso2": "IN", "iso3": "IND", "region_group": "South Asia"},
    "Sri Lanka": {"iso2": "LK", "iso3": "LKA", "region_group": "South Asia"},

    # Phase 3 Target Countries
    "Indonesia": {"iso2": "ID", "iso3": "IDN", "region_group": "Southeast Asia"},
    "Russia": {"iso2": "RU", "iso3": "RUS", "region_group": "Eastern Europe"},

    # Currently Supported Countries
    "Japan": {"iso2": "JP", "iso3": "JPN", "region_group": "East Asia"},
    "Korea": {"iso2": "KR", "iso3": "KOR", "region_group": "East Asia"},
    "Thailand": {"iso2": "TH", "iso3": "THA", "region_group": "Southeast Asia"},
    "Vietnam": {"iso2": "VN", "iso3": "VNM", "region_group": "Southeast Asia"},
    "Brazil": {"iso2": "BR", "iso3": "BRA", "region_group": "Americas"},
    "Portugal": {"iso2": "PT", "iso3": "PRT", "region_group": "Europe"},
    "United States": {"iso2": "US", "iso3": "USA", "region_group": "Americas"},
    "United Kingdom": {"iso2": "GB", "iso3": "GBR", "region_group": "Europe"},
    "Australia": {"iso2": "AU", "iso3": "AUS", "region_group": "Oceania"},
}


class LanguageService:
    """Service providing language strategy metadata, country mappings, and report generation."""

    @staticmethod
    def get_language_catalog() -> List[Dict[str, Any]]:
        """Return full list of languages and their strategic metadata."""
        catalog = []
        for lang, data in LANGUAGE_STRATEGY.items():
            catalog.append({
                "language": lang,
                "priority": data["priority"],
                "phase": data["phase"],
                "phase_num": data["phase_num"],
                "target_countries": list(data["target_countries"]),
                "economic_status": data["economic_status"],
                "why_consider_it": data["why_consider_it"],
                "market_categories": list(data["market_categories"]),
            })
        return catalog

    @staticmethod
    def get_rollout_phases() -> List[Dict[str, Any]]:
        """Return rollout phases and target objectives."""
        return list(ROLLOUT_PHASES)

    @staticmethod
    def get_supported_countries() -> List[Dict[str, Any]]:
        """
        Return all unique countries with ISO codes, geographic region group,
        and all associated languages and priority tiers.
        """
        result = []
        for country, cdata in STANDARDIZED_COUNTRIES.items():
            # Find all languages that map to this country
            associated_languages = []
            highest_priority = "Selective"
            phases = []

            for lang, ldata in LANGUAGE_STRATEGY.items():
                if country in ldata["target_countries"]:
                    associated_languages.append(lang)
                    phases.append(ldata["phase"])
                    if ldata["priority"] == "High":
                        highest_priority = "High"
                    elif ldata["priority"] == "Medium" and highest_priority != "High":
                        highest_priority = "Medium"
                    elif ldata["priority"] == "Supported" and highest_priority not in ("High", "Medium"):
                        highest_priority = "Supported"

            result.append({
                "country": country,
                "iso2": cdata["iso2"],
                "iso3": cdata["iso3"],
                "region_group": cdata["region_group"],
                "associated_languages": associated_languages,
                "phases": list(set(phases)),
                "highest_priority": highest_priority,
            })
        return result

    @staticmethod
    def get_languages_for_country(country_name: str) -> List[str]:
        """Return all languages associated with a specific country."""
        langs = []
        for lang, ldata in LANGUAGE_STRATEGY.items():
            if country_name in ldata["target_countries"]:
                langs.append(lang)
        return langs

    @staticmethod
    def get_strategy_summary() -> Dict[str, Any]:
        """Return structured summary matching the full Languages.pdf document."""
        return {
            "title": "FACEIFY LABS — Language Expansion Strategy – Summary",
            "currently_supported": ["English", "Japanese", "Portuguese", "Thai", "Korean"],
            "languages": LanguageService.get_language_catalog(),
            "rollout_phases": LanguageService.get_rollout_phases(),
            "total_countries": len(STANDARDIZED_COUNTRIES),
            "countries": LanguageService.get_supported_countries(),
        }

    @staticmethod
    def generate_strategy_pdf(output_path: Optional[Path] = None) -> bytes:
        """
        Generate a professional PDF report containing the complete Language Expansion Strategy
        using ReportLab, matching the structure, content, and rationale from Languages.pdf.
        """
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#1A365D"),
            spaceAfter=4,
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#2B6CB0"),
            spaceAfter=10,
        )
        body_style = ParagraphStyle(
            "DocBody",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
        )
        header_cell_style = ParagraphStyle(
            "HeaderCell",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
            fontName="Helvetica-Bold",
            textColor=colors.white,
        )

        story = []

        # 1. Title Block
        story.append(Paragraph("<b>FACEIFY LABS</b>", title_style))
        story.append(Paragraph("Language Expansion Strategy – Summary", subtitle_style))
        story.append(
            Paragraph(
                "<b>Currently supported:</b> English, Japanese, Portuguese, Thai, Korean.",
                ParagraphStyle("CurrSupp", parent=styles["Normal"], fontSize=9, leading=12, textColor=colors.HexColor("#2D3748")),
            )
        )
        story.append(Spacer(1, 10))

        # 2. Main Languages Table
        story.append(
            Paragraph(
                "<b>Languages to add, target countries and why</b>",
                ParagraphStyle("SecHeader", parent=styles["Heading2"], fontSize=11, leading=14, textColor=colors.HexColor("#2C5282")),
            )
        )
        story.append(Spacer(1, 6))

        table_data = [
            [
                Paragraph("<b>Priority</b>", header_cell_style),
                Paragraph("<b>Language</b>", header_cell_style),
                Paragraph("<b>Target countries</b>", header_cell_style),
                Paragraph("<b>Economic status</b>", header_cell_style),
                Paragraph("<b>Why consider it</b>", header_cell_style),
            ]
        ]

        # Filter only expansion languages (High, Medium, Selective)
        expansion_langs = [
            ("High", "Spanish"),
            ("High", "Arabic"),
            ("High", "German"),
            ("High", "French"),
            ("High", "Turkish"),
            ("Medium", "Italian"),
            ("Medium", "Mandarin Chinese"),
            ("Medium", "Dutch"),
            ("Medium", "Malay"),
            ("Medium", "Tamil"),
            ("Selective", "Hindi"),
            ("Selective", "Indonesian"),
            ("Selective", "Russian"),
            ("Selective", "Malayalam"),
            ("Selective", "Telugu"),
        ]

        for prio, lang in expansion_langs:
            info = LANGUAGE_STRATEGY[lang]
            countries_str = ", ".join(info["target_countries"])
            if info["market_categories"]:
                countries_str += f" and {', '.join(info['market_categories'])}"

            table_data.append([
                Paragraph(f"<b>{prio}</b>", body_style),
                Paragraph(f"<b>{lang}</b>", body_style),
                Paragraph(countries_str, body_style),
                Paragraph(info["economic_status"], body_style),
                Paragraph(info["why_consider_it"], body_style),
            ])

        col_widths = [55, 75, 110, 100, 200]
        lang_table = Table(table_data, colWidths=col_widths, repeatRows=1)
        lang_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
            ])
        )
        story.append(lang_table)
        story.append(Spacer(1, 14))

        # 3. Rollout Phases Table
        story.append(
            Paragraph(
                "<b>Rollout phases</b>",
                ParagraphStyle("SecHeader2", parent=styles["Heading2"], fontSize=11, leading=14, textColor=colors.HexColor("#2C5282")),
            )
        )
        story.append(Spacer(1, 6))

        phase_data = [
            [
                Paragraph("<b>Phase</b>", header_cell_style),
                Paragraph("<b>Languages</b>", header_cell_style),
                Paragraph("<b>Objective</b>", header_cell_style),
            ]
        ]
        for p in ROLLOUT_PHASES:
            phase_data.append([
                Paragraph(f"<b>{p['phase']}</b>", body_style),
                Paragraph(", ".join(p["languages"]), body_style),
                Paragraph(p["objective"], body_style),
            ])

        phase_table = Table(phase_data, colWidths=[70, 220, 250], repeatRows=1)
        phase_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
            ])
        )
        story.append(phase_table)

        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "wb") as pf:
                pf.write(pdf_bytes)
            logger.info("Saved language strategy PDF to %s (%d bytes)", out_p, len(pdf_bytes))

        return pdf_bytes


language_service = LanguageService()
