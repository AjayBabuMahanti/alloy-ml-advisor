"""
Alloy cost and sourcing data.

Prices are INDICATIVE RANGES in USD/kg (2025-2026 market levels) for
mill-product aluminum — sheet, plate, extrusion, rod, forging. They reflect
approximate wholesale pricing collected from public Alcoa, Kaiser Aluminum,
Constellium, and Hydro datasheets plus distributor websites (Metalmen Sales,
Online Metals, Metal Supermarkets). Form-specific premiums (extrusion,
forging) are captured in `FORM_PREMIUM`.

For any serious procurement decision, users should request live quotes.
"""
from typing import Dict, List


# USD per kg, approximate mid-point of published ranges for wrought product.
# Ranges given as (low, high); the midpoint is what we display by default.
ALLOY_COST_USD_PER_KG: Dict[str, Dict[str, float]] = {
    "1xxx": {"low": 2.20, "high": 3.20, "note": "Commercially pure. Cheapest; foil and electrical."},
    "2xxx": {"low": 5.80, "high": 9.50, "note": "Heat-treatable aerospace grade. Copper adds cost."},
    "3xxx": {"low": 2.60, "high": 3.80, "note": "Can/cookware stock. Mass-produced, low cost."},
    "4xxx": {"low": 3.80, "high": 5.50, "note": "Welding filler and piston forgings."},
    "5xxx": {"low": 3.00, "high": 4.80, "note": "Marine-grade. Stable pricing, wide availability."},
    "6xxx": {"low": 2.80, "high": 4.20, "note": "Most common structural. Best cost-to-strength."},
    "7xxx": {"low": 6.50, "high": 11.00, "note": "Premium aerospace. Zinc and plate thickness drive price."},
    "8xxx": {"low": 10.00, "high": 18.00, "note": "Al-Li alloys. Lithium makes this the most expensive."},
    "Cast": {"low": 3.20, "high": 5.50, "note": "Cast product. Si content and alloy grade vary."},
}

# Multiplicative form premium applied on top of the series midpoint.
FORM_PREMIUM: Dict[str, float] = {
    "Sheet":     1.00,
    "Plate":     1.10,
    "Bar":       1.05,
    "Rod":       1.05,
    "Extrusion": 1.15,
    "Wire":      1.20,
    "Forging":   1.40,
    "Casting":   0.95,
    "Foil":      1.30,
    "Tube":      1.25,
    "Powder":    1.80,
}

# Sourcing suggestions (public distributors / mills for each form).
SOURCING: Dict[str, List[str]] = {
    "mills": [
        "Alcoa (alcoa.com)",
        "Kaiser Aluminum (kaiseraluminum.com)",
        "Constellium (constellium.com)",
        "Hydro Aluminium (hydro.com)",
        "Novelis (novelis.com)",
        "Hindalco (India) (hindalco.com)",
    ],
    "distributors_small_qty": [
        "Online Metals (onlinemetals.com)",
        "Metal Supermarkets (metalsupermarkets.com)",
        "Metalmen Sales (metalmensales.com)",
        "Speedy Metals (speedymetals.com)",
        "McMaster-Carr (mcmaster.com)",
    ],
    "india_specific": [
        "Hindalco Industries",
        "Jindal Aluminium",
        "NALCO (National Aluminium Company)",
        "Vedanta Aluminium",
        "Bharat Aluminium Company (BALCO)",
    ],
    "forgings_castings": [
        "Arconic (forgings)",
        "Precision Castparts (castings)",
        "Metalcasting.com (foundry directory)",
        "Aluminum Precision Products",
    ],
}


def estimate_cost(series: str, form: str = "Sheet") -> Dict[str, float]:
    """Return indicative cost dict for a given series + form.

    Returns
    -------
    dict with keys: unit_cost_usd_kg (mid), low, high, note, relative_tier
    """
    info = ALLOY_COST_USD_PER_KG.get(series, ALLOY_COST_USD_PER_KG["6xxx"])
    premium = FORM_PREMIUM.get(form, 1.00)
    mid = (info["low"] + info["high"]) / 2 * premium
    low = info["low"] * premium
    high = info["high"] * premium

    # Relative tier based on mid-price
    if mid < 4.0:
        tier = "economy"
    elif mid < 6.5:
        tier = "standard"
    elif mid < 10.0:
        tier = "premium"
    else:
        tier = "specialty"

    return {
        "unit_cost_usd_kg": round(mid, 2),
        "low": round(low, 2),
        "high": round(high, 2),
        "note": info["note"],
        "form_premium": premium,
        "tier": tier,
    }


def cost_for_volume(series: str, form: str, mass_kg: float) -> Dict[str, float]:
    """Estimate total USD for a given mass."""
    unit = estimate_cost(series, form)
    return {
        **unit,
        "mass_kg": mass_kg,
        "total_usd_mid": round(unit["unit_cost_usd_kg"] * mass_kg, 2),
        "total_usd_low": round(unit["low"] * mass_kg, 2),
        "total_usd_high": round(unit["high"] * mass_kg, 2),
    }


def sourcing_suggestions(series: str, form: str) -> List[str]:
    """Return a short list of where to buy this alloy/form."""
    suggestions: List[str] = []
    # Specialty paths first
    if form in ("Forging", "Casting"):
        suggestions.extend(SOURCING["forgings_castings"][:3])
    # General mills + distributors
    if series in ("2xxx", "7xxx", "8xxx"):
        # Aerospace-grade: mills first, specialty distributors
        suggestions.extend(SOURCING["mills"][:3])
    else:
        # Structural: mills plus general distributors for small qty
        suggestions.extend(SOURCING["distributors_small_qty"][:3])
        suggestions.append(SOURCING["mills"][0])
    # Deduplicate preserving order
    seen = set()
    out = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:5]


def cost_tier_color(tier: str) -> str:
    """Return the styles-module badge kind for a cost tier."""
    return {
        "economy":   "success",
        "standard":  "primary",
        "premium":   "warn",
        "specialty": "warn",
    }.get(tier, "muted")
