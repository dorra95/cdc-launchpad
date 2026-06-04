"""
CDC LAUNCHPAD - AI-powered startup assessment and decision support.

Single-file Streamlit app built around the real CDC Tunisia workbook.
Pure engine functions are kept above the Streamlit UI so they can be checked
without launching the app.
"""

from __future__ import annotations

import datetime as dt
import io
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "Startups_Tunisia_Master_v4.xlsx")
LOGO_FILE = os.path.join(SCRIPT_DIR, "cdc_logo.png")
STORE_FILE = os.path.join(SCRIPT_DIR, "cdc_learning_store.csv")

ACCESS_CODE = "CDC2026"
ADMIN_EMAIL = "dorra.fadhloun@msb.tn"
ANALYSIS_YEAR = 2026
RANDOM_STATE = 42

NAVY = "#272E5F"
RED = "#D10A11"
INK = "#1F2937"
MUTED = "#6B7280"
LINE = "#D9DCE6"
LIGHT = "#F4F5F9"
GREEN = "#227A4A"
AMBER = "#B7791F"
BLUE = "#2563EB"

REGIONS_TN = [
    "Tunis",
    "Ariana",
    "Ben Arous",
    "Manouba",
    "Nabeul",
    "Zaghouan",
    "Bizerte",
    "Beja",
    "Jendouba",
    "Kef",
    "Siliana",
    "Sousse",
    "Monastir",
    "Mahdia",
    "Sfax",
    "Kairouan",
    "Kasserine",
    "Sidi Bouzid",
    "Gabes",
    "Medenine",
    "Tataouine",
    "Gafsa",
    "Tozeur",
    "Kebili",
]

FEATURES = [
    "company_age",
    "n_founders",
    "is_labelled",
    "has_email",
    "has_web",
    "sector_baserate",
]

PRETTY_FEATURES = {
    "company_age": "Company maturity",
    "n_founders": "Founding team size",
    "is_labelled": "Startup Act label",
    "has_email": "Verified contact",
    "has_web": "Web presence",
    "sector_baserate": "Sector funding track record",
}

TX = {
    "subtitle": {
        "EN": "AI-powered startup assessment and investment decision support",
        "FR": "Evaluation startup par IA et aide a la decision d'investissement",
    },
    "access": {"EN": "Secure access", "FR": "Acces securise"},
    "email": {"EN": "Professional email", "FR": "Email professionnel"},
    "request": {"EN": "Request access code", "FR": "Demander un code"},
    "code": {"EN": "Access code", "FR": "Code d'acces"},
    "enter": {"EN": "Enter platform", "FR": "Entrer"},
    "bad_code": {
        "EN": "Incorrect code. Please check with the administrator.",
        "FR": "Code incorrect. Veuillez verifier avec l'administrateur.",
    },
    "sent": {
        "EN": "Access request registered. Administrator contact:",
        "FR": "Demande enregistree. Contact administrateur :",
    },
    "tab_ecosystem": {"EN": "Ecosystem", "FR": "Ecosysteme"},
    "tab_portfolio": {"EN": "Portfolio", "FR": "Portefeuille"},
    "tab_assessment": {"EN": "Assessment", "FR": "Evaluation"},
    "tab_valuation": {"EN": "Valuation", "FR": "Valorisation"},
    "tab_learning": {"EN": "Data & Learning", "FR": "Donnees & apprentissage"},
    "tab_reports": {"EN": "Reports", "FR": "Rapports"},
    "logout": {"EN": "Log out", "FR": "Deconnexion"},
}


def t(key: str, lang: str) -> str:
    return TX.get(key, {}).get(lang, key)


# ---------------------------------------------------------------------------
# Data loading and feature engineering
# ---------------------------------------------------------------------------
def _first_sheet_containing(xls: pd.ExcelFile, keyword: str) -> str:
    for sheet in xls.sheet_names:
        if keyword.lower() in sheet.lower():
            return sheet
    raise ValueError(f"No sheet containing '{keyword}' was found.")


def _normalize_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def _normalize_header(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _find_column(df: pd.DataFrame, *tokens: str) -> str | None:
    wanted = [_normalize_header(token) for token in tokens]
    for column in df.columns:
        norm = _normalize_header(column)
        if all(token in norm for token in wanted):
            return str(column)
    return None


def _count_founders(value: Any) -> int:
    if pd.isna(value) or str(value).strip() == "":
        return 1
    text = str(value).strip()
    for sep in [";", "/", "|", ",", "\n", " et ", " and ", "&"]:
        text = text.replace(sep, ";")
    parts = [p.strip() for p in text.split(";") if p.strip()]
    if len(parts) > 1:
        return min(12, len(parts))

    # Some cells concatenate names without separators, for example
    # "Karim KharratNizar Ghram". Count likely capitalized name starts.
    starts = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?", text)
    return max(1, min(12, math.ceil(len(starts) / 2) if starts else 1))


def _as_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _money_to_float(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    text = str(value)
    if not text.strip():
        return float("nan")
    text = text.replace("\xa0", " ")
    match = re.search(r"-?\d[\d\s,.]*", text)
    if not match:
        return float("nan")
    number = match.group(0).replace(" ", "")
    if "," in number and "." in number:
        number = number.replace(",", "")
    elif "," in number:
        number = number.replace(",", ".")
    try:
        return float(number)
    except ValueError:
        return float("nan")


def _yes_from_notna(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0, index=df.index, dtype=int)
    return df[column].notna().astype(int)


def load_base(path: str = DATA_FILE, store: str = STORE_FILE) -> pd.DataFrame:
    """Load the workbook, create a funded label, and derive model features."""
    xls = pd.ExcelFile(path)
    base = pd.read_excel(xls, _first_sheet_containing(xls, "base"), header=1)
    financed = pd.read_excel(xls, _first_sheet_containing(xls, "financ"), header=1)

    base = base.dropna(subset=["Nom"]).copy()
    financed = financed.dropna(subset=["Nom"]).copy()

    funded_names = set(financed["Nom"].map(_normalize_name))
    base["funded"] = base["Nom"].map(_normalize_name).isin(funded_names).astype(int)

    year_col = _find_column(base, "annee", "creation")
    year = pd.to_numeric(base[year_col], errors="coerce") if year_col else pd.Series(np.nan, index=base.index)
    median_year = int(year.dropna().median()) if year.notna().any() else 2018
    base["founding_year"] = year.fillna(median_year).clip(1980, ANALYSIS_YEAR)
    base["company_age"] = (ANALYSIS_YEAR - base["founding_year"]).clip(0, 60)

    sector_col = "Secteur" if "Secteur" in base.columns else None
    if sector_col:
        base["sector"] = base[sector_col].astype(str).str.strip()
    else:
        base["sector"] = "Unknown"
    base["sector"] = base["sector"].replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})

    base["n_founders"] = (
        base["Founders"].apply(_count_founders) if "Founders" in base.columns else 1
    )
    base["is_labelled"] = _yes_from_notna(base, "Label Date")
    base["has_email"] = _yes_from_notna(base, "Courriel")
    base["has_web"] = _yes_from_notna(base, "Site web")

    if os.path.exists(store):
        try:
            extra = pd.read_csv(store)
            base = pd.concat([base, extra], ignore_index=True, sort=False)
        except Exception:
            pass

    for col in ["funded", "n_founders", "is_labelled", "has_email", "has_web"]:
        base[col] = _as_numeric(base[col], 0).astype(int)
    base["company_age"] = _as_numeric(base["company_age"], 8)
    base["sector"] = base["sector"].fillna("Unknown").astype(str)
    return base


def sector_rates(df: pd.DataFrame) -> tuple[dict[str, float], float]:
    global_rate = float(df["funded"].mean()) if len(df) else 0.0
    rates = df.groupby("sector")["funded"].mean().to_dict()
    return {str(k): float(v) for k, v in rates.items()}, global_rate


def build_features(
    df: pd.DataFrame, rate_map: dict[str, float], global_rate: float
) -> pd.DataFrame:
    sector = df["sector"].fillna("Unknown").astype(str)
    x = pd.DataFrame(
        {
            "company_age": _as_numeric(df["company_age"], 8).clip(0, 60),
            "n_founders": _as_numeric(df["n_founders"], 1).clip(1, 12),
            "is_labelled": _as_numeric(df["is_labelled"], 0).clip(0, 1),
            "has_email": _as_numeric(df["has_email"], 0).clip(0, 1),
            "has_web": _as_numeric(df["has_web"], 0).clip(0, 1),
            "sector_baserate": sector.map(rate_map).fillna(global_rate),
        }
    )
    return x[FEATURES]


@dataclass
class ModelBundle:
    model: Any
    metrics: dict[str, Any]
    rate_map: dict[str, float]
    global_rate: float
    importance: dict[str, float]
    signs: dict[str, float]
    feature_mean: dict[str, float]
    engine: str


class RuleScoreModel:
    """Small fallback scorer used when scikit-learn is unavailable."""

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        age_score = np.clip(x["company_age"].to_numpy() / 8.0, 0, 1)
        founder_score = np.clip(x["n_founders"].to_numpy() / 4.0, 0, 1)
        raw = (
            0.26 * x["sector_baserate"].to_numpy()
            + 0.22 * x["is_labelled"].to_numpy()
            + 0.16 * x["has_email"].to_numpy()
            + 0.16 * x["has_web"].to_numpy()
            + 0.10 * age_score
            + 0.10 * founder_score
        )
        proba = np.clip(raw, 0.03, 0.92)
        return np.column_stack([1 - proba, proba])


def train_selection(df: pd.DataFrame) -> ModelBundle:
    rate_map, global_rate = sector_rates(df)
    x = build_features(df, rate_map, global_rate)
    y = df["funded"].astype(int).to_numpy()

    model: Any = RuleScoreModel()
    engine = "Rule-based benchmark"
    metrics: dict[str, Any] = {
        "roc_auc": None,
        "f1": None,
        "accuracy": None,
        "n_rows": int(len(df)),
        "n_funded": int(df["funded"].sum()),
        "engine": engine,
    }

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        from sklearn.model_selection import train_test_split

        stratify = y if len(np.unique(y)) == 2 and min(np.bincount(y)) >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )
        weights = np.where(
            y_train == 1,
            max(1, int((y_train == 0).sum())) / max(1, int((y_train == 1).sum())),
            1.0,
        )
        model = GradientBoostingClassifier(
            n_estimators=180,
            max_depth=3,
            learning_rate=0.045,
            random_state=RANDOM_STATE,
        )
        model.fit(x_train, y_train, sample_weight=weights)
        proba = model.predict_proba(x_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = {
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 3),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 3),
            "accuracy": round(float(accuracy_score(y_test, pred)), 3),
            "n_rows": int(len(df)),
            "n_funded": int(df["funded"].sum()),
            "engine": "Gradient Boosting",
        }
        model.fit(x, y)
        engine = "Gradient Boosting"
    except Exception:
        pass

    proba_all = model.predict_proba(x)[:, 1]
    importance = {}
    if hasattr(model, "feature_importances_"):
        importance = dict(zip(FEATURES, [float(v) for v in model.feature_importances_]))
    if not importance or sum(importance.values()) == 0:
        importance = {
            "company_age": 0.10,
            "n_founders": 0.10,
            "is_labelled": 0.22,
            "has_email": 0.16,
            "has_web": 0.16,
            "sector_baserate": 0.26,
        }

    signs = {}
    for feature in FEATURES:
        try:
            corr = float(np.corrcoef(x[feature], y)[0, 1])
            signs[feature] = 1.0 if math.isnan(corr) or corr >= 0 else -1.0
        except Exception:
            signs[feature] = 1.0

    metrics["avg_score"] = round(float(np.mean(proba_all) * 100), 1)
    return ModelBundle(
        model=model,
        metrics=metrics,
        rate_map=rate_map,
        global_rate=global_rate,
        importance=importance,
        signs=signs,
        feature_mean=x.mean().to_dict(),
        engine=engine,
    )


def assess_one(bundle: ModelBundle, inputs: dict[str, Any]) -> dict[str, Any]:
    age = float(np.clip(ANALYSIS_YEAR - int(inputs.get("founding_year", 2021)), 0, 60))
    row = pd.DataFrame(
        [
            {
                "company_age": age,
                "n_founders": int(inputs.get("n_founders", 1)),
                "is_labelled": int(bool(inputs.get("is_labelled", False))),
                "has_email": int(bool(inputs.get("has_email", False))),
                "has_web": int(bool(inputs.get("has_web", False))),
                "sector": str(inputs.get("sector", "Unknown")),
            }
        ]
    )
    x = build_features(row, bundle.rate_map, bundle.global_rate)
    proba = float(bundle.model.predict_proba(x)[0, 1])
    score = round(proba * 100, 1)
    if proba >= 0.62:
        verdict, tone = "Recommended for selection", "ok"
    elif proba >= 0.42:
        verdict, tone = "Conditional - further due diligence", "warn"
    else:
        verdict, tone = "Not recommended - elevated risk", "bad"

    drivers = []
    for feature in FEATURES:
        delta = float(x[feature].iloc[0] - bundle.feature_mean.get(feature, 0))
        contribution = bundle.importance.get(feature, 0) * bundle.signs.get(feature, 1) * delta
        label = "supports" if contribution >= 0 else "weighs against"
        drivers.append((PRETTY_FEATURES[feature], label, abs(contribution)))
    drivers.sort(key=lambda item: item[2], reverse=True)
    return {
        "score": score,
        "proba": proba,
        "verdict": verdict,
        "tone": tone,
        "drivers": [(name, label) for name, label, _ in drivers[:5]],
    }


def lookup_startup(df: pd.DataFrame, query: str) -> dict[str, Any]:
    if not query or not str(query).strip():
        return {"level": "none", "title": "Enter a startup name or RNE.", "details": []}

    q = str(query).strip().lower()
    frames = [
        df[df["Nom"].astype(str).str.lower().str.contains(re.escape(q), na=False, regex=True)]
    ]
    for column in ["RNE", "Headquarter Unique Identifier", "Subsidiary Unique Identifier"]:
        if column in df.columns:
            frames.append(df[df[column].astype(str).str.lower().str.contains(re.escape(q), na=False)])
    hit = pd.concat(frames, ignore_index=False).drop_duplicates() if frames else pd.DataFrame()
    if hit.empty:
        return {"level": "green", "title": "No existing record in the CDC database.", "details": []}

    row = hit.iloc[0]
    details = [
        f"Name: {row.get('Nom', '-')}",
        f"Sector: {row.get('Secteur', row.get('sector', '-'))}",
        f"Founded: {row.get(_find_column(df, 'annee', 'creation') or 'founding_year', '-')}",
        f"Region: {row.get('Region', '-')}",
    ]

    advance = float("nan")
    for column in [
        "Avance Remboursable Flywheel (TND)",
        "Avance remboursable",
        "Montant Avance Remboursable",
    ]:
        if column in df.columns:
            advance = _money_to_float(row.get(column))
            if not math.isnan(advance):
                break

    if int(row.get("funded", 0)) == 1 and not math.isnan(advance) and advance > 0:
        details.append(f"Repayable advance: {advance:,.0f} TND")
        return {
            "level": "red",
            "title": "High risk - repayable advance on file; verify reimbursement.",
            "details": details,
        }
    if int(row.get("funded", 0)) == 1:
        return {
            "level": "amber",
            "title": "Caution - existing CDC beneficiary; review programme history.",
            "details": details,
        }
    return {
        "level": "blue",
        "title": "Information - in database, not previously funded.",
        "details": details,
    }


def valuation_engine(v: dict[str, Any]) -> dict[str, Any]:
    stage = int(v.get("stage", 1))
    team = float(v.get("team", 0.65))
    market = float(v.get("market", 0.65))
    product = float(v.get("product", 0.60))
    competition = float(v.get("competition", 0.50))
    revenue = float(v.get("revenue_tnd", 0.0) or 0.0)
    growth = float(v.get("growth", 0.40))

    cap = 500_000
    berkus = cap * 1.0 + cap * (stage / 2) + cap * team + cap * market + cap * product

    base_pre_money = 1_200_000
    factor = (
        team * 0.30
        + market * 0.25
        + product * 0.15
        + (1 - competition) * 0.10
        + 0.20
    )
    scorecard = base_pre_money * (0.5 + factor)

    risk_adj = (team - 0.5) + (market - 0.5) + (product - 0.5) + (0.5 - competition)
    risk_adj += stage / 2 - 0.5
    rfs = base_pre_money + risk_adj * 500_000

    if revenue > 0:
        exit_value = revenue * (1 + growth) ** 5 * 3.0
    else:
        exit_value = 8_000_000 * (0.4 + 0.4 * market)
    vc_method = exit_value / 10.0

    methods = {
        "Berkus": max(0, berkus),
        "Scorecard": max(0, scorecard),
        "Risk-Factor Summation": max(0, rfs),
        "Venture Capital Method": max(0, vc_method),
    }

    if revenue > 0:
        discount = 0.35
        cashflows = [revenue * (1 + growth) ** i * 0.20 for i in range(1, 6)]
        terminal = cashflows[-1] * 1.5
        dcf = sum(cf / (1 + discount) ** i for i, cf in enumerate(cashflows, 1))
        dcf += terminal / (1 + discount) ** 5
        methods["Hybrid DCF"] = max(0, dcf)

    values = np.array(list(methods.values()), dtype=float)
    return {
        "methods": methods,
        "low": float(np.percentile(values, 25)),
        "mid": float(np.percentile(values, 50)),
        "high": float(np.percentile(values, 75)),
    }


def segment_portfolio(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        rate_map, global_rate = sector_rates(work)
        features = pd.DataFrame(
            {
                "company_age": _as_numeric(work["company_age"], 8),
                "n_founders": _as_numeric(work["n_founders"], 1),
                "is_labelled": _as_numeric(work["is_labelled"], 0),
                "sector_baserate": work["sector"].map(rate_map).fillna(global_rate),
            }
        )
        z = StandardScaler().fit_transform(features)
        km = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE).fit(z)
        work["cluster"] = km.labels_
        age_by = work.groupby("cluster")["company_age"].mean()
        label_by = work.groupby("cluster")["is_labelled"].mean()
        mature = int(age_by.idxmax())
        remaining = [int(c) for c in age_by.index if int(c) != mature]
        labelled = max(remaining, key=lambda c: float(label_by.loc[c]))
        other = [c for c in remaining if c != labelled][0]
        names = {
            mature: "Mature funded-track cluster",
            labelled: "Label-ready digital cluster",
            other: "Early pipeline cluster",
        }
        work["profile"] = work["cluster"].map(names)
    except Exception:
        def _profile(row: pd.Series) -> str:
            if row["company_age"] >= 7:
                return "Mature funded-track cluster"
            if row["is_labelled"] == 1:
                return "Label-ready digital cluster"
            return "Early pipeline cluster"

        work["profile"] = work.apply(_profile, axis=1)

    summary = (
        work.groupby("profile")
        .agg(
            startups=("Nom", "count"),
            funded_rate=("funded", "mean"),
            avg_age=("company_age", "mean"),
        )
        .reset_index()
    )
    summary["funded_rate"] = (summary["funded_rate"] * 100).round(1)
    summary["avg_age"] = summary["avg_age"].round(1)
    return work, summary


def fetch_news(limit: int = 6) -> list[dict[str, str]]:
    feeds = [
        "https://news.google.com/rss/search?q=startup+Tunisie&hl=fr&gl=TN&ceid=TN:fr",
        "https://news.google.com/rss/search?q=Tunisia+startup+funding&hl=en-US&gl=US&ceid=US:en",
    ]
    items: list[dict[str, str]] = []
    try:
        import feedparser

        for feed in feeds:
            parsed = feedparser.parse(feed)
            for entry in parsed.entries[:limit]:
                source = entry.get("source", {}).get("title", "") if entry.get("source") else ""
                items.append(
                    {
                        "title": str(entry.get("title", "")),
                        "link": str(entry.get("link", "")),
                        "source": source,
                    }
                )
            if len(items) >= limit:
                break
    except Exception:
        pass
    if not items:
        items = [
            {
                "title": "Live ecosystem feed unavailable in this environment.",
                "link": "",
                "source": "CDC LAUNCHPAD",
            }
        ]
    return items[:limit]


def append_record(record: dict[str, Any], store: str = STORE_FILE) -> int:
    row = {
        "Nom": record["name"],
        "Secteur": record["sector"],
        "sector": record["sector"],
        "founding_year": int(record["year"]),
        "company_age": int(np.clip(ANALYSIS_YEAR - int(record["year"]), 0, 60)),
        "n_founders": int(record["founders"]),
        "is_labelled": int(bool(record.get("labelled", False))),
        "has_email": int(bool(record.get("email", True))),
        "has_web": int(bool(record.get("web", True))),
        "funded": int(record["funded"]),
    }
    new_row = pd.DataFrame([row])
    if os.path.exists(store):
        current = pd.read_csv(store)
        new_row = pd.concat([current, new_row], ignore_index=True)
    new_row.to_csv(store, index=False)
    return len(new_row)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def _gauge_png(score: float) -> io.BytesIO:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.3, 1.8), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    for low, high, color in [(0, 42, RED), (42, 62, "#E8A33D"), (62, 100, GREEN)]:
        ax.barh(
            1,
            np.radians((high - low) * 1.8),
            left=np.radians(low * 1.8),
            height=0.45,
            color=color,
        )
    angle = np.radians(score * 1.8)
    ax.plot([angle, angle], [0, 1.15], color=NAVY, lw=3)
    ax.set_axis_off()
    ax.set_title(f"{score:.0f}/100", color=NAVY, fontweight="bold", pad=2)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buffer.seek(0)
    return buffer


def assessment_pdf(payload: dict[str, Any]) -> io.BytesIO:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleCDC", parent=styles["Title"], textColor=colors.HexColor(NAVY))
    h2 = ParagraphStyle("H2CDC", parent=styles["Heading2"], textColor=colors.HexColor(NAVY))
    body = styles["BodyText"]
    elements: list[Any] = []

    if os.path.exists(LOGO_FILE):
        try:
            elements.append(Image(LOGO_FILE, width=48 * mm, height=19 * mm))
            elements.append(Spacer(1, 8))
        except Exception:
            pass

    elements.extend(
        [
            Paragraph("Startup Assessment Report", title),
            Paragraph(
                f"<b>{payload['name']}</b> | {payload['sector']} | {payload['region']} | "
                f"{dt.date.today():%d %b %Y}",
                body,
            ),
            Spacer(1, 8),
            Image(_gauge_png(payload["score"]), width=72 * mm, height=40 * mm),
        ]
    )
    tone = {"ok": GREEN, "warn": "#E8A33D", "bad": RED}.get(payload["tone"], NAVY)
    elements.extend(
        [
            Paragraph(f"<font color='{tone}'><b>{payload['verdict']}</b></font>", h2),
            Paragraph(
                f"Funding-likelihood score: <b>{payload['score']:.1f}/100</b> "
                f"(model probability {payload['proba']:.2f}).",
                body,
            ),
            Spacer(1, 8),
            Paragraph("Decision drivers", h2),
        ]
    )
    rows = [["Factor", "Effect"]] + [[name, effect] for name, effect in payload["drivers"]]
    table = Table(rows, colWidths=[95 * mm, 60 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(LINE)),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(LIGHT)]),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    elements.extend([table, Spacer(1, 8)])

    if payload.get("alert"):
        elements.extend([Paragraph(f"<b>Database check:</b> {payload['alert']}", body), Spacer(1, 8)])

    val = payload.get("valuation")
    if val:
        elements.append(Paragraph("Indicative valuation", h2))
        rows = [["Method", "Value (TND)"]]
        rows.extend([[name, f"{value:,.0f}"] for name, value in val["methods"].items()])
        rows.append(["Reconciled range", f"{val['low']:,.0f} - {val['high']:,.0f}"])
        vt = Table(rows, colWidths=[95 * mm, 60 * mm])
        vt.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(RED)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(LINE)),
                    ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor(NAVY)),
                    ("TEXTCOLOR", (0, -1), (-1, -1), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                ]
            )
        )
        elements.extend([vt, Spacer(1, 8)])

    elements.append(
        Paragraph(
            "<font size=8 color='#6B7280'>Decision-support output only. Final investment "
            "decisions remain with authorised CDC officers.</font>",
            body,
        )
    )
    doc.build(elements)
    buffer.seek(0)
    return buffer


def assessment_excel(payload: dict[str, Any]) -> io.BytesIO:
    from openpyxl import Workbook
    from openpyxl.styles import Border, Font, PatternFill, Side

    wb = Workbook()
    ws = wb.active
    ws.title = "Assessment"
    navy = PatternFill("solid", fgColor="272E5F")
    red = PatternFill("solid", fgColor="D10A11")
    white = Font(color="FFFFFF", bold=True)
    bold = Font(bold=True)
    thin = Border(
        left=Side(style="thin", color="D9DCE6"),
        right=Side(style="thin", color="D9DCE6"),
        top=Side(style="thin", color="D9DCE6"),
        bottom=Side(style="thin", color="D9DCE6"),
    )
    ws["A1"] = "CDC LAUNCHPAD - Startup Assessment"
    ws["A1"].font = Font(size=15, bold=True, color="272E5F")
    rows = [
        ("Startup", payload["name"]),
        ("Sector", payload["sector"]),
        ("Region", payload["region"]),
        ("Date", f"{dt.date.today():%Y-%m-%d}"),
        ("Score", f"{payload['score']:.1f}/100"),
        ("Model probability", f"{payload['proba']:.2f}"),
        ("Verdict", payload["verdict"]),
    ]
    row_idx = 3
    for key, value in rows:
        ws.cell(row_idx, 1, key).font = bold
        ws.cell(row_idx, 2, value)
        row_idx += 1
    row_idx += 1
    ws.cell(row_idx, 1, "Decision drivers").font = white
    ws.cell(row_idx, 1).fill = navy
    ws.cell(row_idx, 2, "Effect").font = white
    ws.cell(row_idx, 2).fill = navy
    row_idx += 1
    for name, effect in payload["drivers"]:
        ws.cell(row_idx, 1, name)
        ws.cell(row_idx, 2, effect)
        row_idx += 1

    val = payload.get("valuation")
    if val:
        row_idx += 1
        ws.cell(row_idx, 1, "Valuation method").font = white
        ws.cell(row_idx, 1).fill = red
        ws.cell(row_idx, 2, "Value (TND)").font = white
        ws.cell(row_idx, 2).fill = red
        row_idx += 1
        for method, value in val["methods"].items():
            ws.cell(row_idx, 1, method)
            ws.cell(row_idx, 2, round(value))
            ws.cell(row_idx, 2).number_format = "#,##0"
            row_idx += 1
        ws.cell(row_idx, 1, "Reconciled range").font = bold
        ws.cell(row_idx, 2, f"{val['low']:,.0f} - {val['high']:,.0f}")

    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 44
    for row in ws.iter_rows(min_row=3, max_row=max(row_idx, 3), max_col=2):
        for cell in row:
            cell.border = thin

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer


def portfolio_pdf(df: pd.DataFrame, summary: pd.DataFrame) -> io.BytesIO:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleCDC", parent=styles["Title"], textColor=colors.HexColor(NAVY))
    h2 = ParagraphStyle("H2CDC", parent=styles["Heading2"], textColor=colors.HexColor(NAVY))
    elements: list[Any] = []

    if os.path.exists(LOGO_FILE):
        try:
            elements.append(Image(LOGO_FILE, width=48 * mm, height=19 * mm))
            elements.append(Spacer(1, 8))
        except Exception:
            pass

    elements.extend(
        [
            Paragraph("CDC Portfolio Report", title),
            Paragraph(f"Generated {dt.date.today():%d %b %Y}", styles["BodyText"]),
            Spacer(1, 8),
        ]
    )
    kpis = [
        ["Portfolio startups", f"{len(df):,}"],
        ["Funded beneficiaries", f"{int(df['funded'].sum()):,}"],
        ["Funding rate", f"{df['funded'].mean() * 100:.1f}%"],
        ["Distinct sectors", f"{df['sector'].nunique():,}"],
    ]
    kt = Table(kpis, colWidths=[80 * mm, 60 * mm])
    kt.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(LINE)),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor(LIGHT)),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
            ]
        )
    )
    elements.extend([Paragraph("Headline indicators", h2), kt, Spacer(1, 10)])

    rows = [["Segment", "Startups", "Funded %", "Avg age"]]
    rows.extend(summary[["profile", "startups", "funded_rate", "avg_age"]].values.tolist())
    st = Table(rows, colWidths=[72 * mm, 30 * mm, 30 * mm, 30 * mm])
    st.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(LINE)),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(LIGHT)]),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    elements.extend([Paragraph("Portfolio segments", h2), st])
    doc.build(elements)
    buffer.seek(0)
    return buffer


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def _inject_css() -> None:
    import streamlit as st

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {LIGHT};
            color: {INK};
        }}
        h1, h2, h3 {{
            color: {NAVY};
            letter-spacing: 0;
        }}
        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1380px;
        }}
        .cdc-header {{
            display: flex;
            align-items: center;
            gap: 1.25rem;
            border-bottom: 4px solid {RED};
            padding: 0.7rem 0 1rem 0;
            margin-bottom: 1rem;
        }}
        .cdc-title {{
            font-size: clamp(1.65rem, 3vw, 2.6rem);
            line-height: 1.05;
            color: {NAVY};
            font-weight: 800;
            margin: 0;
        }}
        .cdc-subtitle {{
            color: {MUTED};
            margin: 0.2rem 0 0 0;
            font-size: 1rem;
        }}
        .cdc-logo {{
            max-width: 150px;
            min-width: 110px;
        }}
        .metric-card {{
            background: white;
            border: 1px solid {LINE};
            border-radius: 8px;
            padding: 0.85rem 1rem;
            min-height: 96px;
        }}
        .metric-label {{
            color: {MUTED};
            font-size: 0.82rem;
            margin-bottom: 0.25rem;
        }}
        .metric-value {{
            color: {NAVY};
            font-size: 1.55rem;
            line-height: 1.2;
            font-weight: 800;
        }}
        .alert-box {{
            background: white;
            border: 1px solid {LINE};
            border-left: 6px solid {BLUE};
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin: 0.5rem 0 1rem 0;
        }}
        .alert-title {{
            font-weight: 800;
            color: {INK};
            margin-bottom: 0.35rem;
        }}
        .small-muted {{
            color: {MUTED};
            font-size: 0.85rem;
        }}
        div[data-testid="stMetricValue"] {{
            color: {NAVY};
        }}
        .stButton>button, .stDownloadButton>button {{
            border-radius: 8px;
            border: 1px solid {NAVY};
            background: {NAVY};
            color: white;
            font-weight: 700;
        }}
        .stButton>button:hover, .stDownloadButton>button:hover {{
            border-color: {RED};
            background: {RED};
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metric(label: str, value: str, help_text: str = "") -> None:
    import streamlit as st

    help_html = f"<div class='small-muted'>{help_text}</div>" if help_text else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _alert(level: str, title: str, details: list[str]) -> None:
    import streamlit as st

    colors = {"green": GREEN, "amber": AMBER, "red": RED, "blue": BLUE, "none": MUTED}
    color = colors.get(level, BLUE)
    body = "<br>".join(details) if details else ""
    st.markdown(
        f"""
        <div class="alert-box" style="border-left-color:{color}">
            <div class="alert-title">{title}</div>
            <div class="small-muted">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _header(lang: str) -> None:
    import base64
    import streamlit as st

    logo_html = ""
    if os.path.exists(LOGO_FILE):
        with open(LOGO_FILE, "rb") as file:
            encoded = base64.b64encode(file.read()).decode("ascii")
        logo_html = f"<img class='cdc-logo' src='data:image/png;base64,{encoded}' alt='CDC logo'>"
    st.markdown(
        f"""
        <div class="cdc-header">
            {logo_html}
            <div>
                <div class="cdc-title">CDC LAUNCHPAD</div>
                <p class="cdc-subtitle">{t("subtitle", lang)}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _kpi_row(df: pd.DataFrame, bundle: ModelBundle) -> None:
    import streamlit as st

    m = bundle.metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric("Startups in base", f"{m['n_rows']:,}")
    with c2:
        _metric("Funded beneficiaries", f"{m['n_funded']:,}")
    with c3:
        _metric("Funding rate", f"{df['funded'].mean() * 100:.1f}%")
    with c4:
        _metric("Distinct sectors", f"{df['sector'].nunique():,}")


def run_app() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="CDC LAUNCHPAD",
        page_icon=LOGO_FILE if os.path.exists(LOGO_FILE) else None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    if not os.path.exists(DATA_FILE):
        st.error("Dataset not found. Place Startups_Tunisia_Master_v4.xlsx next to app.py.")
        st.write("Current folder:", SCRIPT_DIR)
        st.stop()

    session = st.session_state
    session.setdefault("lang", "EN")
    session.setdefault("auth", False)

    @st.cache_resource(show_spinner="Loading CDC portfolio and training the model...")
    def _load_all() -> tuple[pd.DataFrame, ModelBundle, pd.DataFrame, pd.DataFrame]:
        df = load_base()
        bundle = train_selection(df)
        segmented, summary = segment_portfolio(df)
        return df, bundle, segmented, summary

    df, bundle, segmented_df, segment_summary = _load_all()

    with st.sidebar:
        session.lang = st.radio(
            "Language",
            ["EN", "FR"],
            index=0 if session.lang == "EN" else 1,
            horizontal=True,
        )
        st.divider()
        st.caption("Access profile")
        if session.auth:
            st.success("Authenticated")
            if st.button(t("logout", session.lang), use_container_width=True):
                session.auth = False
                st.rerun()
        else:
            st.info("Restricted platform")

    lang = session.lang
    _header(lang)

    if not session.auth:
        st.subheader(t("access", lang))
        left, right = st.columns([1, 1])
        with left:
            st.text_input(t("email", lang), placeholder="name@cdc.tn")
            if st.button(t("request", lang), use_container_width=True):
                st.info(f"{t('sent', lang)} {ADMIN_EMAIL}. Demo code: {ACCESS_CODE}")
        with right:
            code = st.text_input(t("code", lang), type="password")
            if st.button(t("enter", lang), use_container_width=True):
                if code == ACCESS_CODE:
                    session.auth = True
                    st.rerun()
                else:
                    st.error(t("bad_code", lang))
        st.stop()

    tabs = st.tabs(
        [
            t("tab_ecosystem", lang),
            t("tab_portfolio", lang),
            t("tab_assessment", lang),
            t("tab_valuation", lang),
            t("tab_learning", lang),
            t("tab_reports", lang),
        ]
    )

    with tabs[0]:
        _kpi_row(df, bundle)
        st.markdown("### Tunisian startup ecosystem")
        left, right = st.columns([1.25, 1])
        with left:
            top = df["sector"].value_counts().head(12).sort_values()
            st.bar_chart(top, color=RED)
        with right:
            st.markdown("#### Live ecosystem watch")
            for item in fetch_news():
                if item["link"]:
                    st.markdown(f"- [{item['title']}]({item['link']})")
                else:
                    st.markdown(f"- {item['title']}")

    with tabs[1]:
        st.markdown("### Portfolio dashboard")
        _kpi_row(df, bundle)
        left, right = st.columns(2)
        with left:
            sector_counts = df["sector"].value_counts().head(15)
            st.dataframe(
                sector_counts.rename_axis("Sector").reset_index(name="Startups"),
                use_container_width=True,
                hide_index=True,
            )
        with right:
            years = pd.to_numeric(df["founding_year"], errors="coerce").dropna().astype(int)
            st.line_chart(years.value_counts().sort_index().tail(20), color=NAVY)
        st.markdown("### Segments")
        st.dataframe(
            segment_summary.rename(
                columns={
                    "profile": "Segment",
                    "startups": "Startups",
                    "funded_rate": "Funded %",
                    "avg_age": "Avg age",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("### Searchable portfolio")
        display_cols = [
            col
            for col in ["Nom", "Secteur", "Region", "founding_year", "funded", "profile"]
            if col in segmented_df.columns
        ]
        st.dataframe(segmented_df[display_cols].head(300), use_container_width=True, hide_index=True)

    with tabs[2]:
        st.markdown("### Startup assessment")
        query = st.text_input("Database check - startup name or identifier")
        if query:
            alert = lookup_startup(df, query)
            _alert(alert["level"], alert["title"], alert["details"])

        with st.form("assessment_form"):
            c1, c2, c3 = st.columns(3)
            name = c1.text_input("Startup name", value=query or "Demo Health Tunisia")
            sector = c2.selectbox("Sector", sorted(df["sector"].dropna().astype(str).unique()))
            region = c3.selectbox("Region", REGIONS_TN)

            c4, c5, c6 = st.columns(3)
            year = c4.number_input("Founding year", min_value=2000, max_value=ANALYSIS_YEAR, value=2021)
            founders = c5.number_input("Number of founders", min_value=1, max_value=12, value=2)
            stage = c6.selectbox("Product stage", ["Idea", "MVP", "Revenue-generating"], index=1)

            c7, c8, c9 = st.columns(3)
            labelled = c7.checkbox("Startup Act label", value=True)
            has_email = c8.checkbox("Verified contact", value=True)
            has_web = c9.checkbox("Website", value=True)

            st.markdown("#### Valuation inputs")
            v1, v2, v3 = st.columns(3)
            revenue = v1.number_input(
                "Latest annual revenue (TND)",
                min_value=0,
                max_value=50_000_000,
                value=0,
                step=50_000,
            )
            growth = v2.slider("Expected annual growth", 0.0, 1.5, 0.40, 0.05)
            competition = v3.slider("Competitive pressure", 0.0, 1.0, 0.50, 0.05)
            v4, v5, v6 = st.columns(3)
            team = v4.slider("Team strength", 0.0, 1.0, 0.70, 0.05)
            market = v5.slider("Market opportunity", 0.0, 1.0, 0.65, 0.05)
            product = v6.slider("Product maturity", 0.0, 1.0, 0.60, 0.05)
            submitted = st.form_submit_button("Run assessment", use_container_width=True)

        if submitted:
            result = assess_one(
                bundle,
                {
                    "sector": sector,
                    "founding_year": year,
                    "n_founders": founders,
                    "is_labelled": labelled,
                    "has_email": has_email,
                    "has_web": has_web,
                },
            )
            valuation = valuation_engine(
                {
                    "stage": ["Idea", "MVP", "Revenue-generating"].index(stage),
                    "team": team,
                    "market": market,
                    "product": product,
                    "competition": competition,
                    "revenue_tnd": revenue,
                    "growth": growth,
                }
            )
            db_alert = lookup_startup(df, name)["title"]
            payload = {
                "name": name,
                "sector": sector,
                "region": region,
                **result,
                "valuation": valuation,
                "alert": db_alert,
            }
            session["last_assessment"] = payload

        payload = session.get("last_assessment")
        if payload:
            tone_color = {"ok": GREEN, "warn": "#E8A33D", "bad": RED}[payload["tone"]]
            c1, c2, c3 = st.columns([1, 1.2, 1.2])
            with c1:
                st.metric("Funding-likelihood score", f"{payload['score']:.1f}/100")
                st.progress(min(1.0, payload["proba"]))
            with c2:
                st.markdown(
                    f"<h4 style='color:{tone_color}; margin-top:0'>{payload['verdict']}</h4>",
                    unsafe_allow_html=True,
                )
                st.caption(f"Model probability: {payload['proba']:.2f}")
            with c3:
                val = payload["valuation"]
                st.metric("Valuation range", f"{val['low']/1e6:.2f}M - {val['high']/1e6:.2f}M TND")
                st.caption(f"Median: {val['mid']/1e6:.2f}M TND")

            drivers = pd.DataFrame(payload["drivers"], columns=["Factor", "Effect"])
            st.dataframe(drivers, use_container_width=True, hide_index=True)

            val_rows = pd.DataFrame(
                {
                    "Method": list(payload["valuation"]["methods"].keys()),
                    "Value (TND)": [
                        f"{value:,.0f}" for value in payload["valuation"]["methods"].values()
                    ],
                }
            )
            st.dataframe(val_rows, use_container_width=True, hide_index=True)
            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "Download assessment PDF",
                    assessment_pdf(payload),
                    file_name=f"Assessment_{payload['name'].replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            with d2:
                st.download_button(
                    "Download assessment Excel",
                    assessment_excel(payload),
                    file_name=f"Assessment_{payload['name'].replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

    with tabs[3]:
        st.markdown("### Standalone valuation")
        c1, c2, c3 = st.columns(3)
        v_stage = c1.selectbox("Stage", ["Idea", "MVP", "Revenue-generating"], index=1)
        v_revenue = c2.number_input(
            "Annual revenue (TND)",
            min_value=0,
            max_value=50_000_000,
            value=0,
            step=50_000,
            key="standalone_revenue",
        )
        v_growth = c3.slider("Growth", 0.0, 1.5, 0.40, 0.05, key="standalone_growth")
        c4, c5, c6, c7 = st.columns(4)
        v_team = c4.slider("Team", 0.0, 1.0, 0.70, 0.05, key="standalone_team")
        v_market = c5.slider("Market", 0.0, 1.0, 0.65, 0.05, key="standalone_market")
        v_product = c6.slider("Product", 0.0, 1.0, 0.60, 0.05, key="standalone_product")
        v_comp = c7.slider("Competition", 0.0, 1.0, 0.50, 0.05, key="standalone_comp")
        valuation = valuation_engine(
            {
                "stage": ["Idea", "MVP", "Revenue-generating"].index(v_stage),
                "revenue_tnd": v_revenue,
                "growth": v_growth,
                "team": v_team,
                "market": v_market,
                "product": v_product,
                "competition": v_comp,
            }
        )
        st.metric(
            "Reconciled range",
            f"{valuation['low']:,.0f} - {valuation['high']:,.0f} TND",
            f"Median {valuation['mid']:,.0f} TND",
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Method": list(valuation["methods"].keys()),
                    "Value (TND)": [f"{value:,.0f}" for value in valuation["methods"].values()],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with tabs[4]:
        st.markdown("### Model performance")
        m = bundle.metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Engine", m["engine"])
        c2.metric("ROC-AUC", "-" if m["roc_auc"] is None else m["roc_auc"])
        c3.metric("F1", "-" if m["f1"] is None else m["f1"])
        c4.metric("Accuracy", "-" if m["accuracy"] is None else m["accuracy"])

        st.markdown("### Add labelled outcome")
        with st.form("learning_form"):
            l1, l2, l3 = st.columns(3)
            new_name = l1.text_input("Name", "NewCo Tunisia")
            new_sector = l2.selectbox("Sector", sorted(df["sector"].dropna().astype(str).unique()), key="learn_sector")
            new_year = l3.number_input("Founding year", 2000, ANALYSIS_YEAR, 2022, key="learn_year")
            l4, l5, l6 = st.columns(3)
            new_founders = l4.number_input("Founders", 1, 12, 3, key="learn_founders")
            outcome = l5.selectbox("Outcome", ["funded", "not funded"])
            new_labelled = l6.checkbox("Startup Act label", True, key="learn_label")
            append = st.form_submit_button("Append and retrain", use_container_width=True)
        if append:
            total = append_record(
                {
                    "name": new_name,
                    "sector": new_sector,
                    "year": new_year,
                    "founders": new_founders,
                    "labelled": new_labelled,
                    "funded": 1 if outcome == "funded" else 0,
                }
            )
            st.cache_resource.clear()
            st.success(f"Stored {total} learning rows. Reloading model...")
            st.rerun()

    with tabs[5]:
        st.markdown("### Reports")
        st.download_button(
            "Download portfolio PDF",
            portfolio_pdf(df, segment_summary),
            file_name="CDC_Portfolio_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        if session.get("last_assessment"):
            st.download_button(
                "Download last assessment PDF",
                assessment_pdf(session["last_assessment"]),
                file_name="Last_Assessment.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


if __name__ == "__main__":
    run_app()
