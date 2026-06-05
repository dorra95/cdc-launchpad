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
LOGO_ANIMATION = os.path.join(SCRIPT_DIR, "cdc_animation.mp4")
STORE_FILE = os.path.join(SCRIPT_DIR, "cdc_learning_store.csv")
ACCESS_LOG = os.path.join(SCRIPT_DIR, "cdc_access_log.csv")

ADMIN_EMAIL = "dorra.fadhloun@msb.tn"
ANALYSIS_YEAR = 2026
RANDOM_STATE = 42
ACCESS_CODE_TTL_MIN = 20

NAVY = "#272E5F"
RED = "#D10A11"
GOLD = "#C9A227"
TEAL = "#0FB5A6"
INK = "#1F2937"
MUTED = "#6B7280"
LINE = "#D9DCE6"
LIGHT = "#F4F5F9"
GREEN = "#227A4A"
AMBER = "#B7791F"
BLUE = "#2563EB"
VIOLET = "#7C3AED"
ROSE = "#E11D48"

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
    "tab_ecosystem": {"EN": "At a Glance", "FR": "A la une"},
    "tab_programs": {"EN": "Our Programs", "FR": "Nos Programmes"},
    "tab_portfolio": {"EN": "Portfolio", "FR": "Portefeuille"},
    "tab_assessment": {"EN": "Assessment", "FR": "Evaluation"},
    "tab_news": {"EN": "Newsroom", "FR": "Veille"},
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


# ---------------------------------------------------------------------------
# VAIR Greentech committee scoring grid (7 axes, 0-5 per criterion)
# ---------------------------------------------------------------------------
SCORING_GRID: list[dict[str, Any]] = [
    {
        "axis": "Innovation et Proposition de valeur",
        "guidance": (
            "Mesure la nouveauté et l'originalité de la solution ainsi que sa pertinence "
            "pour l'utilisateur final. Un score élevé correspond à une solution "
            "différenciante, répondant à un problème réel, avec un potentiel d'adoption."
        ),
        "criteria": [
            {
                "name": "Originalité de la solution",
                "angle": "Dans quelle mesure la solution se différencie-t-elle des pratiques/technologies existantes ?",
                "fields": "En quoi votre solution est-elle innovante ? / Pitch Deck / Vidéo YouTube",
                "anchors": [
                    "Absence totale d'innovation",
                    "Copie d'existant, aucune différenciation",
                    "Innovation mineure, amélioration marginale",
                    "Innovation modérée, différenciateur clair",
                    "Innovation forte, rupture partielle",
                    "Innovation de rupture, unique sur le marché",
                ],
            },
            {
                "name": "Pertinence de la valeur ajoutée pour l'utilisateur",
                "angle": "La solution répond-elle à un vrai besoin et crée-t-elle une utilité claire pour le client final ?",
                "fields": "Décrivez le problème que vous adressez / Décrivez votre projet / Pitch Deck",
                "anchors": [
                    "Aucun besoin identifié, hors sujet",
                    "Proposition confuse, besoin non démontré",
                    "Valeur ajoutée faible ou partiellement pertinente",
                    "Répond à un besoin identifié mais sans validation terrain",
                    "Retours initiaux positifs / intérêt exprimé",
                    "Proposition démontrée comme indispensable (painkiller)",
                ],
            },
        ],
    },
    {
        "axis": "Adéquation au marché ciblé",
        "guidance": (
            "Évalue la clarté et la crédibilité de la définition du marché, les preuves "
            "d'intérêt ou de traction, ainsi que la stratégie d'accès au marché."
        ),
        "criteria": [
            {
                "name": "Clarté et pertinence de la définition du marché",
                "angle": "Le projet a-t-il identifié un marché réel et pertinent avec des données crédibles ?",
                "fields": "Quelle est la taille de votre marché cible ? / Pays ciblés",
                "anchors": [
                    "Aucun marché identifié",
                    "Marché très flou, hypothèses non étayées",
                    "Marché identifié mais peu documenté",
                    "Marché défini avec segments clairs, premières estimations",
                    "Marché bien documenté avec données fiables",
                    "Marché clair, documenté, solide, comparatif sectoriel",
                ],
            },
            {
                "name": "Validation et accessibilité du marché",
                "angle": "La solution a-t-elle des preuves concrètes d'intérêt ou d'accès au marché ?",
                "fields": "Preuves de traction / Lettres de références ou contrats / LOI",
                "anchors": [
                    "Aucun signe de demande",
                    "Hypothèse sans preuve",
                    "Premiers signaux d'intérêt très limités",
                    "Premiers retours clients ou LOI",
                    "Validation par pilotes, partenaires, premiers contrats",
                    "Forte traction, marché validé avec adoption tangible",
                ],
            },
            {
                "name": "Crédibilité de la stratégie d'accès au marché",
                "angle": "La stratégie pour pénétrer le marché est-elle réaliste et cohérente ?",
                "fields": "Part de marché cible / Stratégie de croissance 3 ans avec KPIs",
                "anchors": [
                    "Aucune stratégie décrite",
                    "Stratégie vague, irréaliste",
                    "Stratégie partielle, lacunes importantes",
                    "Stratégie structurée, premiers KPIs crédibles",
                    "Stratégie claire, chiffrée, cohérente",
                    "Stratégie solide, démontrée comme réaliste",
                ],
            },
            {
                "name": "Partenaires stratégiques (optionnel)",
                "angle": "Le projet a-t-il embarqué des partenaires clés qui renforcent son accès marché ?",
                "fields": "Partenaires stratégiques / Lettres d'intention / LOI",
                "anchors": [
                    "Aucun partenaire",
                    "Partenaires annoncés mais non crédibles",
                    "Partenaires mineurs, rôle peu clair",
                    "Partenaires identifiés avec premiers engagements",
                    "Partenaires solides et actifs",
                    "Partenariats stratégiques structurés et confirmés",
                ],
            },
        ],
    },
    {
        "axis": "Qualité et complémentarité de l'équipe",
        "guidance": (
            "L'équipe est au cœur de la réussite. Mesure les compétences techniques et "
            "business, la capacité à exécuter, ainsi que l'historique entrepreneurial."
        ),
        "criteria": [
            {
                "name": "Compétences et expérience",
                "angle": "Les fondateurs possèdent-ils les compétences techniques et/ou business nécessaires ?",
                "fields": "Bio courte / Background académique / Années d'expérience",
                "anchors": [
                    "Aucune compétence pertinente",
                    "Compétences très limitées",
                    "Compétences présentes mais lacunes critiques",
                    "Compétences clés présentes, premières expériences sectorielles",
                    "Forte expertise tech ou business, expérience probante",
                    "Expertise solide tech et business, expérience confirmée dans le secteur",
                ],
            },
            {
                "name": "Capacité d'exécution et organisation",
                "angle": "Les fondateurs ont-ils une organisation claire et la capacité à exécuter le projet ?",
                "fields": "Pitch Deck / Moyens techniques et humains / Plan de développement",
                "anchors": [
                    "Aucun signe de capacité",
                    "Organisation très faible",
                    "Organisation embryonnaire",
                    "Organisation fonctionnelle, exécution crédible",
                    "Exécution solide, plan clair, responsabilités assumées",
                    "Exécution excellente, gouvernance lisible, livrables tenus",
                ],
            },
            {
                "name": "Historique entrepreneurial & PI",
                "angle": "L'historique (projets, brevets, accompagnements, prix) renforce-t-il la crédibilité ?",
                "fields": "Programmes d'accompagnement / Brevets / Références / LOI",
                "anchors": [
                    "Aucun historique",
                    "Historique très faible, non pertinent",
                    "Historique limité (petits projets/initiatives)",
                    "Projets antérieurs pertinents ou un brevet/prix",
                    "Plusieurs projets pertinents et/ou brevets/prix significatifs",
                    "Track record fort (lancements, traction, PI stratégique, prix reconnus)",
                ],
            },
        ],
    },
    {
        "axis": "Évaluation du PoC",
        "guidance": (
            "Évalue la crédibilité technique du projet et la capacité à réaliser un PoC : "
            "clarté de la description, préparation TRL, ressources, vision d'industrialisation."
        ),
        "criteria": [
            {
                "name": "Clarté de la description technique",
                "angle": "Le projet présente-t-il une description claire, compréhensible et crédible ?",
                "fields": "Description du projet / Pitch Deck / Vidéo",
                "anchors": [
                    "Aucune description",
                    "Description floue ou incohérente",
                    "Description faible",
                    "Description claire, premiers éléments",
                    "Description solide et cohérente",
                    "Description très claire, illustrée",
                ],
            },
            {
                "name": "Préparation technique pour le PoC",
                "angle": "Le projet montre-t-il une préparation crédible pour progresser (TRL 1-3) ?",
                "fields": "Stade actuel / Avancement / Vidéo démo",
                "anchors": [
                    "Aucun signe de préparation",
                    "Préparation limitée",
                    "Préparation insuffisante",
                    "Préparation crédible, premiers éléments",
                    "Préparation solide et alignée",
                    "Préparation excellente, plan clair et démonstrations",
                ],
            },
            {
                "name": "Ressources techniques mobilisées",
                "angle": "Les moyens humains/techniques sont-ils adaptés pour réaliser le PoC ?",
                "fields": "Moyens techniques et humains / Plan PoC",
                "anchors": [
                    "Aucun moyen",
                    "Moyens faibles",
                    "Moyens insuffisants",
                    "Moyens adaptés",
                    "Moyens solides",
                    "Moyens complets, expertise claire",
                ],
            },
            {
                "name": "Vision de passage à l'échelle",
                "angle": "Le projet a-t-il anticipé l'industrialisation après le PoC ?",
                "fields": "Plan de développement / Calendrier & potentiel de passage à l'échelle",
                "anchors": [
                    "Aucune vision",
                    "Vision très vague",
                    "Vision limitée",
                    "Vision partielle mais crédible",
                    "Vision claire et cohérente",
                    "Vision solide, structurée, alignée sur croissance",
                ],
            },
        ],
    },
    {
        "axis": "Cohérence TRL, budget et remboursement",
        "guidance": (
            "Vérifie si le budget demandé est adapté au TRL et au plan, la capacité de "
            "gestion financière, et la crédibilité de la trajectoire de remboursement."
        ),
        "criteria": [
            {
                "name": "Maturité de l'innovation (TRL atteint)",
                "angle": "Le stade de développement est-il clair et cohérent avec la demande VAIR (PoC, TRL 1-3) ?",
                "fields": "Stade TRL / Avancement actuel / Vidéo démo",
                "anchors": [
                    "Stade incohérent ou non déclaré",
                    "Stade très flou, pas justifié",
                    "Stade décrit mais peu crédible",
                    "Stade clair, premiers éléments tangibles",
                    "Stade bien défini avec preuves (proto, tests, vidéos)",
                    "Stade parfaitement défini avec validations solides",
                ],
            },
            {
                "name": "Pertinence et réalisme du budget",
                "angle": "Le budget demandé est-il adapté au TRL et au plan de développement annoncé ?",
                "fields": "Tableau budget détaillé du PoC / Pitch Deck / Calendrier",
                "anchors": [
                    "Budget absent ou incohérent",
                    "Budget approximatif, sans lien avec TRL",
                    "Budget décrit mais incohérent sur plusieurs postes",
                    "Budget globalement aligné avec TRL, incohérences mineures",
                    "Budget structuré, cohérent et bien justifié",
                    "Budget très solide, aligné TRL et stratégie, justification poste par poste",
                ],
            },
            {
                "name": "Capacité (ou plan) de gestion financière",
                "angle": "L'équipe démontre-t-elle une capacité à gérer et reporter les fonds, même sans historique ?",
                "fields": "Tableau budget / États financiers / Levées précédentes",
                "anchors": [
                    "Aucun signe de capacité ni plan",
                    "Plan flou ou irréaliste",
                    "Plan basique, plusieurs incohérences",
                    "Plan structuré, reporting prévu, quelques limites",
                    "Plan détaillé, reporting clair, alignement avec besoins du PoC",
                    "Plan de gestion et reporting très solide, prêt à l'exécution",
                ],
            },
            {
                "name": "Crédibilité de la trajectoire de revenus / remboursement",
                "angle": "Le modèle économique permet-il d'anticiper la capacité à rembourser l'avance ?",
                "fields": "Modèle de revenus / Business Model Canvas / Stratégie 3 ans / Traction",
                "anchors": [
                    "Aucun modèle économique, aucune piste",
                    "Modèle théorique, irréaliste",
                    "Modèle décrit mais flou ou trop optimiste",
                    "Modèle crédible, premiers signaux de traction possibles",
                    "Modèle clair, pipeline commercial ou partenariats en cours",
                    "Modèle très crédible, pipeline solide, forte probabilité de remboursement",
                ],
            },
        ],
    },
    {
        "axis": "Impact environnemental et social",
        "guidance": (
            "Mesure le potentiel d'impact environnemental, la durabilité, la capacité à "
            "mesurer l'impact, ainsi que les aspects sociaux (genre, emplois qualifiés)."
        ),
        "criteria": [
            {
                "name": "Potentiel d'impact environnemental",
                "angle": "Quel est le niveau d'impact environnemental positif attendu ?",
                "fields": "Type d'impact environnemental / Défis environnementaux adressés",
                "anchors": [
                    "Aucun impact prévisible",
                    "Impact hypothétique, non démontré",
                    "Impact potentiel mais limité",
                    "Premiers résultats qualitatifs attendus",
                    "Impact significatif démontré sur un périmètre restreint",
                    "Impact significatif démontré sur une large échelle",
                ],
            },
            {
                "name": "Durabilité de la solution",
                "angle": "La solution est-elle pensée pour être durable à long terme ?",
                "fields": "Contribution aux ODD / Pitch Deck",
                "anchors": [
                    "Aucune considération de durabilité",
                    "Déclaration d'intention sans actions",
                    "Premières initiatives mises en place",
                    "Démarches structurées et mesurées",
                    "Stratégie de durabilité claire, suivie",
                    "Durabilité au cœur du projet, rôle de leader",
                ],
            },
            {
                "name": "Mesure et suivi de l'impact",
                "angle": "L'entreprise mesure-t-elle son impact environnemental/social ?",
                "fields": "Modalités de mesure d'impact ODD / Niveau de maîtrise",
                "anchors": [
                    "Aucun suivi ni mesure",
                    "Intention déclarée mais pas d'outil",
                    "Données collectées de façon limitée",
                    "Données partielles + reporting basique",
                    "Système clair de mesure avec objectifs",
                    "Mesure systématique, reporting complet, labels ou certifications",
                ],
            },
            {
                "name": "Prise en compte du genre",
                "angle": "Le projet prend-il en compte l'égalité femmes/hommes ?",
                "fields": "Actions pour favoriser l'égalité / Nombre de femmes employées",
                "anchors": [
                    "Aucune prise en compte",
                    "Déclaration d'intention",
                    "Initiatives ponctuelles",
                    "Initiatives structurées mais limitées",
                    "Politiques claires et suivies",
                    "Égalité intégrée dans la culture et reconnue",
                ],
            },
            {
                "name": "Création d'emplois",
                "angle": "Le projet a-t-il un potentiel de création d'emplois verts/qualifiés ?",
                "fields": "Nombre d'employés / Création d'emplois attendue",
                "anchors": [
                    "Aucun potentiel",
                    "Très faible, non qualifié",
                    "Potentiel modéré, peu qualifié",
                    "Potentiel modéré, qualifié",
                    "Potentiel important et qualifié",
                    "Potentiel important et très qualifié",
                ],
            },
        ],
    },
    {
        "axis": "Risques et crédibilité globale",
        "guidance": (
            "Évalue la capacité du projet à anticiper ses risques et à proposer des "
            "stratégies de mitigation crédibles, ainsi que la vision post-PoC."
        ),
        "criteria": [
            {
                "name": "Identification et pertinence des risques",
                "angle": "Les risques principaux (techniques, marché, réglementaires, humains) sont-ils identifiés ?",
                "fields": "Risques auxquels l'activité est confrontée",
                "anchors": [
                    "Aucun risque",
                    "Risques superficiels",
                    "Risques listés sans analyse",
                    "Risques identifiés, analyse partielle",
                    "Risques bien identifiés et analysés",
                    "Analyse complète et priorisée",
                ],
            },
            {
                "name": "Stratégies de mitigation",
                "angle": "Le projet prévoit-il des moyens concrets pour gérer ces risques ?",
                "fields": "Moyens de contournement / Pitch Deck",
                "anchors": [
                    "Aucun plan",
                    "Plan vague",
                    "Plan partiel",
                    "Plan crédible mais incomplet",
                    "Plan structuré et réaliste",
                    "Plan solide, proactif, crédible",
                ],
            },
            {
                "name": "Vision de passage à l'échelle post-PoC",
                "angle": "L'entreprise a-t-elle anticipé son développement après VAIR ?",
                "fields": "Plan PoC / Calendrier de réalisation",
                "anchors": [
                    "Aucune vision",
                    "Vision vague",
                    "Vision limitée",
                    "Vision crédible mais partielle",
                    "Vision claire et cohérente",
                    "Vision solide, alignée sur croissance",
                ],
            },
        ],
    },
]

# ---------------------------------------------------------------------------
# Email-code access flow
# ---------------------------------------------------------------------------
import csv
import hashlib
import secrets
import smtplib
from email.mime.text import MIMEText


def _hash_code(email: str, code: str) -> str:
    return hashlib.sha256(f"{email.strip().lower()}|{code}".encode()).hexdigest()


def _smtp_setting(key: str) -> str:
    """Read SMTP setting from Streamlit secrets first, then env, then empty."""
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            value = st.secrets.get(key, "") if hasattr(st.secrets, "get") else ""
            if value:
                return str(value)
    except Exception:
        pass
    return os.environ.get(key, "")


def _try_send_email(to_addr: str, code: str) -> tuple[bool, str]:
    """Attempt SMTP send; fall back to admin log + stdout if creds absent."""
    host = _smtp_setting("CDC_SMTP_HOST")
    user = _smtp_setting("CDC_SMTP_USER")
    pwd = _smtp_setting("CDC_SMTP_PASS")
    sender = _smtp_setting("CDC_SMTP_FROM") or user or ADMIN_EMAIL
    body = (
        "Bonjour,\n\nVotre code d'acces a la plateforme CDC LAUNCHPAD est : "
        f"{code}\n\nCe code expire dans {ACCESS_CODE_TTL_MIN} minutes.\n\n"
        "Si vous n'avez pas demande d'acces, ignorez ce message.\n\n— CDC Tunisie"
    )
    if host and user and pwd:
        try:
            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = "CDC LAUNCHPAD - votre code d'acces"
            msg["From"] = sender
            msg["To"] = to_addr
            with smtplib.SMTP_SSL(host, 465, timeout=10) as smtp:
                smtp.login(user, pwd)
                smtp.sendmail(sender, [to_addr], msg.as_string())
            return True, "email"
        except Exception as exc:
            print(f"[CDC LAUNCHPAD] SMTP error: {exc!r}", flush=True)
    # Fallback: emit the code to stdout (visible in Streamlit Cloud logs) and write to a local file.
    banner = (
        "================================================================\n"
        f"[CDC LAUNCHPAD] DEV-MODE ACCESS CODE (no SMTP configured)\n"
        f"  email   : {to_addr}\n"
        f"  code    : {code}\n"
        f"  expires : {ACCESS_CODE_TTL_MIN} min from now\n"
        "================================================================"
    )
    print(banner, flush=True)
    try:
        new = not os.path.exists(ACCESS_LOG)
        with open(ACCESS_LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timestamp", "email", "code"])
            w.writerow([dt.datetime.utcnow().isoformat(), to_addr, code])
    except Exception:
        pass
    return True, "admin_log"


def issue_access_code(email: str) -> dict[str, Any]:
    code = f"{secrets.randbelow(1_000_000):06d}"
    ok, channel = _try_send_email(email, code)
    return {
        "ok": ok,
        "channel": channel,
        "hash": _hash_code(email, code),
        "expires": dt.datetime.utcnow() + dt.timedelta(minutes=ACCESS_CODE_TTL_MIN),
    }


def verify_access_code(email: str, code: str, issued: dict[str, Any]) -> bool:
    if not issued or not email or not code:
        return False
    if dt.datetime.utcnow() > issued.get("expires", dt.datetime.utcnow()):
        return False
    return secrets.compare_digest(_hash_code(email, code), issued.get("hash", ""))


# ---------------------------------------------------------------------------
# Tunisia ecosystem - curated landing references for the "A la une" panel
# ---------------------------------------------------------------------------
ECOSYSTEM_ARTICLES: list[dict[str, str]] = [
    {
        "title": "Startup Act Tunisie - label, fiscalite et financement",
        "source": "startup.gov.tn",
        "url": "https://startup.gov.tn/",
        "tag": "Politique publique",
        "summary": (
            "Le Startup Act tunisien (loi 2018-20) ouvre un guichet unique pour "
            "labelliser les startups innovantes : exoneration fiscale, conge "
            "creation pour salaries, garantie de capital et acces simplifie aux "
            "devises. Plus de 1100 startups labellisees depuis 2019."
        ),
        "color": "navy",
    },
    {
        "title": "Anava - fund of funds tunisien pour le venture capital",
        "source": "smartcapital.tn",
        "url": "https://smartcapital.tn/",
        "tag": "Capital risque",
        "summary": (
            "Anava, gere par Smart Capital, mobilise jusqu'a 200 MDT pour "
            "investir dans une dizaine de fonds VC adressant les startups "
            "tunisiennes - de l'amorcage au capital developpement, avec un "
            "co-investissement de bailleurs internationaux (Banque Mondiale, "
            "AfDB, KfW)."
        ),
        "color": "red",
    },
    {
        "title": "Caisse des Depots et Consignations - moteur du financement long",
        "source": "cdc.tn",
        "url": "https://www.cdc.tn/",
        "tag": "Financement public",
        "summary": (
            "La CDC Tunisie deploie des programmes d'investissement long terme : "
            "ANAVA, VAIR (avances remboursables pour Greentech), participations "
            "directes et programmes regionaux. Mission : densifier le marche "
            "tunisien du capital innovation."
        ),
        "color": "gold",
    },
    {
        "title": "VAIR Greentech - avances pour PoC bas carbone",
        "source": "smartcapital.tn / cdc.tn",
        "url": "https://smartcapital.tn/",
        "tag": "Greentech",
        "summary": (
            "Le programme VAIR finance le passage du concept au PoC (TRL 1-3) "
            "des startups greentech tunisiennes. Avance remboursable sur "
            "trajectoire de revenus, evaluation par comite multi-criteres "
            "(innovation, marche, equipe, impact ESG)."
        ),
        "color": "teal",
    },
    {
        "title": "Flat6Labs Tunis - accelerateur seed",
        "source": "flat6labs.com",
        "url": "https://www.flat6labs.com/",
        "tag": "Acceleration",
        "summary": (
            "Cohortes biannuelles de 8 a 10 startups tunisiennes, ticket initial "
            "et programme de 4 mois de mentorat. Plus de 80 startups passees par "
            "le programme depuis 2016, focus tech et fintech."
        ),
        "color": "violet",
    },
    {
        "title": "216 Capital - VC tunisien actif sur seed et serie A",
        "source": "216capital.com",
        "url": "https://216capital.com/",
        "tag": "Capital risque",
        "summary": (
            "Fonds early stage base a Tunis, investit en seed et pre-serie A "
            "dans des startups tunisiennes et nord-africaines a portee MENA. "
            "Tickets typiquement entre 50K et 500K USD."
        ),
        "color": "rose",
    },
    {
        "title": "Wamda - veille ecosysteme MENA et Tunisie",
        "source": "wamda.com",
        "url": "https://www.wamda.com/tags/tunisia",
        "tag": "Media",
        "summary": (
            "Plateforme de reference pour l'actualite startup MENA - couverture "
            "reguliere des levees de fonds, sorties et programmes tunisiens. "
            "Outil de veille pour identifier rapidement les tendances regionales."
        ),
        "color": "blue",
    },
    {
        "title": "Africa Report - lecture macro de l'innovation tunisienne",
        "source": "theafricareport.com",
        "url": "https://www.theafricareport.com/tag/tunisia/",
        "tag": "Media",
        "summary": (
            "Analyses geopolitiques et economiques sur la Tunisie, incluant la "
            "dynamique entrepreneuriale, le climat des affaires et les annonces "
            "des bailleurs. Source utile pour benchmarker la regulation."
        ),
        "color": "amber",
    },
]

IMPACT_KPI_DEFINITIONS: list[dict[str, str]] = [
    {"key": "total", "label": "Startups suivies", "icon": "RC", "tone": "navy"},
    {"key": "funded", "label": "Financees au moins une fois", "icon": "$", "tone": "red"},
    {"key": "labelled", "label": "Labellisees Startup Act", "icon": "L", "tone": "gold"},
    {"key": "sectors", "label": "Secteurs couverts", "icon": "S", "tone": "teal"},
    {"key": "women", "label": "Equipes feminines", "icon": "F", "tone": "rose"},
    {"key": "recent", "label": "Crees apres 2020", "icon": "N", "tone": "violet"},
]


def compute_impact_kpis(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    total = int(len(df))
    funded = int(df["funded"].sum()) if "funded" in df.columns else 0
    labelled = int(df.get("is_labelled", pd.Series([])).sum()) if "is_labelled" in df.columns else 0
    sectors = int(df["sector"].dropna().astype(str).nunique()) if "sector" in df.columns else 0
    women = 0
    if "Nom des fondateurs" in df.columns:
        text = df["Nom des fondateurs"].astype(str).str.lower()
        women = int(text.str.contains(r"\b(ms\.?|mme|miss|madame|mlle)\b", regex=True, na=False).sum())
    recent = 0
    if "founding_year" in df.columns:
        years = pd.to_numeric(df["founding_year"], errors="coerce")
        recent = int((years >= 2020).sum())
    funded_rate = (funded / total * 100) if total else 0
    return {
        "total": {"value": total, "secondary": "dans la base CDC"},
        "funded": {"value": funded, "secondary": f"{funded_rate:.0f}% du portefeuille"},
        "labelled": {"value": labelled, "secondary": "Startup Act"},
        "sectors": {"value": sectors, "secondary": "verticales"},
        "women": {"value": women, "secondary": "founders feminins"},
        "recent": {"value": recent, "secondary": "post-2020"},
    }


# ---------------------------------------------------------------------------
# AI rationale engines - turn numbers into argued justifications
# ---------------------------------------------------------------------------
def _anchor_for(axis: str, criterion: str, score: int) -> str:
    for block in SCORING_GRID:
        if block["axis"] == axis:
            for crit in block["criteria"]:
                if crit["name"] == criterion:
                    return crit["anchors"][max(0, min(5, score))]
    return ""


def axis_rationale(axis: str, criterion_scores: dict[str, int]) -> dict[str, Any]:
    """Generate an argued rationale per axis citing strongest/weakest criteria."""
    pairs = [(c, int(s)) for c, s in criterion_scores.items()]
    if not pairs:
        return {"summary": "Aucune donnee.", "strengths": [], "gaps": [], "advice": ""}
    pairs.sort(key=lambda x: x[1], reverse=True)
    strongest = pairs[0]
    weakest = pairs[-1]
    avg = sum(s for _, s in pairs) / len(pairs)
    if avg >= 4:
        verdict = "atout structurant"
    elif avg >= 3:
        verdict = "axe solide mais perfectible"
    elif avg >= 2:
        verdict = "axe a renforcer"
    else:
        verdict = "axe critique"
    summary = (
        f"Sur l'axe {axis}, la moyenne est de {avg:.1f}/5 ({verdict}). "
        f"Le critere le mieux note est \"{strongest[0]}\" ({strongest[1]}/5 - "
        f"{_anchor_for(axis, strongest[0], strongest[1])}). Le point faible "
        f"identifie est \"{weakest[0]}\" ({weakest[1]}/5 - "
        f"{_anchor_for(axis, weakest[0], weakest[1])})."
    )
    strengths = [
        f"{c} note {s}/5 : {_anchor_for(axis, c, s)}"
        for c, s in pairs if s >= 4
    ]
    gaps = [
        f"{c} note {s}/5 : {_anchor_for(axis, c, s)}"
        for c, s in pairs if s <= 2
    ]
    if avg < 3:
        advice = (
            f"Demander un complement de dossier sur \"{weakest[0]}\" avant "
            "presentation au comite. Convoquer eventuellement le porteur en audition."
        )
    elif avg < 4:
        advice = (
            f"Eligible mais conditionner l'avis favorable a une clarification sur "
            f"\"{weakest[0]}\"."
        )
    else:
        advice = "Axe non bloquant - peut etre validee en l'etat."
    return {"summary": summary, "strengths": strengths, "gaps": gaps, "advice": advice}


def method_rationale(method: str, value_usd: float, fmva: dict[str, Any], result: dict[str, Any]) -> str:
    """Plain-language rationale for each FMVA valuation method."""
    if method == "Berkus":
        top = max(fmva["berkus"].items(), key=lambda kv: kv[1])
        return (
            f"Berkus aggrege 5 facteurs de derisque a USD 500k chacun. "
            f"Total attribue : ${value_usd:,.0f}. Le facteur dominant est "
            f"\"{top[0]}\" a ${top[1]:,.0f}. "
            f"Cap (USD 2.5M) {'respecte' if result['berkus_cap_ok'] else 'depasse'}."
        )
    if method == "Scorecard (Payne)":
        mult = result["scorecard_weighted_multiplier"]
        return (
            f"La methode Scorecard de Payne applique un multiplicateur pondere "
            f"de {mult:.2f}x sur la baseline pre-money tunisienne "
            f"(${fmva.get('baseline_usd', TUNISIA_BASELINE_USD):,.0f}). "
            f"Valorisation : ${value_usd:,.0f}. "
            f"Pondere principalement par Management (25%) et Opportunite (20%)."
        )
    if method == "Risk Factor Summation":
        adj = result["rfs_adjustment_usd"]
        sign = "majoree" if adj >= 0 else "diminuee"
        return (
            f"RFS evalue 12 dimensions de risque entre -2 et +2, chaque "
            f"increment valant USD 250k. La baseline est {sign} de "
            f"${abs(adj):,.0f} pour atteindre ${value_usd:,.0f}."
        )
    if method == "Venture Capital Method":
        vcb = result["vc_breakdown"]
        return (
            f"Methode VC : revenus actuels projetes a la sortie an {fmva['vc']['exit_year']} "
            f"avec un multiple de {fmva['vc']['exit_multiple']:.1f}x = "
            f"${vcb['projected_exit_usd']:,.0f}. Discount par le rendement cible "
            f"({fmva['vc']['target_return']:.0f}x) puis dilution = pre-money "
            f"${value_usd:,.0f}."
        )
    if method == "Hybrid DCF":
        db = result["dcf_breakdown"]
        return (
            f"DCF a 5 ans : EBITDA initial {fmva['dcf']['starting_ebitda_tnd']:,.0f} TND "
            f"croissant a {fmva['dcf']['ebitda_growth']:.0%}/an, actualise au "
            f"WACC de {fmva['dcf']['wacc']:.1%}. Valeur terminale = EBITDA an 5 "
            f"x {fmva['dcf']['terminal_multiple']:.1f}. "
            f"Enterprise value = {db['enterprise_tnd']:,.0f} TND "
            f"({value_usd:,.0f} USD)."
        )
    return f"Valorisation : ${value_usd:,.0f}."


def overall_recommendation(
    score: float,
    scorecard: dict[str, Any],
    fmva_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Produce a SELECT / DEFER / DECLINE call with argued rationale."""
    global_note = scorecard["global_note"]
    iqr = fmva_result["iqr_ratio"] if fmva_result else 0.0
    rationale: list[str] = []
    if score >= 70:
        rationale.append(f"Score de financement modelise {score:.0f}/100 (au-dessus du seuil de selection 70).")
    elif score >= 50:
        rationale.append(f"Score modelise {score:.0f}/100 (zone d'arbitrage 50-70).")
    else:
        rationale.append(f"Score modelise {score:.0f}/100 (sous le seuil de viabilite).")

    if global_note >= 3.5:
        rationale.append(f"Note de comite agregee {global_note}/5 ({scorecard['recommendation']}).")
    elif global_note >= 2.5:
        rationale.append(f"Note de comite {global_note}/5 - dossier moyen, axes a renforcer.")
    else:
        rationale.append(f"Note de comite {global_note}/5 - dossier insuffisant en l'etat.")

    weak = sorted(scorecard["axes"], key=lambda a: a["note"])[:2]
    if weak:
        rationale.append(
            "Axes les plus faibles : "
            + ", ".join(f"{w['axis']} ({w['note']}/5)" for w in weak)
            + "."
        )

    if fmva_result:
        if iqr > 0.6:
            rationale.append(
                f"Triangulation FMVA dispersee (IQR {iqr:.0%}) - hypotheses a "
                "stresser avant decision finale."
            )
        else:
            rationale.append(
                f"Triangulation FMVA convergente (IQR {iqr:.0%}) - estimation "
                f"d'ensemble ${fmva_result['ensemble_usd']:,.0f}."
            )

    if score >= 70 and global_note >= 3.5:
        action = "SELECTIONNER"
        tone = "ok"
        color = GREEN
    elif score >= 50 and global_note >= 2.5:
        action = "DIFFERER (complement de dossier)"
        tone = "warn"
        color = AMBER
    else:
        action = "REJETER"
        tone = "bad"
        color = RED
    return {
        "action": action,
        "tone": tone,
        "color": color,
        "rationale": rationale,
        "next_steps": _next_steps(scorecard, fmva_result, action),
    }


def _next_steps(
    scorecard: dict[str, Any],
    fmva_result: dict[str, Any] | None,
    action: str,
) -> list[str]:
    steps: list[str] = []
    if action.startswith("REJETER"):
        steps.append("Notifier le porteur par lettre motivee (avec axes faibles).")
        steps.append("Proposer une orientation vers un programme d'incubation.")
        return steps
    if action.startswith("DIFFERER"):
        gaps = [ax for ax in scorecard["axes"] if ax["note"] <= 2]
        for ax in gaps[:3]:
            steps.append(f"Demander pieces complementaires sur \"{ax['axis']}\".")
    else:
        steps.append("Convoquer audition de selection (comite VAIR).")
        steps.append("Programmer une due diligence financiere et juridique.")
    if fmva_result and fmva_result["iqr_ratio"] > 0.6:
        steps.append("Re-tester les hypotheses VC et DCF (croissance, multiple, WACC).")
    steps.append("Faire signer la convention de confidentialite.")
    return steps


AXIS_WEIGHTS: dict[str, float] = {
    "Innovation et Proposition de valeur": 0.18,
    "Adéquation au marché ciblé": 0.18,
    "Qualité et complémentarité de l'équipe": 0.18,
    "Évaluation du PoC": 0.16,
    "Cohérence TRL, budget et remboursement": 0.14,
    "Impact environnemental et social": 0.10,
    "Risques et crédibilité globale": 0.06,
}


def committee_scorecard(scores: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Aggregate per-criterion scores (0-5) into per-axis and global notes."""
    axis_rows: list[dict[str, Any]] = []
    global_weighted = 0.0
    total_weight = 0.0
    for block in SCORING_GRID:
        axis = block["axis"]
        axis_scores = scores.get(axis, {})
        crit_values = [
            max(0, min(5, int(axis_scores.get(crit["name"], 0))))
            for crit in block["criteria"]
        ]
        avg = round(sum(crit_values) / max(1, len(crit_values)))
        weight = AXIS_WEIGHTS.get(axis, 1.0 / len(SCORING_GRID))
        global_weighted += avg * weight
        total_weight += weight
        axis_rows.append(
            {
                "axis": axis,
                "weight": weight,
                "criteria": [
                    {"name": crit["name"], "score": value}
                    for crit, value in zip(block["criteria"], crit_values)
                ],
                "note": avg,
            }
        )
    global_note = round(global_weighted / max(1e-9, total_weight), 2)
    if global_note >= 3.5:
        recommendation = "Positif"
        tone = "ok"
    elif global_note >= 2.5:
        recommendation = "Neutre"
        tone = "warn"
    else:
        recommendation = "Négatif"
        tone = "bad"
    return {
        "axes": axis_rows,
        "global_note": global_note,
        "recommendation": recommendation,
        "tone": tone,
    }


def auto_score_grid(inputs: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Heuristic auto-scoring from assessment form inputs (0-5 per criterion)."""
    stage = int(inputs.get("stage", 1))
    team = float(inputs.get("team", 0.65))
    market = float(inputs.get("market", 0.65))
    product = float(inputs.get("product", 0.60))
    competition = float(inputs.get("competition", 0.50))
    revenue = float(inputs.get("revenue_tnd", 0.0) or 0.0)
    growth = float(inputs.get("growth", 0.40))
    is_labelled = bool(inputs.get("is_labelled", False))
    n_founders = int(inputs.get("n_founders", 1))
    has_web = bool(inputs.get("has_web", False))
    has_email = bool(inputs.get("has_email", False))

    def clamp(x: float) -> int:
        return int(max(0, min(5, round(x))))

    base_innov = product * 4 + (1 if stage >= 1 else 0)
    base_market = market * 4 + (1 if revenue > 0 else 0)
    base_team = team * 4 + (1 if n_founders >= 2 else 0) + (1 if is_labelled else 0)
    base_poc = product * 4 + stage
    base_budget = (team * 0.5 + product * 0.5) * 4 + (1 if revenue > 0 else 0)
    base_impact = market * 3 + (1 if is_labelled else 0) + (1 if has_web else 0)
    base_risk = (1 - competition) * 4 + (1 if has_email else 0)

    return {
        "Innovation et Proposition de valeur": {
            "Originalité de la solution": clamp(base_innov - 1),
            "Pertinence de la valeur ajoutée pour l'utilisateur": clamp(base_innov),
        },
        "Adéquation au marché ciblé": {
            "Clarté et pertinence de la définition du marché": clamp(base_market),
            "Validation et accessibilité du marché": clamp(base_market - 1 + (1 if revenue > 0 else 0)),
            "Crédibilité de la stratégie d'accès au marché": clamp(market * 4 + growth),
            "Partenaires stratégiques (optionnel)": clamp(market * 3 + (1 if is_labelled else 0)),
        },
        "Qualité et complémentarité de l'équipe": {
            "Compétences et expérience": clamp(base_team),
            "Capacité d'exécution et organisation": clamp(team * 4 + stage),
            "Historique entrepreneurial & PI": clamp(team * 3 + (1 if is_labelled else 0)),
        },
        "Évaluation du PoC": {
            "Clarté de la description technique": clamp(product * 4 + (1 if has_web else 0)),
            "Préparation technique pour le PoC": clamp(base_poc),
            "Ressources techniques mobilisées": clamp(team * 3 + product * 2),
            "Vision de passage à l'échelle": clamp(market * 3 + growth * 2),
        },
        "Cohérence TRL, budget et remboursement": {
            "Maturité de l'innovation (TRL atteint)": clamp(stage * 2 + product * 2),
            "Pertinence et réalisme du budget": clamp(base_budget),
            "Capacité (ou plan) de gestion financière": clamp(team * 4 + (1 if revenue > 0 else 0)),
            "Crédibilité de la trajectoire de revenus / remboursement": clamp(
                (revenue > 0) * 3 + growth * 2 + market * 1
            ),
        },
        "Impact environnemental et social": {
            "Potentiel d'impact environnemental": clamp(base_impact),
            "Durabilité de la solution": clamp(market * 3 + (1 if is_labelled else 0)),
            "Mesure et suivi de l'impact": clamp(team * 3 + (1 if is_labelled else 0)),
            "Prise en compte du genre": clamp(2 + (1 if is_labelled else 0)),
            "Création d'emplois": clamp(team * 3 + (1 if revenue > 0 else 0)),
        },
        "Risques et crédibilité globale": {
            "Identification et pertinence des risques": clamp(base_risk),
            "Stratégies de mitigation": clamp((1 - competition) * 4 + team),
            "Vision de passage à l'échelle post-PoC": clamp(market * 3 + growth * 2),
        },
    }


def scoring_grid_excel(payload: dict[str, Any], scorecard: dict[str, Any]) -> io.BytesIO:
    """Export a committee scorecard in the VAIR Greentech grid format."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    wb = Workbook()
    cover = wb.active
    cover.title = "Synthèse"
    navy = PatternFill("solid", fgColor="272E5F")
    light = PatternFill("solid", fgColor="F4F5F9")
    white_bold = Font(color="FFFFFF", bold=True, size=12)
    bold = Font(bold=True)
    wrap = Alignment(wrap_text=True, vertical="top")
    thin = Side(style="thin", color="D9DCE6")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    cover["A1"] = "CDC LAUNCHPAD — Grille de scoring VAIR"
    cover["A1"].font = Font(size=15, bold=True, color="272E5F")
    rows = [
        ("Startup", payload.get("name", "")),
        ("Secteur", payload.get("sector", "")),
        ("Région", payload.get("region", "")),
        ("Date", f"{dt.date.today():%Y-%m-%d}"),
        ("Note globale (/5)", scorecard["global_note"]),
        ("Recommandation", scorecard["recommendation"]),
    ]
    for i, (key, value) in enumerate(rows, start=3):
        cover.cell(i, 1, key).font = bold
        cover.cell(i, 2, value)
    cover.cell(3 + len(rows) + 1, 1, "Axe").font = white_bold
    cover.cell(3 + len(rows) + 1, 1).fill = navy
    cover.cell(3 + len(rows) + 1, 2, "Pondération").font = white_bold
    cover.cell(3 + len(rows) + 1, 2).fill = navy
    cover.cell(3 + len(rows) + 1, 3, "Note (/5)").font = white_bold
    cover.cell(3 + len(rows) + 1, 3).fill = navy
    for j, axis_row in enumerate(scorecard["axes"], start=3 + len(rows) + 2):
        cover.cell(j, 1, axis_row["axis"])
        cover.cell(j, 2, f"{axis_row['weight']:.0%}")
        cover.cell(j, 3, axis_row["note"])
    cover.column_dimensions["A"].width = 46
    cover.column_dimensions["B"].width = 18
    cover.column_dimensions["C"].width = 14

    for block, axis_row in zip(SCORING_GRID, scorecard["axes"]):
        title = block["axis"][:28]
        ws = wb.create_sheet(title=title or "Axe")
        ws["A1"] = block["axis"]
        ws["A1"].font = Font(size=13, bold=True, color="FFFFFF")
        ws["A1"].fill = navy
        ws.merge_cells("A1:I1")
        ws["A2"] = block["guidance"]
        ws["A2"].alignment = wrap
        ws.merge_cells("A2:I2")
        headers = ["Critère", "Angle d'analyse", "Éléments à consulter",
                   "0", "1", "2", "3", "4", "5", "Note"]
        for col, header in enumerate(headers, start=1):
            cell = ws.cell(4, col, header)
            cell.font = white_bold
            cell.fill = navy
            cell.alignment = wrap
            cell.border = border
        scores_by_name = {c["name"]: c["score"] for c in axis_row["criteria"]}
        for row_i, crit in enumerate(block["criteria"], start=5):
            ws.cell(row_i, 1, crit["name"]).font = bold
            ws.cell(row_i, 2, crit["angle"])
            ws.cell(row_i, 3, crit["fields"])
            for k, anchor in enumerate(crit["anchors"]):
                ws.cell(row_i, 4 + k, anchor)
            ws.cell(row_i, 10, scores_by_name.get(crit["name"], 0)).font = bold
            for col in range(1, 11):
                ws.cell(row_i, col).alignment = wrap
                ws.cell(row_i, col).border = border
        total_row = 5 + len(block["criteria"])
        ws.cell(total_row, 1, f"Note {block['axis']}").font = bold
        ws.cell(total_row, 1).fill = light
        ws.cell(total_row, 10, axis_row["note"]).font = bold
        ws.cell(total_row, 10).fill = light
        widths = [32, 38, 30, 18, 18, 18, 18, 18, 18, 8]
        for col, w in enumerate(widths, start=1):
            ws.column_dimensions[chr(64 + col)].width = w

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer


# ---------------------------------------------------------------------------
# FMVA valuation workbench (Berkus / Scorecard / RFS / VC / Hybrid DCF + Ensemble)
# ---------------------------------------------------------------------------
TND_PER_USD = 3.1
TUNISIA_BASELINE_USD = 1_800_000

SCORECARD_WEIGHTS: dict[str, float] = {
    "Strength of Management": 0.25,
    "Size of Opportunity": 0.20,
    "Product/Technology": 0.15,
    "Competitive Environment": 0.10,
    "Marketing/Sales/Partnerships": 0.10,
    "Need for more investment": 0.10,
    "Other (barriers, quality, speed)": 0.10,
}

RFS_DIMENSIONS: list[str] = [
    "Management risk",
    "Stage of business",
    "Legislation/Political",
    "Manufacturing",
    "Sales & marketing",
    "Funding/capital",
    "Competition",
    "Technology",
    "Litigation",
    "International",
    "Reputation",
    "Exit value",
]

BERKUS_FACTORS: list[str] = [
    "Sound idea",
    "Prototype (reduces tech risk)",
    "Quality management team",
    "Strategic relationships",
    "Product rollout or sales",
]

ENSEMBLE_WEIGHTS: dict[str, float] = {
    "Berkus": 0.20,
    "Scorecard (Payne)": 0.25,
    "Risk Factor Summation": 0.15,
    "Venture Capital Method": 0.20,
    "Hybrid DCF": 0.20,
}


def auto_fmva_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Translate assessment form answers into FMVA pre-fill inputs."""
    stage = int(inputs.get("stage", 1))
    team = float(inputs.get("team", 0.65))
    market = float(inputs.get("market", 0.65))
    product = float(inputs.get("product", 0.60))
    competition = float(inputs.get("competition", 0.50))
    revenue_tnd = float(inputs.get("revenue_tnd", 0.0) or 0.0)
    growth = float(inputs.get("growth", 0.40))
    is_labelled = bool(inputs.get("is_labelled", False))

    def cap(v: float, lo: float = 0.0, hi: float = 500_000) -> int:
        return int(round(max(lo, min(hi, v))))

    berkus = {
        "Sound idea": cap(500_000 * (0.4 + 0.6 * market)),
        "Prototype (reduces tech risk)": cap(500_000 * (0.3 + 0.7 * product)),
        "Quality management team": cap(500_000 * (0.3 + 0.7 * team)),
        "Strategic relationships": cap(500_000 * (0.2 + 0.6 * market + 0.2 * (1 if is_labelled else 0))),
        "Product rollout or sales": cap(500_000 * (0.2 + 0.8 * min(1.0, stage / 2 + (revenue_tnd > 0)))),
    }

    def to_payne(x: float) -> float:
        return round(0.5 + x, 2)

    scorecard = {
        "Strength of Management": to_payne(team),
        "Size of Opportunity": to_payne(market),
        "Product/Technology": to_payne(product),
        "Competitive Environment": to_payne(1 - competition),
        "Marketing/Sales/Partnerships": to_payne((market + (1 if is_labelled else 0) * 0.2) / 1.2),
        "Need for more investment": to_payne(0.5),
        "Other (barriers, quality, speed)": to_payne((product + (1 if is_labelled else 0) * 0.2) / 1.2),
    }

    def to_rfs(level: float) -> int:
        if level >= 0.75:
            return 2
        if level >= 0.55:
            return 1
        if level >= 0.40:
            return 0
        if level >= 0.25:
            return -1
        return -2

    rfs = {
        "Management risk": to_rfs(team),
        "Stage of business": to_rfs(0.3 + 0.35 * stage),
        "Legislation/Political": 1 if is_labelled else 0,
        "Manufacturing": to_rfs(product),
        "Sales & marketing": to_rfs(market),
        "Funding/capital": to_rfs(0.5 + 0.3 * (revenue_tnd > 0)),
        "Competition": to_rfs(1 - competition),
        "Technology": to_rfs(product),
        "Litigation": 0,
        "International": to_rfs(0.4 * market),
        "Reputation": to_rfs(team),
        "Exit value": to_rfs(market),
    }

    vc = {
        "current_revenue_tnd": revenue_tnd if revenue_tnd > 0 else 850_000,
        "growth": max(0.10, growth),
        "exit_year": 5,
        "exit_multiple": 4.5,
        "target_return": 10.0,
        "round_size_usd": 500_000,
    }

    dcf = {
        "starting_ebitda_tnd": max(120_000, revenue_tnd * 0.15) if revenue_tnd > 0 else 120_000,
        "ebitda_growth": max(0.20, growth),
        "wacc": 0.275,
        "terminal_multiple": 5.0,
    }

    return {
        "baseline_usd": TUNISIA_BASELINE_USD,
        "tnd_per_usd": TND_PER_USD,
        "berkus": berkus,
        "scorecard": scorecard,
        "rfs": rfs,
        "vc": vc,
        "dcf": dcf,
    }


def fmva_valuation(fmva: dict[str, Any]) -> dict[str, Any]:
    """Run all 5 FMVA methods + ensemble triangulation."""
    baseline_usd = float(fmva.get("baseline_usd", TUNISIA_BASELINE_USD))
    tnd_per_usd = float(fmva.get("tnd_per_usd", TND_PER_USD))

    berkus_values = fmva.get("berkus", {})
    berkus_usd = sum(float(berkus_values.get(f, 0)) for f in BERKUS_FACTORS)
    berkus_cap_ok = berkus_usd <= 2_500_000

    sc = fmva.get("scorecard", {})
    weighted = sum(SCORECARD_WEIGHTS[f] * float(sc.get(f, 1.0)) for f in SCORECARD_WEIGHTS)
    scorecard_usd = baseline_usd * weighted

    rfs_inputs = fmva.get("rfs", {})
    rfs_adjust = sum(int(rfs_inputs.get(d, 0)) * 250_000 for d in RFS_DIMENSIONS)
    rfs_usd = baseline_usd + rfs_adjust

    vc = fmva.get("vc", {})
    current = float(vc.get("current_revenue_tnd", 0))
    g = float(vc.get("growth", 0.4))
    n = int(vc.get("exit_year", 5))
    exit_mult = float(vc.get("exit_multiple", 4.5))
    target_return = float(vc.get("target_return", 10.0))
    projected_exit_tnd = current * (1 + g) ** n * exit_mult
    projected_exit_usd = projected_exit_tnd / tnd_per_usd
    round_size_usd = float(vc.get("round_size_usd", 500_000))
    post_money_usd = projected_exit_usd / max(1e-6, target_return)
    vc_usd = max(0.0, post_money_usd - round_size_usd)

    dcf = fmva.get("dcf", {})
    ebitda0 = float(dcf.get("starting_ebitda_tnd", 120_000))
    ebitda_g = float(dcf.get("ebitda_growth", 0.45))
    wacc = float(dcf.get("wacc", 0.275))
    term_mult = float(dcf.get("terminal_multiple", 5.0))
    pv_sum_tnd = 0.0
    ebitda_y = ebitda0
    ebitda_path = []
    for year in range(1, 6):
        ebitda_y = ebitda_y * (1 + ebitda_g)
        discount = (1 + wacc) ** year
        pv = ebitda_y / discount
        pv_sum_tnd += pv
        ebitda_path.append({"year": year, "ebitda": round(ebitda_y), "pv": round(pv)})
    terminal_tnd = ebitda_y * term_mult
    pv_terminal_tnd = terminal_tnd / (1 + wacc) ** 5
    enterprise_tnd = pv_sum_tnd + pv_terminal_tnd
    dcf_usd = enterprise_tnd / tnd_per_usd

    methods = {
        "Berkus": berkus_usd,
        "Scorecard (Payne)": scorecard_usd,
        "Risk Factor Summation": rfs_usd,
        "Venture Capital Method": vc_usd,
        "Hybrid DCF": dcf_usd,
    }
    weighted_sum = sum(ENSEMBLE_WEIGHTS[m] * v for m, v in methods.items())
    low = min(methods.values())
    high = max(methods.values())
    iqr_ratio = (high - low) / max(1.0, weighted_sum)
    review_flag = iqr_ratio > 0.6

    return {
        "methods_usd": methods,
        "methods_tnd": {m: v * tnd_per_usd for m, v in methods.items()},
        "ensemble_usd": weighted_sum,
        "ensemble_tnd": weighted_sum * tnd_per_usd,
        "low_usd": low,
        "high_usd": high,
        "iqr_ratio": iqr_ratio,
        "review_flag": review_flag,
        "berkus_cap_ok": berkus_cap_ok,
        "berkus_breakdown": {f: float(berkus_values.get(f, 0)) for f in BERKUS_FACTORS},
        "scorecard_weighted_multiplier": weighted,
        "rfs_adjustment_usd": rfs_adjust,
        "vc_breakdown": {
            "projected_exit_tnd": projected_exit_tnd,
            "projected_exit_usd": projected_exit_usd,
            "post_money_usd": post_money_usd,
            "round_size_usd": round_size_usd,
        },
        "dcf_breakdown": {
            "path": ebitda_path,
            "pv_explicit_tnd": pv_sum_tnd,
            "terminal_tnd": terminal_tnd,
            "pv_terminal_tnd": pv_terminal_tnd,
            "enterprise_tnd": enterprise_tnd,
            "enterprise_usd": dcf_usd,
        },
    }


def fmva_workbook_excel(payload: dict[str, Any], fmva: dict[str, Any], result: dict[str, Any]) -> io.BytesIO:
    """Export a full FMVA workbook (Inputs, Berkus, Scorecard, RFS, VC, DCF, Ensemble)."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    wb = Workbook()
    navy = PatternFill("solid", fgColor="272E5F")
    red = PatternFill("solid", fgColor="D10A11")
    light = PatternFill("solid", fgColor="F4F5F9")
    white_bold = Font(color="FFFFFF", bold=True)
    bold = Font(bold=True)
    thin = Side(style="thin", color="D9DCE6")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    def header(ws, row, cells, fill=navy):
        for col, val in enumerate(cells, start=1):
            c = ws.cell(row, col, val)
            c.font = white_bold
            c.fill = fill
            c.border = border
            c.alignment = Alignment(wrap_text=True)

    def body_row(ws, row, cells):
        for col, val in enumerate(cells, start=1):
            c = ws.cell(row, col, val)
            c.border = border

    ws = wb.active
    ws.title = "Inputs"
    ws["A1"] = "FMVA AutoFill — Inputs"
    ws["A1"].font = Font(size=14, bold=True, color="272E5F")
    rows = [
        ("Startup name", payload.get("name", "")),
        ("Sector", payload.get("sector", "")),
        ("Region", payload.get("region", "")),
        ("Date", f"{dt.date.today():%Y-%m-%d}"),
        ("Tunisia baseline pre-money (USD)", fmva.get("baseline_usd", TUNISIA_BASELINE_USD)),
        ("TND to USD conversion rate", fmva.get("tnd_per_usd", TND_PER_USD)),
        ("WACC", fmva.get("dcf", {}).get("wacc", 0.275)),
        ("Terminal multiple", fmva.get("dcf", {}).get("terminal_multiple", 5.0)),
        ("Target VC return (multiple)", fmva.get("vc", {}).get("target_return", 10.0)),
        ("Expected exit years", fmva.get("vc", {}).get("exit_year", 5)),
    ]
    for i, (k, v) in enumerate(rows, start=3):
        ws.cell(i, 1, k).font = bold
        ws.cell(i, 2, v)
    ws.column_dimensions["A"].width = 40
    ws.column_dimensions["B"].width = 26

    ws = wb.create_sheet("Berkus")
    ws["A1"] = "Berkus — 5 factors x up to USD 500k"
    ws["A1"].font = Font(size=13, bold=True, color="272E5F")
    header(ws, 3, ["Factor", "Max (USD)", "Assigned (USD)"])
    for i, factor in enumerate(BERKUS_FACTORS, start=4):
        body_row(ws, i, [factor, 500_000, fmva["berkus"].get(factor, 0)])
    total_row = 4 + len(BERKUS_FACTORS)
    ws.cell(total_row, 1, "TOTAL (USD)").font = bold
    ws.cell(total_row, 3, sum(fmva["berkus"].values())).font = bold
    ws.cell(total_row, 1).fill = light
    ws.cell(total_row, 3).fill = light
    ws.cell(total_row + 1, 1, "Cap check (<= USD 2,500,000)").font = bold
    ws.cell(total_row + 1, 3, "OK" if result["berkus_cap_ok"] else "EXCEEDED")
    ws.column_dimensions["A"].width = 40
    for c in ["B", "C"]:
        ws.column_dimensions[c].width = 18

    ws = wb.create_sheet("Scorecard")
    ws["A1"] = "Scorecard (Payne) — weighted vs Tunisia baseline"
    ws["A1"].font = Font(size=13, bold=True, color="272E5F")
    ws.cell(3, 1, "Regional baseline pre-money (USD)").font = bold
    ws.cell(3, 2, fmva.get("baseline_usd", TUNISIA_BASELINE_USD))
    header(ws, 5, ["Factor", "Weight", "Target Score (0.5-1.5)", "Weighted"])
    for i, (factor, weight) in enumerate(SCORECARD_WEIGHTS.items(), start=6):
        target = float(fmva["scorecard"].get(factor, 1.0))
        body_row(ws, i, [factor, weight, target, round(weight * target, 4)])
    total_row = 6 + len(SCORECARD_WEIGHTS)
    ws.cell(total_row, 1, "Total weighted multiplier").font = bold
    ws.cell(total_row, 4, round(result["scorecard_weighted_multiplier"], 4)).font = bold
    ws.cell(total_row + 1, 1, "SCORECARD VALUATION (USD)").font = bold
    ws.cell(total_row + 1, 2, round(result["methods_usd"]["Scorecard (Payne)"]))
    ws.cell(total_row + 1, 2).fill = light
    ws.column_dimensions["A"].width = 40
    for c in ["B", "C", "D"]:
        ws.column_dimensions[c].width = 18

    ws = wb.create_sheet("RiskFactor")
    ws["A1"] = "Risk Factor Summation — 12 dimensions"
    ws["A1"].font = Font(size=13, bold=True, color="272E5F")
    ws.cell(3, 1, "Base valuation (USD)").font = bold
    ws.cell(3, 2, fmva.get("baseline_usd", TUNISIA_BASELINE_USD))
    ws.cell(4, 1, "Increment size (USD)").font = bold
    ws.cell(4, 2, 250_000)
    header(ws, 6, ["Risk Factor", "Rating (-2 to +2)", "Adjustment (USD)"])
    for i, dim in enumerate(RFS_DIMENSIONS, start=7):
        rating = int(fmva["rfs"].get(dim, 0))
        body_row(ws, i, [dim, rating, rating * 250_000])
    total_row = 7 + len(RFS_DIMENSIONS)
    ws.cell(total_row, 1, "Total adjustment").font = bold
    ws.cell(total_row, 3, result["rfs_adjustment_usd"]).font = bold
    ws.cell(total_row + 1, 1, "RFS VALUATION (USD)").font = bold
    ws.cell(total_row + 1, 3, round(result["methods_usd"]["Risk Factor Summation"]))
    ws.cell(total_row + 1, 3).fill = light
    ws.column_dimensions["A"].width = 36
    for c in ["B", "C"]:
        ws.column_dimensions[c].width = 22

    ws = wb.create_sheet("VC_Method")
    ws["A1"] = "Venture Capital Method"
    ws["A1"].font = Font(size=13, bold=True, color="272E5F")
    vc = fmva.get("vc", {})
    vcb = result["vc_breakdown"]
    pairs = [
        ("Current annual revenue (TND)", vc.get("current_revenue_tnd", 0)),
        ("Revenue growth rate (annual)", vc.get("growth", 0.4)),
        ("Exit year", vc.get("exit_year", 5)),
        ("Exit revenue multiple", vc.get("exit_multiple", 4.5)),
        ("Projected exit value (TND)", round(vcb["projected_exit_tnd"])),
        ("TND to USD rate", fmva.get("tnd_per_usd", TND_PER_USD)),
        ("Projected exit value (USD)", round(vcb["projected_exit_usd"])),
        ("Target VC return (multiple)", vc.get("target_return", 10.0)),
        ("Post-money valuation (USD)", round(vcb["post_money_usd"])),
        ("Round size (USD)", round(vcb["round_size_usd"])),
        ("PRE-MONEY VALUATION (USD)", round(result["methods_usd"]["Venture Capital Method"])),
    ]
    for i, (k, v) in enumerate(pairs, start=3):
        ws.cell(i, 1, k).font = bold
        ws.cell(i, 2, v)
    ws.column_dimensions["A"].width = 36
    ws.column_dimensions["B"].width = 22

    ws = wb.create_sheet("Hybrid_DCF")
    ws["A1"] = "Hybrid DCF — 5y explicit + terminal multiple"
    ws["A1"].font = Font(size=13, bold=True, color="272E5F")
    d = fmva.get("dcf", {})
    db = result["dcf_breakdown"]
    pre = [
        ("Starting EBITDA (TND)", d.get("starting_ebitda_tnd", 120_000)),
        ("EBITDA growth rate", d.get("ebitda_growth", 0.45)),
        ("WACC", d.get("wacc", 0.275)),
        ("Terminal multiple", d.get("terminal_multiple", 5.0)),
    ]
    for i, (k, v) in enumerate(pre, start=3):
        ws.cell(i, 1, k).font = bold
        ws.cell(i, 2, v)
    header(ws, 8, ["Item", "Year 1", "Year 2", "Year 3", "Year 4", "Year 5"])
    ws.cell(9, 1, "EBITDA (TND)").font = bold
    ws.cell(10, 1, "PV of EBITDA (TND)").font = bold
    for j, p in enumerate(db["path"], start=2):
        ws.cell(9, j, p["ebitda"])
        ws.cell(10, j, p["pv"])
    base = 12
    rows2 = [
        ("Sum of PV (explicit, TND)", round(db["pv_explicit_tnd"])),
        ("Terminal value (TND)", round(db["terminal_tnd"])),
        ("PV of terminal (TND)", round(db["pv_terminal_tnd"])),
        ("Enterprise value (TND)", round(db["enterprise_tnd"])),
        ("Enterprise value (USD)", round(db["enterprise_usd"])),
    ]
    for i, (k, v) in enumerate(rows2, start=base):
        ws.cell(i, 1, k).font = bold
        ws.cell(i, 2, v)
    ws.column_dimensions["A"].width = 36
    for c in ["B", "C", "D", "E", "F"]:
        ws.column_dimensions[c].width = 16

    ws = wb.create_sheet("Ensemble")
    ws["A1"] = "Ensemble Valuation — Triangulated"
    ws["A1"].font = Font(size=13, bold=True, color="272E5F")
    header(ws, 3, ["Method", "Valuation (USD)", "Weight", "Weighted (USD)"], fill=red)
    for i, (m, v) in enumerate(result["methods_usd"].items(), start=4):
        w = ENSEMBLE_WEIGHTS[m]
        body_row(ws, i, [m, round(v), w, round(v * w)])
    end = 4 + len(result["methods_usd"])
    ws.cell(end + 1, 1, "ENSEMBLE VALUATION (USD)").font = bold
    ws.cell(end + 1, 4, round(result["ensemble_usd"])).font = bold
    ws.cell(end + 1, 1).fill = light
    ws.cell(end + 1, 4).fill = light
    ws.cell(end + 2, 1, "Low (min of 5 methods)")
    ws.cell(end + 2, 4, round(result["low_usd"]))
    ws.cell(end + 3, 1, "High (max of 5 methods)")
    ws.cell(end + 3, 4, round(result["high_usd"]))
    ws.cell(end + 4, 1, "IQR (% of ensemble)")
    ws.cell(end + 4, 4, f"{result['iqr_ratio']:.1%}")
    ws.cell(end + 5, 1, "Data-quality flag (>60% IQR = review)").font = bold
    ws.cell(end + 5, 4, "Review required" if result["review_flag"] else "OK")
    ws.column_dimensions["A"].width = 38
    for c in ["B", "C", "D"]:
        ws.column_dimensions[c].width = 18

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer


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


def committee_pdf(
    payload: dict[str, Any],
    scorecard: dict[str, Any],
    rationales: dict[str, dict[str, Any]] | None = None,
    overall: dict[str, Any] | None = None,
) -> io.BytesIO:
    """Render the committee scorecard as a fully argued PDF report."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
        SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=15 * mm, bottomMargin=15 * mm,
        leftMargin=16 * mm, rightMargin=16 * mm,
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("CTitle", parent=styles["Title"], textColor=colors.HexColor(NAVY))
    h2 = ParagraphStyle("CH2", parent=styles["Heading2"], textColor=colors.HexColor(NAVY), spaceAfter=4)
    h3 = ParagraphStyle("CH3", parent=styles["Heading3"], textColor=colors.HexColor(RED), spaceAfter=2)
    body = styles["BodyText"]
    small = ParagraphStyle("Small", parent=body, fontSize=8, textColor=colors.HexColor(MUTED))
    elements: list[Any] = []

    if os.path.exists(LOGO_FILE):
        try:
            elements.append(Image(LOGO_FILE, width=42 * mm, height=17 * mm))
            elements.append(Spacer(1, 6))
        except Exception:
            pass

    elements.append(Paragraph("Rapport de comite - Grille VAIR", title))
    elements.append(Paragraph(
        f"<b>{payload.get('name', '')}</b> | {payload.get('sector', '')} | "
        f"{payload.get('region', '')} | {dt.date.today():%d %b %Y}", body))
    if payload.get("evaluator"):
        elements.append(Paragraph(f"Evaluateur : {payload['evaluator']}", small))
    elements.append(Spacer(1, 6))

    if overall:
        rec_color = overall.get("color", NAVY)
        elements.append(Paragraph(
            f"<font color='{rec_color}'><b>Recommandation : {overall['action']}</b></font>", h2))
        elements.append(Paragraph(
            f"Note globale ponderee : <b>{scorecard['global_note']}/5</b> "
            f"({scorecard['recommendation']})", body))
        for line in overall.get("rationale", []):
            elements.append(Paragraph(f"- {line}", body))
        elements.append(Spacer(1, 6))
        if overall.get("next_steps"):
            elements.append(Paragraph("Prochaines etapes", h3))
            for step in overall["next_steps"]:
                elements.append(Paragraph(f"- {step}", body))
            elements.append(Spacer(1, 6))
    else:
        elements.append(Paragraph(
            f"Note globale ponderee : <b>{scorecard['global_note']}/5</b> "
            f"({scorecard['recommendation']})", h2))
        elements.append(Spacer(1, 6))

    rows = [["Axe", "Ponderation", "Note /5"]]
    for ax in scorecard["axes"]:
        rows.append([ax["axis"], f"{ax['weight']:.0%}", str(ax["note"])])
    summary_table = Table(rows, colWidths=[100 * mm, 30 * mm, 25 * mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(LINE)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(LIGHT)]),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 8))

    for ax in scorecard["axes"]:
        block_elements: list[Any] = [
            Paragraph(f"{ax['axis']} - Note {ax['note']}/5", h2),
        ]
        rat = (rationales or {}).get(ax["axis"]) or {}
        if rat.get("summary"):
            block_elements.append(Paragraph(rat["summary"], body))
        crit_rows = [["Critere", "Note", "Niveau atteint"]]
        for crit in ax["criteria"]:
            anchor = _anchor_for(ax["axis"], crit["name"], crit["score"])
            crit_rows.append([crit["name"], f"{crit['score']}/5", anchor])
        crit_table = Table(crit_rows, colWidths=[55 * mm, 15 * mm, 90 * mm])
        crit_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(RED)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor(LINE)),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(LIGHT)]),
            ("ALIGN", (1, 1), (1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        block_elements.append(crit_table)
        if rat.get("strengths"):
            block_elements.append(Paragraph("<b>Points forts</b>", small))
            for s in rat["strengths"]:
                block_elements.append(Paragraph(f"- {s}", small))
        if rat.get("gaps"):
            block_elements.append(Paragraph("<b>Points faibles</b>", small))
            for g in rat["gaps"]:
                block_elements.append(Paragraph(f"- {g}", small))
        if rat.get("advice"):
            block_elements.append(Paragraph(f"<i>Conseil : {rat['advice']}</i>", small))
        block_elements.append(Spacer(1, 6))
        elements.append(KeepTogether(block_elements))

    elements.append(Paragraph(
        "<font size=8 color='#6B7280'>Document genere automatiquement par CDC LAUNCHPAD. "
        "Decision finale soumise a la validation du comite.</font>", body))
    doc.build(elements)
    buffer.seek(0)
    return buffer


def fmva_pdf(
    payload: dict[str, Any],
    fmva: dict[str, Any],
    result: dict[str, Any],
    overall: dict[str, Any] | None = None,
) -> io.BytesIO:
    """Render the FMVA valuation as an argued PDF report."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=15 * mm, bottomMargin=15 * mm,
        leftMargin=16 * mm, rightMargin=16 * mm,
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("FTitle", parent=styles["Title"], textColor=colors.HexColor(NAVY))
    h2 = ParagraphStyle("FH2", parent=styles["Heading2"], textColor=colors.HexColor(NAVY))
    body = styles["BodyText"]
    small = ParagraphStyle("Sm", parent=body, fontSize=8, textColor=colors.HexColor(MUTED))
    elements: list[Any] = []

    if os.path.exists(LOGO_FILE):
        try:
            elements.append(Image(LOGO_FILE, width=42 * mm, height=17 * mm))
            elements.append(Spacer(1, 6))
        except Exception:
            pass
    elements.append(Paragraph("Rapport de valorisation FMVA", title))
    elements.append(Paragraph(
        f"<b>{payload.get('name', '')}</b> | {payload.get('sector', '')} | "
        f"{dt.date.today():%d %b %Y}", body))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(
        f"<b>Ensemble (USD)</b> : ${result['ensemble_usd']:,.0f} "
        f"(<b>{result['ensemble_tnd']:,.0f} TND</b>)", h2))
    elements.append(Paragraph(
        f"Fourchette des 5 methodes : ${result['low_usd']:,.0f} - "
        f"${result['high_usd']:,.0f} | IQR {result['iqr_ratio']:.0%} | "
        f"{'REVUE REQUISE' if result['review_flag'] else 'Convergent'}", body))
    if overall:
        elements.append(Paragraph(
            f"<font color='{overall['color']}'><b>Recommandation : "
            f"{overall['action']}</b></font>", h2))
        for line in overall.get("rationale", []):
            elements.append(Paragraph(f"- {line}", body))
        elements.append(Spacer(1, 6))

    rows = [["Methode", "USD", "TND", "Poids"]]
    for m, v in result["methods_usd"].items():
        rows.append([m, f"${v:,.0f}",
                     f"{result['methods_tnd'][m]:,.0f}",
                     f"{ENSEMBLE_WEIGHTS[m]:.0%}"])
    rows.append(["ENSEMBLE", f"${result['ensemble_usd']:,.0f}",
                 f"{result['ensemble_tnd']:,.0f}", "100%"])
    table = Table(rows, colWidths=[55 * mm, 35 * mm, 35 * mm, 22 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(LINE)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor(LIGHT)]),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor(RED)),
        ("TEXTCOLOR", (0, -1), (-1, -1), colors.white),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("Detail par methode", h2))
    for m, v in result["methods_usd"].items():
        elements.append(Paragraph(f"<b>{m}</b>", body))
        elements.append(Paragraph(method_rationale(m, v, fmva, result), body))
        elements.append(Spacer(1, 4))

    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "<font size=8 color='#6B7280'>Document genere automatiquement par CDC LAUNCHPAD. "
        "Hypotheses a re-tester par le comite avant decision.</font>", body))
    doc.build(elements)
    buffer.seek(0)
    return buffer


# ---------------------------------------------------------------------------
# Financial-statement analyser - parses an uploaded P&L / balance sheet
# and produces a structured investment memo.
# ---------------------------------------------------------------------------
FIN_KEYWORDS: dict[str, list[str]] = {
    "revenue": ["revenue", "sales", "chiffre d'affaires", "ca net", "turnover", "produit"],
    "cogs": ["cogs", "cost of goods", "cost of sales", "cout des ventes", "achats consommes"],
    "gross_profit": ["gross profit", "marge brute"],
    "opex": ["operating expenses", "opex", "charges d'exploitation", "operating cost"],
    "ebitda": ["ebitda"],
    "ebit": ["ebit", "operating income", "resultat d'exploitation"],
    "net_income": ["net income", "net profit", "resultat net", "benefice net"],
    "interest_expense": ["interest expense", "charges financieres", "interest paid"],
    "tax": ["income tax", "impot sur les benefices", "impot societes"],
    "depreciation": ["depreciation", "amortissement", "amortization"],
    "total_assets": ["total assets", "total actif"],
    "current_assets": ["current assets", "actif circulant", "actif courant"],
    "cash": ["cash and equivalents", "cash", "tresorerie", "disponibilites"],
    "receivables": ["accounts receivable", "creances clients", "trade receivables"],
    "inventory": ["inventory", "stocks", "inventaire"],
    "current_liabilities": ["current liabilities", "passif circulant", "passif courant", "dettes court terme"],
    "total_liabilities": ["total liabilities", "total passif", "dettes totales"],
    "equity": ["total equity", "shareholders equity", "capitaux propres", "fonds propres"],
    "long_term_debt": ["long term debt", "dettes long terme", "dettes financieres", "non current debt"],
}


def parse_financial_statement(file_bytes: bytes, filename: str) -> dict[str, Any]:
    """Read an uploaded financial statement and extract the most recent year's values."""
    name = filename.lower()
    bio = io.BytesIO(file_bytes)
    if name.endswith((".xls", ".xlsx", ".xlsm")):
        try:
            df = pd.read_excel(bio, header=None)
        except Exception:
            return {"ok": False, "error": "Excel illisible. Verifier le format."}
    elif name.endswith(".csv"):
        try:
            df = pd.read_csv(bio, header=None, sep=None, engine="python")
        except Exception:
            return {"ok": False, "error": "CSV illisible."}
    else:
        return {"ok": False, "error": "Format non supporte (utiliser xlsx ou csv)."}

    # Drop fully empty rows/cols
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if df.empty:
        return {"ok": False, "error": "Fichier vide."}

    # The label column is the first column with mostly strings, the numeric columns are the rest.
    label_col_idx = 0
    for col in df.columns:
        vals = df[col].astype(str).str.lower()
        if vals.apply(lambda v: any(ch.isalpha() for ch in v)).sum() > len(df) // 2:
            label_col_idx = col
            break

    extracted: dict[str, float] = {}
    label_series = df[label_col_idx].astype(str).str.lower()
    for row_i, label in enumerate(label_series):
        norm = re.sub(r"[^a-z0-9 ]+", " ", label).strip()
        for key, kws in FIN_KEYWORDS.items():
            if key in extracted:
                continue
            for kw in kws:
                if kw in norm:
                    numeric_cells: list[float] = []
                    for col in df.columns:
                        if col == label_col_idx:
                            continue
                        val = _money_to_float(df.iloc[row_i][col])
                        if not pd.isna(val):
                            numeric_cells.append(val)
                    if numeric_cells:
                        extracted[key] = numeric_cells[-1]
                    break

    if not extracted:
        return {"ok": False, "error": "Aucune ligne reconnue. Renommer les libelles selon les standards."}

    # Fill derivations
    if "gross_profit" not in extracted and "revenue" in extracted and "cogs" in extracted:
        extracted["gross_profit"] = extracted["revenue"] - abs(extracted["cogs"])
    if "ebit" not in extracted and "ebitda" in extracted and "depreciation" in extracted:
        extracted["ebit"] = extracted["ebitda"] - abs(extracted["depreciation"])

    return {"ok": True, "extracted": extracted, "n_rows": int(len(df))}


def _safe_div(num: float, den: float) -> float | None:
    if den is None or den == 0 or pd.isna(den):
        return None
    return num / den


def compute_financial_ratios(data: dict[str, float]) -> dict[str, dict[str, Any]]:
    """Return a dict {ratio_name: {value, flag, label, explanation}}."""
    revenue = data.get("revenue", 0.0)
    gp = data.get("gross_profit")
    ebitda = data.get("ebitda")
    ebit = data.get("ebit")
    ni = data.get("net_income")
    interest = abs(data.get("interest_expense", 0.0))
    total_assets = data.get("total_assets")
    current_assets = data.get("current_assets")
    cash = data.get("cash")
    inventory = data.get("inventory", 0.0)
    current_liab = data.get("current_liabilities")
    total_liab = data.get("total_liabilities")
    equity = data.get("equity")
    lt_debt = data.get("long_term_debt", 0.0)

    def flag(v: float | None, good: float, warn: float, higher_is_better: bool = True) -> str:
        if v is None or pd.isna(v):
            return "na"
        if higher_is_better:
            if v >= good: return "ok"
            if v >= warn: return "warn"
            return "bad"
        else:
            if v <= good: return "ok"
            if v <= warn: return "warn"
            return "bad"

    gm = _safe_div(gp, revenue) if gp is not None else None
    em = _safe_div(ebitda, revenue) if ebitda is not None else None
    nm = _safe_div(ni, revenue) if ni is not None else None
    roa = _safe_div(ni, total_assets) if (ni is not None and total_assets) else None
    roe = _safe_div(ni, equity) if (ni is not None and equity) else None
    current = _safe_div(current_assets, current_liab) if (current_assets and current_liab) else None
    quick = (
        _safe_div((current_assets or 0) - inventory, current_liab)
        if (current_assets and current_liab) else None
    )
    cash_r = _safe_div(cash, current_liab) if (cash is not None and current_liab) else None
    de = _safe_div((lt_debt + (total_liab or 0) - (current_liab or 0) if total_liab else lt_debt),
                   equity) if equity else None
    debt_to_assets = _safe_div(total_liab, total_assets) if (total_liab and total_assets) else None
    int_cov = _safe_div(ebit, interest) if (ebit is not None and interest) else None
    asset_turn = _safe_div(revenue, total_assets) if total_assets else None

    return {
        "Marge brute": {
            "value": gm, "fmt": "pct",
            "flag": flag(gm, 0.35, 0.20),
            "explanation": "Capacite de la startup a transformer chaque dinar de CA en marge avant frais d'exploitation.",
        },
        "Marge EBITDA": {
            "value": em, "fmt": "pct",
            "flag": flag(em, 0.15, 0.05),
            "explanation": "Rentabilite operationnelle apres charges d'exploitation, hors elements financiers et fiscaux.",
        },
        "Marge nette": {
            "value": nm, "fmt": "pct",
            "flag": flag(nm, 0.10, 0.0),
            "explanation": "Resultat net rapporte au chiffre d'affaires - rentabilite finale apres impots.",
        },
        "ROA": {
            "value": roa, "fmt": "pct",
            "flag": flag(roa, 0.08, 0.02),
            "explanation": "Rendement des actifs - efficacite d'utilisation du bilan pour generer du resultat.",
        },
        "ROE": {
            "value": roe, "fmt": "pct",
            "flag": flag(roe, 0.15, 0.05),
            "explanation": "Rendement des fonds propres - effet de levier financier et rentabilite pour l'actionnaire.",
        },
        "Current ratio": {
            "value": current, "fmt": "x",
            "flag": flag(current, 1.5, 1.0),
            "explanation": "Liquidite generale - couverture des dettes court terme par l'actif circulant.",
        },
        "Quick ratio": {
            "value": quick, "fmt": "x",
            "flag": flag(quick, 1.0, 0.7),
            "explanation": "Liquidite restrictive (hors stocks) - capacite a payer le passif court terme rapidement.",
        },
        "Cash ratio": {
            "value": cash_r, "fmt": "x",
            "flag": flag(cash_r, 0.5, 0.2),
            "explanation": "Liquidite stricte - tresorerie / passif court terme.",
        },
        "Dette / Fonds propres": {
            "value": de, "fmt": "x",
            "flag": flag(de, 1.0, 2.0, higher_is_better=False),
            "explanation": "Levier financier - dettes financieres rapportees aux capitaux propres.",
        },
        "Dette / Actif total": {
            "value": debt_to_assets, "fmt": "pct",
            "flag": flag(debt_to_assets, 0.40, 0.60, higher_is_better=False),
            "explanation": "Poids global de l'endettement dans le bilan.",
        },
        "Couverture interets": {
            "value": int_cov, "fmt": "x",
            "flag": flag(int_cov, 3.0, 1.5),
            "explanation": "EBIT / charges financieres - capacite de servir la dette par le resultat operationnel.",
        },
        "Rotation actifs": {
            "value": asset_turn, "fmt": "x",
            "flag": flag(asset_turn, 1.0, 0.4),
            "explanation": "Chiffre d'affaires / actif total - intensite d'utilisation du bilan.",
        },
    }


def _format_ratio(value: float | None, fmt: str) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    if fmt == "pct":
        return f"{value*100:.1f}%"
    return f"{value:.2f}x"


def financial_memo(data: dict[str, float], ratios: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Generate an argued investment memo: thesis, strengths, risks, recommendation."""
    score = 0
    weight_sum = 0
    weights = {
        "Marge brute": 2.0, "Marge EBITDA": 3.0, "Marge nette": 2.0,
        "ROA": 1.5, "ROE": 1.5,
        "Current ratio": 1.5, "Quick ratio": 1.5, "Cash ratio": 1.0,
        "Dette / Fonds propres": 1.5, "Dette / Actif total": 1.0,
        "Couverture interets": 2.0, "Rotation actifs": 1.0,
    }
    for name, r in ratios.items():
        if r["flag"] == "na":
            continue
        w = weights.get(name, 1.0)
        weight_sum += w
        if r["flag"] == "ok":
            score += w * 1.0
        elif r["flag"] == "warn":
            score += w * 0.5
    health = (score / weight_sum) if weight_sum else 0.0

    strengths = [f"{n} : {_format_ratio(r['value'], r['fmt'])}"
                 for n, r in ratios.items() if r["flag"] == "ok"]
    risks = [f"{n} : {_format_ratio(r['value'], r['fmt'])}"
             for n, r in ratios.items() if r["flag"] == "bad"]
    watch = [f"{n} : {_format_ratio(r['value'], r['fmt'])}"
             for n, r in ratios.items() if r["flag"] == "warn"]

    rev = data.get("revenue", 0.0)
    thesis: list[str] = []
    if rev > 0:
        thesis.append(
            f"Activite generant {rev:,.0f} de chiffre d'affaires sur le dernier "
            "exercice rapporte."
        )
    em = ratios["Marge EBITDA"]["value"]
    if em is not None:
        if em >= 0.15:
            thesis.append(f"Marge EBITDA de {em*100:.1f}% indique une rentabilite operationnelle solide.")
        elif em >= 0:
            thesis.append(f"Marge EBITDA de {em*100:.1f}% confirme une rentabilite operationnelle naissante.")
        else:
            thesis.append(f"EBITDA negatif ({em*100:.1f}%) - rentabilite operationnelle non encore atteinte.")
    cur = ratios["Current ratio"]["value"]
    if cur is not None:
        if cur >= 1.5:
            thesis.append(f"Liquidite confortable (current ratio {cur:.2f}x).")
        elif cur >= 1.0:
            thesis.append(f"Liquidite juste (current ratio {cur:.2f}x), a surveiller.")
        else:
            thesis.append(f"Risque de liquidite (current ratio {cur:.2f}x).")
    de = ratios["Dette / Fonds propres"]["value"]
    if de is not None:
        if de <= 1.0:
            thesis.append(f"Structure capitalistique saine (D/E {de:.2f}x).")
        elif de <= 2.0:
            thesis.append(f"Levier modere (D/E {de:.2f}x), gerable.")
        else:
            thesis.append(f"Levier eleve (D/E {de:.2f}x) - risque de service de la dette.")

    if health >= 0.7:
        action = "INVESTIR"
        tone = "ok"
        color = GREEN
        rationale = "Profil financier solide - rentabilite, liquidite et structure capitalistique convergent positivement."
    elif health >= 0.45:
        action = "INVESTIR SOUS CONDITIONS"
        tone = "warn"
        color = AMBER
        rationale = "Profil mixte - convaincant sur certains axes mais avec des fragilites a documenter et a corriger avant decaissement."
    else:
        action = "NE PAS INVESTIR"
        tone = "bad"
        color = RED
        rationale = "Profil financier insuffisant a ce stade - les fragilites identifiees portent sur des axes critiques (rentabilite, liquidite ou structure)."

    next_steps: list[str] = []
    if action.startswith("INVESTIR S"):
        next_steps.append("Demander un comparable sectoriel pour valider les marges.")
        if watch or risks:
            for r in (risks + watch)[:3]:
                next_steps.append(f"Documenter et plan d'action sur {r.split(' :')[0]}.")
    elif action.startswith("INVESTIR"):
        next_steps.append("Confirmer en due diligence approfondie (juridique, fiscale, sociale).")
        next_steps.append("Negocier les covenants en lien avec les ratios cles.")
    else:
        next_steps.append("Notifier le porteur avec analyse motivee.")
        next_steps.append("Proposer une orientation vers un accompagnement de restructuration.")

    return {
        "health_score": health,
        "action": action,
        "tone": tone,
        "color": color,
        "thesis": thesis,
        "strengths": strengths,
        "watch": watch,
        "risks": risks,
        "rationale": rationale,
        "next_steps": next_steps,
    }


def financial_pdf(payload: dict[str, Any], data: dict[str, float],
                  ratios: dict[str, dict[str, Any]], memo: dict[str, Any]) -> io.BytesIO:
    """Generate a structured investment memo PDF."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=15 * mm, bottomMargin=15 * mm,
        leftMargin=16 * mm, rightMargin=16 * mm,
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("FinTitle", parent=styles["Title"], textColor=colors.HexColor(NAVY))
    h2 = ParagraphStyle("FinH2", parent=styles["Heading2"], textColor=colors.HexColor(NAVY))
    body = styles["BodyText"]
    small = ParagraphStyle("FinSm", parent=body, fontSize=8, textColor=colors.HexColor(MUTED))
    elements: list[Any] = []

    if os.path.exists(LOGO_FILE):
        try:
            elements.append(Image(LOGO_FILE, width=42 * mm, height=17 * mm))
            elements.append(Spacer(1, 6))
        except Exception:
            pass
    elements.append(Paragraph("Memo d'investissement - analyse financiere", title))
    elements.append(Paragraph(
        f"<b>{payload.get('name', '')}</b> | {payload.get('sector', '')} | "
        f"{dt.date.today():%d %b %Y}", body))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        f"<font color='{memo['color']}'><b>Recommandation : {memo['action']}</b></font>", h2))
    elements.append(Paragraph(
        f"Score de sante financiere : <b>{memo['health_score']*100:.0f}/100</b>. {memo['rationale']}",
        body))
    elements.append(Spacer(1, 6))

    if memo["thesis"]:
        elements.append(Paragraph("These", h2))
        for t in memo["thesis"]:
            elements.append(Paragraph(f"- {t}", body))
        elements.append(Spacer(1, 6))

    elements.append(Paragraph("Tableau des ratios", h2))
    rows = [["Ratio", "Valeur", "Drapeau", "Lecture"]]
    flag_label = {"ok": "Vert", "warn": "Orange", "bad": "Rouge", "na": "n/d"}
    for name, r in ratios.items():
        rows.append([
            name,
            _format_ratio(r["value"], r["fmt"]),
            flag_label[r["flag"]],
            r["explanation"],
        ])
    table = Table(rows, colWidths=[40 * mm, 25 * mm, 22 * mm, 80 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(LINE)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(LIGHT)]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 8))

    if memo["strengths"]:
        elements.append(Paragraph("Points forts", h2))
        for s in memo["strengths"]:
            elements.append(Paragraph(f"- {s}", body))
        elements.append(Spacer(1, 4))
    if memo["watch"]:
        elements.append(Paragraph("Points de vigilance", h2))
        for w in memo["watch"]:
            elements.append(Paragraph(f"- {w}", body))
        elements.append(Spacer(1, 4))
    if memo["risks"]:
        elements.append(Paragraph("Risques materiels", h2))
        for r in memo["risks"]:
            elements.append(Paragraph(f"- {r}", body))
        elements.append(Spacer(1, 4))

    elements.append(Paragraph("Prochaines etapes", h2))
    for step in memo["next_steps"]:
        elements.append(Paragraph(f"- {step}", body))

    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "<font size=8 color='#6B7280'>Analyse generee automatiquement a partir des etats "
        "financiers transmis. Document de support a la decision d'investissement, soumis a "
        "validation par le comite.</font>", body))
    doc.build(elements)
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
# ---------------------------------------------------------------------------
# Interactive Plotly visualisations
# ---------------------------------------------------------------------------
def _plotly_layout(fig: Any, height: int = 320) -> Any:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=INK, size=12),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter"),
    )
    return fig


def plotly_radar(scorecard: dict[str, Any]) -> Any:
    """Radar / spider chart of the 7 axis notes for the committee view."""
    import plotly.graph_objects as go

    axes = [ax["axis"] for ax in scorecard["axes"]]
    notes = [ax["note"] for ax in scorecard["axes"]]
    short = [a if len(a) < 26 else a[:24] + "..." for a in axes]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=notes + [notes[0]],
        theta=short + [short[0]],
        fill="toself",
        line=dict(color=RED, width=2.5),
        fillcolor=f"rgba(209,10,17,0.18)",
        name="Notes par axe",
        hovertemplate="<b>%{theta}</b><br>Note : %{r}/5<extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[3.5] * (len(notes) + 1),
        theta=short + [short[0]],
        line=dict(color=GREEN, width=1, dash="dot"),
        name="Seuil 3.5",
        hoverinfo="skip",
        showlegend=True,
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(247,248,252,0.6)",
            radialaxis=dict(visible=True, range=[0, 5], tickfont=dict(size=10, color=MUTED),
                            gridcolor=LINE, linecolor=LINE),
            angularaxis=dict(tickfont=dict(size=11, color=NAVY), gridcolor=LINE,
                             linecolor=LINE),
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center", font=dict(size=11)),
    )
    return _plotly_layout(fig, height=420)


def plotly_treemap(series: pd.Series, title: str = "") -> Any:
    """Sector or region treemap with brand gradient."""
    import plotly.express as px

    df_in = series.reset_index()
    df_in.columns = ["label", "value"]
    fig = px.treemap(
        df_in, path=["label"], values="value",
        color="value",
        color_continuous_scale=[
            [0.0, "#F4F5F9"], [0.3, "#7C8BC9"], [0.6, NAVY], [1.0, RED],
        ],
        custom_data=["value"],
    )
    fig.update_traces(
        textinfo="label+value", textfont=dict(color="white", size=13, family="Inter"),
        hovertemplate="<b>%{label}</b><br>%{customdata[0]} startups<extra></extra>",
        marker=dict(line=dict(color="white", width=2)),
    )
    fig.update_layout(coloraxis_showscale=False, title=dict(text=title, font=dict(size=13)))
    return _plotly_layout(fig, height=360)


def plotly_donut_methods(result: dict[str, Any]) -> Any:
    """Donut of FMVA method contribution to the ensemble (weighted USD)."""
    import plotly.graph_objects as go

    methods = list(result["methods_usd"].keys())
    weighted = [result["methods_usd"][m] * ENSEMBLE_WEIGHTS[m] for m in methods]
    palette = [NAVY, RED, GOLD, TEAL, VIOLET]
    fig = go.Figure(data=[go.Pie(
        labels=methods, values=weighted, hole=0.62,
        marker=dict(colors=palette, line=dict(color="white", width=3)),
        textfont=dict(color="white", size=12, family="Inter"),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Contribution : $%{value:,.0f}<br>%{percent}<extra></extra>",
    )])
    fig.add_annotation(
        text=f"<b>${result['ensemble_usd']/1e6:.2f}M</b><br><span style='font-size:11px;color:{MUTED}'>Ensemble (USD)</span>",
        x=0.5, y=0.5, showarrow=False, font=dict(size=18, color=NAVY),
    )
    fig.update_layout(showlegend=False)
    return _plotly_layout(fig, height=360)


def plotly_method_bars(result: dict[str, Any]) -> Any:
    """Horizontal comparison of the 5 FMVA methods + ensemble line."""
    import plotly.graph_objects as go

    methods = list(result["methods_usd"].keys())
    values = list(result["methods_usd"].values())
    palette = [NAVY, RED, GOLD, TEAL, VIOLET]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values, y=methods, orientation="h",
        marker=dict(color=palette, line=dict(color="white", width=1.5)),
        text=[f"${v/1e6:.2f}M" for v in values], textposition="outside",
        hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>",
    ))
    fig.add_vline(
        x=result["ensemble_usd"], line=dict(color=RED, dash="dash", width=2),
        annotation_text=f"Ensemble ${result['ensemble_usd']/1e6:.2f}M",
        annotation_position="top right",
        annotation_font=dict(color=RED, size=11),
    )
    fig.update_xaxes(gridcolor=LINE, showgrid=True, zeroline=False, tickformat="$,.0s")
    fig.update_yaxes(showgrid=False)
    return _plotly_layout(fig, height=340)


# Approximate centroids of the 24 Tunisian governorates (lat, lon).
TN_GOVERNORATE_CENTROIDS: dict[str, tuple[float, float]] = {
    "Tunis": (36.8065, 10.1815),
    "Ariana": (36.8625, 10.1956),
    "Ben Arous": (36.7472, 10.2292),
    "Manouba": (36.8101, 10.0967),
    "Nabeul": (36.4561, 10.7376),
    "Zaghouan": (36.4028, 10.1428),
    "Bizerte": (37.2744, 9.8739),
    "Beja": (36.7256, 9.1817),
    "Jendouba": (36.5011, 8.7800),
    "Le Kef": (36.1740, 8.7050),
    "Siliana": (36.0833, 9.3667),
    "Sousse": (35.8254, 10.6360),
    "Monastir": (35.7770, 10.8266),
    "Mahdia": (35.5050, 11.0622),
    "Kairouan": (35.6781, 10.0964),
    "Kasserine": (35.1671, 8.8364),
    "Sidi Bouzid": (35.0381, 9.4848),
    "Sfax": (34.7406, 10.7603),
    "Gabes": (33.8814, 10.0982),
    "Medenine": (33.3548, 10.5055),
    "Tataouine": (32.9297, 10.4518),
    "Gafsa": (34.4250, 8.7842),
    "Tozeur": (33.9197, 8.1335),
    "Kebili": (33.7050, 8.9692),
}


def plotly_tunisia_map(region_counts: pd.Series, df: pd.DataFrame | None = None) -> Any:
    """Bubble map of Tunisia: one bubble per governorate, sized by startup count."""
    import plotly.graph_objects as go

    rows: list[dict[str, Any]] = []
    for region, count in region_counts.items():
        if not isinstance(region, str):
            continue
        coords = TN_GOVERNORATE_CENTROIDS.get(region.strip())
        if not coords:
            continue
        funded = 0
        top_sector = ""
        if df is not None and "Region" in df.columns:
            sub = df[df["Region"].astype(str) == region.strip()]
            if "funded" in sub.columns:
                funded = int(sub["funded"].sum())
            if "sector" in sub.columns and not sub["sector"].dropna().empty:
                top_sector = str(sub["sector"].value_counts().head(1).index[0])
        rows.append({
            "region": region.strip(),
            "lat": coords[0], "lon": coords[1],
            "count": int(count), "funded": funded, "top_sector": top_sector,
        })
    if not rows:
        return None
    plot_df = pd.DataFrame(rows)
    max_c = max(1, plot_df["count"].max())
    plot_df["size"] = 12 + (plot_df["count"] / max_c) * 42

    fig = go.Figure(go.Scattergeo(
        lon=plot_df["lon"], lat=plot_df["lat"],
        text=plot_df["region"],
        customdata=plot_df[["count", "funded", "top_sector"]].values,
        mode="markers+text",
        textposition="top center",
        textfont=dict(size=10, color=NAVY, family="Inter"),
        marker=dict(
            size=plot_df["size"], color=plot_df["count"],
            colorscale=[[0.0, "#7C8BC9"], [0.5, NAVY], [1.0, RED]],
            line=dict(color="white", width=1.5),
            opacity=0.92, showscale=False,
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Startups : %{customdata[0]}<br>"
            "Finances : %{customdata[1]}<br>"
            "Top secteur : %{customdata[2]}<extra></extra>"
        ),
    ))
    fig.update_geos(
        scope="africa",
        center=dict(lat=34.7, lon=9.5),
        projection_scale=8.5,
        showcountries=True, countrycolor="#D9DCE6",
        showcoastlines=True, coastlinecolor="#A1A8C9",
        showland=True, landcolor="#F8F9FC",
        showocean=True, oceancolor="#EAF1FA",
        showframe=False,
        fitbounds=False,
        lataxis=dict(range=[30.0, 38.0]),
        lonaxis=dict(range=[7.0, 12.0]),
    )
    return _plotly_layout(fig, height=480)


def plotly_yearly_sparkline(years: pd.Series, color: str = NAVY) -> Any:
    """Tiny line chart for a KPI card (founding-year cadence)."""
    import plotly.graph_objects as go

    counts = years.value_counts().sort_index().tail(12)
    fig = go.Figure(go.Scatter(
        x=list(counts.index), y=list(counts.values),
        mode="lines", line=dict(color=color, width=2.5, shape="spline"),
        fill="tozeroy", fillcolor=f"rgba(39,46,95,0.10)",
        hovertemplate="%{x} : %{y} startups<extra></extra>",
    ))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=70)
    return fig


# ---------------------------------------------------------------------------
# Beneficiary spotlight - real funded startups from the CDC dataset
# ---------------------------------------------------------------------------
def beneficiary_spotlight(df: pd.DataFrame, k: int = 4) -> list[dict[str, Any]]:
    """Pick a handful of funded beneficiaries with the richest profile to feature."""
    if df is None or df.empty:
        return []
    funded = df[df.get("funded", 0) == 1].copy() if "funded" in df.columns else df.copy()
    if funded.empty:
        funded = df.copy()
    completeness = pd.Series(0, index=funded.index)
    for col in ["sector", "Region", "founding_year", "has_web", "has_email", "is_labelled"]:
        if col in funded.columns:
            completeness = completeness + funded[col].notna().astype(int)
    funded = funded.assign(_score=completeness)
    funded = funded.sort_values(["_score", "founding_year"], ascending=[False, False])

    palette = ["navy", "red", "gold", "teal", "violet", "rose"]
    out: list[dict[str, Any]] = []
    seen_sectors: set[str] = set()
    for _, row in funded.iterrows():
        sector = str(row.get("sector", "")).strip() or "Tech"
        if sector in seen_sectors and len(out) < k - 1:
            continue
        seen_sectors.add(sector)
        try:
            year = int(row.get("founding_year")) if pd.notna(row.get("founding_year")) else None
        except Exception:
            year = None
        out.append({
            "name": str(row.get("Nom", "Startup CDC")).strip() or "Startup CDC",
            "sector": sector,
            "region": str(row.get("Region", "")).strip() or "Tunisie",
            "year": year,
            "labelled": bool(row.get("is_labelled", 0)),
            "color": palette[len(out) % len(palette)],
        })
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------------------
# Programs and Newsroom tabs - curated content from public CDC ecosystem
# ---------------------------------------------------------------------------
CDC_PROGRAMS: list[dict[str, Any]] = [
    {
        "name": "Anava - Fund of Funds",
        "operator": "Smart Capital",
        "url": "https://smartcapital.tn/",
        "budget": "Up to 200 MDT",
        "period": "2020 - ongoing",
        "stage": "Seed to growth",
        "summary_en": (
            "First Tunisian fund-of-funds dedicated to startups. Anava invests in "
            "private VC funds that back labelled startups, multiplying the firepower "
            "of the public-private partnership."
        ),
        "summary_fr": (
            "Premier fonds de fonds tunisien dedie aux startups. Anava investit "
            "dans des fonds de capital risque prives qui financent les startups "
            "labellisees, demultipliant la force de frappe du PPP."
        ),
        "partners": ["Smart Capital", "World Bank", "AFD", "KfW"],
        "highlights_en": [
            "Multi-stage coverage from seed to series B",
            "Crowds in international LPs alongside CDC",
            "Targets locally registered Tunisian funds",
        ],
        "highlights_fr": [
            "Couverture multi-stade du seed a la serie B",
            "Mobilise des LPs internationaux aux cotes de la CDC",
            "Cible les fonds tunisiens enregistres localement",
        ],
        "tone": "navy",
    },
    {
        "name": "VAIR Greentech",
        "operator": "CDC + Smart Capital",
        "url": "https://smartcapital.tn/",
        "budget": "Repayable advance per startup",
        "period": "2024 - ongoing",
        "stage": "PoC, TRL 1-3",
        "summary_en": (
            "Repayable advances to finance the proof-of-concept stage of greentech "
            "startups. The committee scores 7 axes (innovation, market, team, PoC, "
            "budget, impact, risk) before any commitment."
        ),
        "summary_fr": (
            "Avances remboursables pour financer le proof-of-concept des startups "
            "greentech. Le comite note 7 axes (innovation, marche, equipe, PoC, "
            "budget, impact, risque) avant tout engagement."
        ),
        "partners": ["CDC Tunisie", "Smart Capital", "Startup Tunisia"],
        "highlights_en": [
            "Focused on climate, energy, water, circular economy",
            "Repayment indexed on revenue trajectory",
            "Embedded in this very platform's scoring engine",
        ],
        "highlights_fr": [
            "Cible climat, energie, eau, economie circulaire",
            "Remboursement indexe sur la trajectoire de revenus",
            "Integre au moteur de scoring de cette plateforme",
        ],
        "tone": "red",
    },
    {
        "name": "Startup Act - the legal foundation",
        "operator": "Government of Tunisia",
        "url": "https://startup.gov.tn/",
        "budget": "Fiscal & regulatory framework",
        "period": "2018 (Law 2018-20) - ongoing",
        "stage": "All stages",
        "summary_en": (
            "Tunisia's flagship startup law: one-stop label, founder-friendly "
            "incentives (creation leave for salaried co-founders, capital guarantee, "
            "tax holiday) and access to foreign currency. Foundation of the ecosystem."
        ),
        "summary_fr": (
            "Loi phare des startups tunisiennes : label unique, incitations "
            "founder-friendly (conge creation pour les salaries co-fondateurs, "
            "garantie de capital, exoneration fiscale) et acces aux devises. "
            "Socle de tout l'ecosysteme."
        ),
        "partners": ["MTC", "APII", "CDC Tunisie", "Startup Tunisia"],
        "highlights_en": [
            "1100+ labels delivered since 2019",
            "Cross-ministry one-stop digital workflow",
            "Reviewed every 3 years",
        ],
        "highlights_fr": [
            "1100+ labels delivres depuis 2019",
            "Workflow numerique inter-ministeriel",
            "Revu tous les 3 ans",
        ],
        "tone": "navy",
    },
    {
        "name": "Innov'i - regional innovation hubs",
        "operator": "EU + CDC + Smart Capital",
        "url": "https://smartcapital.tn/",
        "budget": "12 MEUR programme",
        "period": "2020 - 2024",
        "stage": "Idea to early traction",
        "summary_en": (
            "Pre-acceleration and acceleration network deployed across multiple "
            "Tunisian regions, pairing local hubs with international mentorship. "
            "Goal: decentralise the ecosystem beyond Tunis."
        ),
        "summary_fr": (
            "Reseau de pre-acceleration et acceleration deploye sur plusieurs "
            "regions tunisiennes, associant des hubs locaux a du mentoring "
            "international. Objectif : decentraliser l'ecosysteme au-dela de Tunis."
        ),
        "partners": ["European Union", "Smart Capital", "Regional incubators"],
        "highlights_en": [
            "Regional reach beyond Greater Tunis",
            "Pre-acceleration + acceleration + investment readiness",
            "Co-financed by the EU delegation",
        ],
        "highlights_fr": [
            "Portee regionale au-dela du Grand Tunis",
            "Pre-acceleration + acceleration + preparation a la levee",
            "Cofinance par la delegation de l'UE",
        ],
        "tone": "red",
    },
    {
        "name": "Diaspora invest - Tunisians overseas",
        "operator": "GIZ + Smart Capital",
        "url": "https://smartcapital.tn/",
        "budget": "Sub-grant + accompaniment",
        "period": "2022 - ongoing",
        "stage": "Seed and growth",
        "summary_en": (
            "Targets Tunisian founders abroad considering a return-and-build, plus "
            "diaspora investors backing local startups. Provides matching grants "
            "and a network of in-country advisors."
        ),
        "summary_fr": (
            "Cible les fondateurs tunisiens a l'etranger envisageant un retour, "
            "ainsi que les investisseurs de la diaspora soutenant des startups "
            "locales. Subventions de matching et reseau d'advisors en Tunisie."
        ),
        "partners": ["GIZ", "Smart Capital", "Diaspora networks"],
        "highlights_en": [
            "Founder return programme + investor track",
            "Matching grants for diaspora-led rounds",
            "Strong cross-border pipeline",
        ],
        "highlights_fr": [
            "Programme retour fondateurs + volet investisseurs",
            "Subventions de matching pour les tours diaspora",
            "Pipeline transfrontalier solide",
        ],
        "tone": "navy",
    },
    {
        "name": "CDC direct equity tickets",
        "operator": "Caisse des Depots et Consignations",
        "url": "https://www.cdc.tn/",
        "budget": "Selective tickets",
        "period": "Ongoing",
        "stage": "Series A and beyond",
        "summary_en": (
            "Long-tenor equity participations made directly by the Caisse des "
            "Depots in scale-stage Tunisian companies with strategic national value "
            "(infrastructure, sovereignty, regional impact)."
        ),
        "summary_fr": (
            "Participations en capital long terme prises directement par la Caisse "
            "des Depots dans des entreprises tunisiennes en phase de scale a forte "
            "valeur strategique nationale (infrastructure, souverainete, impact regional)."
        ),
        "partners": ["CDC Tunisie", "Selected co-investors"],
        "highlights_en": [
            "Patient capital with 7-10 year horizons",
            "Co-invests with private VCs and DFIs",
            "Aligned with national strategic priorities",
        ],
        "highlights_fr": [
            "Capital patient sur 7-10 ans",
            "Co-investissements avec VCs prives et DFIs",
            "Aligne sur les priorites strategiques nationales",
        ],
        "tone": "red",
    },
]


def _render_programs_tab(lang: str) -> None:
    import streamlit as st

    is_fr = (lang == "FR")
    pill_label = "Nos Programmes" if is_fr else "Our Programs"
    title = (
        "Capital, accompagnement et legal - le stack public deploye par la CDC"
        if is_fr
        else "Capital, support and legal - the public stack deployed by CDC"
    )
    st.markdown(
        f"<div class='section-h'><span class='pill'>{pill_label}</span>"
        f"<h3>{title}</h3></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Chaque carte resume un programme actif, ses partenaires, son budget "
        "indicatif et ses points cles. Cliquez pour ouvrir la source officielle."
        if is_fr
        else "Each card summarises an active programme, its partners, indicative "
             "budget and key highlights. Click through for the official source."
    )

    cards_html = []
    for prog in CDC_PROGRAMS:
        c1, c2 = _TONE_GRADIENTS.get(prog["tone"], (NAVY, "#1B2150"))
        summary = prog["summary_fr"] if is_fr else prog["summary_en"]
        highlights = prog["highlights_fr"] if is_fr else prog["highlights_en"]
        partners_html = "".join(
            f"<span class='prog-chip'>{p}</span>" for p in prog["partners"]
        )
        hl_html = "".join(f"<li>{h}</li>" for h in highlights)
        labels = ("Budget", "Periode", "Stade") if is_fr else ("Budget", "Period", "Stage")
        cta = "Ouvrir la source" if is_fr else "Open source"
        cards_html.append(
            f"<div class='prog-card'>"
            f"  <div class='prog-cover' style='background:linear-gradient(135deg,{c1},{c2})'>"
            f"    <div class='prog-name'>{prog['name']}</div>"
            f"    <div class='prog-op'>{prog['operator']}</div>"
            f"  </div>"
            f"  <div class='prog-body'>"
            f"    <p class='prog-summary'>{summary}</p>"
            f"    <div class='prog-meta'>"
            f"      <div><span class='l'>{labels[0]}</span><span class='v'>{prog['budget']}</span></div>"
            f"      <div><span class='l'>{labels[1]}</span><span class='v'>{prog['period']}</span></div>"
            f"      <div><span class='l'>{labels[2]}</span><span class='v'>{prog['stage']}</span></div>"
            f"    </div>"
            f"    <div class='prog-partners'>{partners_html}</div>"
            f"    <ul class='prog-hl'>{hl_html}</ul>"
            f"    <a class='prog-link' href='{prog['url']}' target='_blank' rel='noopener'>{cta} &nbsp;&rsaquo;</a>"
            f"  </div>"
            f"</div>"
        )
    st.markdown(
        f"<div class='prog-grid'>{''.join(cards_html)}</div>",
        unsafe_allow_html=True,
    )


NEWSROOM_ARTICLES: list[dict[str, Any]] = [
    {
        "title_en": "Startup Act labels cross the 1,100 milestone",
        "title_fr": "Le label Startup Act franchit la barre des 1 100",
        "source": "startup.gov.tn",
        "date": "2025-09-12",
        "tag": "Policy",
        "url": "https://startup.gov.tn/",
        "summary_en": (
            "The one-stop labelling workflow has now issued more than 1,100 active "
            "Startup Act labels, with greentech and fintech absorbing most of the "
            "recent vintages."
        ),
        "summary_fr": (
            "Le guichet unique a delivre plus de 1 100 labels Startup Act actifs, "
            "les vintages recents etant largement absorbes par la greentech et la fintech."
        ),
        "color": "navy",
    },
    {
        "title_en": "Anava commits to a new VC fund focused on MENA scale-ups",
        "title_fr": "Anava engage un nouveau fonds VC dedie au scale-up MENA",
        "source": "smartcapital.tn",
        "date": "2025-08-23",
        "tag": "Capital",
        "url": "https://smartcapital.tn/",
        "summary_en": (
            "The fund-of-funds adds another tranche to a regional VC manager "
            "targeting series A and B opportunities across MENA, with a Tunisian "
            "allocation floor."
        ),
        "summary_fr": (
            "Le fonds de fonds ajoute une tranche a un manager regional cible "
            "series A et B au Maghreb-MENA, avec un plancher d'allocation pour la Tunisie."
        ),
        "color": "red",
    },
    {
        "title_en": "VAIR opens its second greentech window",
        "title_fr": "VAIR ouvre sa deuxieme fenetre greentech",
        "source": "smartcapital.tn",
        "date": "2025-07-04",
        "tag": "Programme",
        "url": "https://smartcapital.tn/",
        "summary_en": (
            "The repayable-advance programme reopens applications for greentech "
            "proofs of concept, with sharpened impact and execution criteria."
        ),
        "summary_fr": (
            "Le programme d'avances remboursables rouvre ses candidatures pour les "
            "proofs of concept greentech, avec des criteres d'impact et d'execution affines."
        ),
        "color": "navy",
    },
    {
        "title_en": "MENA VC pulse: Tunisia tightens its grip on greentech",
        "title_fr": "Pouls VC MENA : la Tunisie consolide sa position en greentech",
        "source": "wamda.com",
        "date": "2025-06-18",
        "tag": "Analysis",
        "url": "https://www.wamda.com/tags/tunisia",
        "summary_en": (
            "Regional reporting highlights a clustering of Tunisian greentech and "
            "fintech rounds, with stronger founder caliber and tighter unit "
            "economics than previous cohorts."
        ),
        "summary_fr": (
            "La presse regionale releve une concentration de tours greentech et "
            "fintech tunisiens, avec un caliber fondateur plus eleve et des unit "
            "economics plus serrees que les cohortes precedentes."
        ),
        "color": "red",
    },
    {
        "title_en": "Diaspora track: matched grants for return founders",
        "title_fr": "Volet diaspora : subventions de matching pour fondateurs returnees",
        "source": "smartcapital.tn",
        "date": "2025-05-30",
        "tag": "Programme",
        "url": "https://smartcapital.tn/",
        "summary_en": (
            "The diaspora pipeline pairs returning Tunisian founders with matched "
            "subgrants and local advisors, easing the soft-landing into the "
            "domestic ecosystem."
        ),
        "summary_fr": (
            "Le pipeline diaspora associe les fondateurs tunisiens de retour a des "
            "subventions de matching et des advisors locaux, facilitant l'atterrissage "
            "dans l'ecosysteme domestique."
        ),
        "color": "navy",
    },
    {
        "title_en": "Africa Report - Tunisia's startup engine quietly accelerates",
        "title_fr": "Africa Report - l'engin startup tunisien accelere en silence",
        "source": "theafricareport.com",
        "date": "2025-04-22",
        "tag": "Press",
        "url": "https://www.theafricareport.com/tag/tunisia/",
        "summary_en": (
            "Long-form coverage on the second wave of Tunisian founders building "
            "for the continent and the Gulf, supported by deepening public-private "
            "capital."
        ),
        "summary_fr": (
            "Long format sur la deuxieme vague de fondateurs tunisiens batissant "
            "pour le continent et le Golfe, soutenus par un capital public-prive "
            "qui s'approfondit."
        ),
        "color": "red",
    },
]


def _fmt_date(date_iso: str, lang: str) -> str:
    try:
        d = dt.date.fromisoformat(date_iso)
    except Exception:
        return date_iso
    if lang == "FR":
        months_fr = [
            "janv.", "fev.", "mars", "avr.", "mai", "juin",
            "juil.", "aout", "sept.", "oct.", "nov.", "dec.",
        ]
        return f"{d.day} {months_fr[d.month - 1]} {d.year}"
    return d.strftime("%d %b %Y")


def _render_newsroom_tab(lang: str) -> None:
    import streamlit as st

    is_fr = (lang == "FR")
    pill = "Veille" if is_fr else "Newsroom"
    title = (
        "Les signaux faibles et forts de l'ecosysteme tunisien et MENA"
        if is_fr
        else "Strong and weak signals from the Tunisian and MENA ecosystem"
    )
    st.markdown(
        f"<div class='section-h'><span class='pill' style='background:linear-gradient(135deg,{NAVY},{RED})'>{pill}</span>"
        f"<h3>{title}</h3></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Les cartes ci-dessous renvoient vers les sources officielles. Pour activer "
        "une veille temps reel, brancher un flux RSS via la configuration plateforme."
        if is_fr
        else "Cards link to the official sources. To activate a live feed, plug an "
             "RSS source through the platform configuration."
    )

    cards_html = []
    for art in NEWSROOM_ARTICLES:
        c1, c2 = _TONE_GRADIENTS.get(art["color"], (NAVY, "#1B2150"))
        title_text = art["title_fr"] if is_fr else art["title_en"]
        summary = art["summary_fr"] if is_fr else art["summary_en"]
        date_str = _fmt_date(art["date"], lang)
        read_cta = "Lire la source" if is_fr else "Read the source"
        cards_html.append(
            f"<div class='news-card'>"
            f"  <div class='news-cover' style='background:linear-gradient(135deg,{c1},{c2})'>"
            f"    <span class='news-tag'>{art['tag']}</span>"
            f"    <span class='news-date'>{date_str}</span>"
            f"  </div>"
            f"  <div class='news-body'>"
            f"    <div class='news-source'>{art['source']}</div>"
            f"    <h4>{title_text}</h4>"
            f"    <p>{summary}</p>"
            f"    <a href='{art['url']}' target='_blank' rel='noopener'>{read_cta} &nbsp;&rsaquo;</a>"
            f"  </div>"
            f"</div>"
        )
    st.markdown(
        f"<div class='news-grid'>{''.join(cards_html)}</div>",
        unsafe_allow_html=True,
    )


def _inject_css() -> None:
    import streamlit as st

    st.markdown(
        f"""
        <style>
        :root {{
            --navy: {NAVY};
            --red: {RED};
            --ink: {INK};
            --muted: {MUTED};
            --line: #E5E7EB;
            --light: #F8F9FC;
            --gold: {GOLD};
            --teal: {TEAL};
            --violet: {VIOLET};
            --rose: {ROSE};
            --blue: {BLUE};
            --amber: {AMBER};
            --green: {GREEN};
            --ink: {INK};
            --muted: {MUTED};
            --line: {LINE};
            --light: {LIGHT};
        }}
        .stApp {{
            background: #FFFFFF;
            color: {INK};
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #FAFBFE 0%, #FFFFFF 100%);
            border-right: 1px solid #EFF1F6;
        }}
        h1, h2, h3, h4 {{
            color: {NAVY};
            letter-spacing: -0.01em;
        }}
        .block-container {{
            padding-top: 1.0rem;
            padding-bottom: 2.2rem;
            max-width: 1440px;
        }}
        /* Hero - white card with brand gradient accent */
        .cdc-hero {{
            position: relative;
            overflow: hidden;
            border-radius: 22px;
            padding: 1.5rem 1.7rem;
            margin-bottom: 1.1rem;
            background:
              radial-gradient(900px 380px at 110% -20%, rgba(209,10,17,0.10), transparent 60%),
              radial-gradient(900px 380px at -10% 110%, rgba(39,46,95,0.10), transparent 60%),
              #FFFFFF;
            border: 1px solid #EEF0F6;
            box-shadow: 0 32px 60px -36px rgba(39,46,95,0.30);
        }}
        .cdc-hero::before {{
            content:''; position: absolute; left: 0; top: 0; bottom: 0; width: 6px;
            background: linear-gradient(180deg, {NAVY} 0%, {RED} 100%);
        }}
        /* Animated title */
        .cdc-title-anim {{
            font-size: clamp(2rem, 4.2vw, 3.4rem);
            font-weight: 900;
            line-height: 1;
            margin: 0;
            letter-spacing: -0.02em;
            background: linear-gradient(110deg, {NAVY} 10%, {RED} 35%, {NAVY} 60%, {RED} 85%);
            background-size: 220% 100%;
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            animation: cdcShine 7s linear infinite;
        }}
        .cdc-title-anim .pad {{
            display: inline-block; transform-origin: 50% 60%;
            animation: cdcFloat 4.5s ease-in-out infinite;
        }}
        @keyframes cdcShine {{
            0% {{ background-position: 0% 50%; }}
            100% {{ background-position: 220% 50%; }}
        }}
        @keyframes cdcFloat {{
            0%, 100% {{ transform: translateY(0) scale(1); }}
            50% {{ transform: translateY(-3px) scale(1.02); }}
        }}
        .cdc-tag-line {{
            font-size: 1.02rem; color: {INK};
            margin: 0.4rem 0 0.5rem 0; font-weight: 500;
            max-width: 720px;
        }}
        .cdc-tag-line .accent {{ color: {RED}; font-weight: 800; }}
        /* Quote rotator */
        .cdc-quote-wrap {{
            margin-top: 0.7rem; position: relative; min-height: 76px;
            border-top: 1px dashed #EEF0F6; padding-top: 0.7rem;
        }}
        .cdc-quote {{
            display: flex; align-items: center; gap: 0.85rem;
            opacity: 0; position: absolute; inset: 0.7rem 0 0 0;
            animation: cdcQuoteCycle 30s infinite;
        }}
        .cdc-quote:nth-child(1) {{ animation-delay: 0s; }}
        .cdc-quote:nth-child(2) {{ animation-delay: 6s; }}
        .cdc-quote:nth-child(3) {{ animation-delay: 12s; }}
        .cdc-quote:nth-child(4) {{ animation-delay: 18s; }}
        .cdc-quote:nth-child(5) {{ animation-delay: 24s; }}
        @keyframes cdcQuoteCycle {{
            0%, 18% {{ opacity: 0; transform: translateY(6px); }}
            2%, 16% {{ opacity: 1; transform: translateY(0); }}
            20%, 100% {{ opacity: 0; transform: translateY(-6px); }}
        }}
        .cdc-quote .avatar {{
            width: 44px; height: 44px; border-radius: 50%;
            display: inline-flex; align-items: center; justify-content: center;
            font-weight: 800; color: white; font-size: 0.95rem;
            box-shadow: 0 8px 22px -10px rgba(39,46,95,0.55);
            flex-shrink: 0;
        }}
        .cdc-quote .text {{ color: {INK}; font-size: 0.96rem; line-height: 1.35; font-style: italic; }}
        .cdc-quote .who {{
            color: {MUTED}; font-size: 0.78rem; margin-top: 0.15rem; font-style: normal; font-weight: 600;
        }}
        .cdc-hero-row {{
            position: relative; z-index: 2;
            display: grid; grid-template-columns: 230px 1fr; gap: 1.5rem; align-items: center;
        }}
        @media (max-width: 900px) {{
            .cdc-hero-row {{ grid-template-columns: 1fr; }}
        }}
        .cdc-hero .badges {{ margin-top: 0.45rem; display:flex; flex-wrap:wrap; gap: 0.4rem; }}
        .cdc-hero .badge {{
            display:inline-flex; align-items:center; gap:0.35rem;
            background: linear-gradient(135deg, rgba(39,46,95,0.06), rgba(209,10,17,0.06));
            border:1px solid #E5E7EB;
            color: {NAVY}; padding:0.28rem 0.7rem; border-radius:999px;
            font-size:0.78rem; font-weight:700;
            transition: transform 160ms ease, box-shadow 160ms ease;
        }}
        .cdc-hero .badge:hover {{
            transform: translateY(-1px);
            box-shadow: 0 10px 24px -16px rgba(39,46,95,0.5);
        }}
        .cdc-hero .badge .dot {{ width:7px; height:7px; border-radius:50%; background:{RED}; }}
        .cdc-hero-logo {{
            display:flex; align-items:center; justify-content:center;
            padding: 10px; border-radius: 18px;
            background: radial-gradient(circle at 30% 20%, #F6F7FB, #FFFFFF 70%);
            border:1px solid #EEF0F6;
            min-height: 170px;
        }}
        .cdc-hero-logo video, .cdc-hero-logo img {{
            width: 100%; max-width: 220px; max-height: 170px; height: auto; border-radius: 12px;
        }}
        /* KPI strip */
        .kpi-strip {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 0.7rem; margin: 0.4rem 0 1.1rem 0;
        }}
        .kpi {{
            position: relative;
            background: white; border:1px solid {LINE}; border-radius: 14px;
            padding: 0.9rem 1rem; min-height: 96px;
            box-shadow: 0 12px 30px -22px rgba(39,46,95,0.45);
            transition: transform 180ms ease, box-shadow 180ms ease;
            overflow: hidden;
        }}
        .kpi:hover {{
            transform: translateY(-3px);
            box-shadow: 0 20px 40px -22px rgba(39,46,95,0.55);
        }}
        .kpi .bar {{
            position: absolute; left:0; top:0; bottom:0; width:5px;
            background: linear-gradient(180deg, var(--c1), var(--c2));
        }}
        .kpi .icon {{
            display:inline-flex; align-items:center; justify-content:center;
            width:34px; height:34px; border-radius:9px;
            background: linear-gradient(135deg, var(--c1), var(--c2)); color: white;
            font-weight: 800; margin-bottom: 0.5rem;
        }}
        .kpi-label {{ color: {MUTED}; font-size: 0.8rem; }}
        .kpi-value {{ color: {NAVY}; font-size: 1.65rem; font-weight: 800; line-height: 1.1; }}
        .kpi-sub {{ color: {MUTED}; font-size: 0.78rem; margin-top: 0.15rem; }}
        /* Article cards "A la une" */
        .alaune-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 0.85rem; margin: 0.4rem 0 1rem 0;
        }}
        .alaune {{
            border-radius: 14px; overflow:hidden; background: white;
            border: 1px solid {LINE};
            box-shadow: 0 12px 30px -24px rgba(39,46,95,0.45);
            transition: transform 200ms ease, box-shadow 200ms ease;
            display:flex; flex-direction:column; height:100%;
        }}
        .alaune:hover {{ transform: translateY(-3px); box-shadow: 0 22px 44px -22px rgba(39,46,95,0.55); }}
        .alaune .cover {{
            height: 96px; position: relative;
            background: linear-gradient(135deg, var(--c1), var(--c2));
            display:flex; align-items:center; justify-content:space-between; padding: 0.65rem 0.9rem;
        }}
        .alaune .cover::after {{
            content:''; position:absolute; inset:0;
            background: radial-gradient(220px 110px at 90% 20%, rgba(255,255,255,0.30), transparent 60%);
        }}
        .alaune .cover .tag {{
            background: rgba(255,255,255,0.18); color:white;
            padding: 0.2rem 0.55rem; border-radius: 999px;
            font-size: 0.72rem; font-weight: 700; backdrop-filter: blur(6px);
        }}
        .alaune .cover .src {{
            color: rgba(255,255,255,0.95); font-size: 0.78rem; font-weight: 600;
        }}
        .alaune .body {{ padding: 0.9rem 1rem 0.95rem 1rem; display:flex; flex-direction:column; gap: 0.45rem; flex:1; }}
        .alaune .body h4 {{ margin: 0; font-size: 0.98rem; color: {NAVY}; font-weight: 800; line-height: 1.25; }}
        .alaune .body p {{ margin: 0; color: {INK}; font-size: 0.85rem; line-height: 1.45; }}
        .alaune .body a {{
            margin-top: auto; align-self: flex-start;
            color: {RED}; font-weight: 700; font-size: 0.83rem; text-decoration: none;
        }}
        .alaune .body a:hover {{ text-decoration: underline; }}
        /* Section heading */
        .section-h {{
            display:flex; align-items:center; gap: 0.6rem;
            margin: 1.0rem 0 0.5rem 0;
        }}
        .section-h .pill {{
            background: linear-gradient(135deg, {NAVY}, {RED}); color: white;
            border-radius: 999px; padding: 0.25rem 0.75rem; font-size:0.78rem; font-weight:700;
        }}
        .section-h h3 {{ margin: 0; }}
        /* Recommendation banner */
        .rec-banner {{
            border-radius: 14px; padding: 1rem 1.2rem; color: white;
            display:flex; justify-content:space-between; align-items:center; gap: 1rem;
            box-shadow: 0 18px 40px -22px rgba(0,0,0,0.5);
        }}
        .rec-banner h3 {{ margin:0; color: white; }}
        .rec-banner .verdict {{ font-size: 1.55rem; font-weight: 800; }}
        /* Beneficiary spotlight cards */
        .spot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.7rem; margin: 0.4rem 0 1.1rem 0;
        }}
        .spot {{
            position: relative;
            border-radius: 14px; overflow: hidden;
            color: white; min-height: 156px;
            padding: 0.95rem 1rem;
            background: linear-gradient(135deg, var(--c1), var(--c2));
            box-shadow: 0 14px 32px -22px rgba(0,0,0,0.55);
            display: flex; flex-direction: column; gap: 0.45rem;
            transition: transform 200ms ease, box-shadow 200ms ease;
            isolation: isolate;
        }}
        .spot::before {{
            content: ''; position: absolute; inset: 0; z-index: 0;
            background:
              radial-gradient(180px 100px at 110% -10%, rgba(255,255,255,0.30), transparent 60%),
              radial-gradient(280px 160px at -10% 110%, rgba(0,0,0,0.18), transparent 60%);
        }}
        .spot > * {{ position: relative; z-index: 1; }}
        .spot:hover {{ transform: translateY(-3px); box-shadow: 0 22px 44px -22px rgba(0,0,0,0.55); }}
        .spot .row {{ display:flex; justify-content:space-between; align-items:center; gap: 0.5rem; }}
        .spot .row .sector {{
            background: rgba(255,255,255,0.18); border:1px solid rgba(255,255,255,0.28);
            padding: 0.18rem 0.55rem; border-radius: 999px; font-size: 0.72rem; font-weight: 700;
            backdrop-filter: blur(6px);
        }}
        .spot .row .label {{
            font-size: 0.7rem; font-weight: 800; letter-spacing: 0.5px;
            background: rgba(255,255,255,0.95); color: var(--c2);
            padding: 0.15rem 0.45rem; border-radius: 999px;
        }}
        .spot h4 {{
            color: white; margin: 0.1rem 0 0 0; font-size: 1.05rem;
            font-weight: 800; line-height: 1.15;
        }}
        .spot .meta {{ font-size: 0.82rem; color: rgba(255,255,255,0.92); }}
        .spot .stats {{
            display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-top: auto;
        }}
        .spot .stat {{
            background: rgba(0,0,0,0.18); border-radius: 8px; padding: 0.35rem 0.5rem;
        }}
        .spot .stat .l {{ font-size: 0.66rem; opacity: 0.82; letter-spacing: 0.3px; }}
        .spot .stat .v {{ font-size: 1rem; font-weight: 800; }}
        /* Portfolio mini cards */
        .pf-grid {{
            display:grid; grid-template-columns: repeat(auto-fit, minmax(230px,1fr));
            gap: 0.6rem; margin-top: 0.5rem;
        }}
        .pf {{ background:white; border:1px solid {LINE}; border-radius: 12px; padding:0.75rem 0.9rem;
              transition: transform 160ms ease, border-color 160ms ease; }}
        .pf:hover {{ transform: translateY(-2px); border-color: {NAVY}; }}
        .pf .name {{ font-weight:800; color:{NAVY}; font-size: 0.93rem; }}
        .pf .meta {{ font-size:0.78rem; color:{MUTED}; margin-top: 0.15rem; }}
        .pf .chips {{ margin-top: 0.4rem; display:flex; flex-wrap:wrap; gap: 0.25rem; }}
        .pf .chip {{ background:{LIGHT}; color:{INK}; font-size:0.7rem; padding:0.12rem 0.45rem; border-radius:999px; border:1px solid {LINE}; }}
        .pf .chip.funded {{ background: rgba(34,122,74,0.10); color:{GREEN}; border-color: rgba(34,122,74,0.30); }}
        /* Alerts */
        .alert-box {{
            background: white; border: 1px solid {LINE}; border-left: 6px solid {BLUE};
            border-radius: 10px; padding: 0.85rem 1rem; margin: 0.4rem 0 0.8rem 0;
        }}
        .alert-title {{ font-weight: 800; color: {INK}; margin-bottom: 0.3rem; }}
        .small-muted {{ color: {MUTED}; font-size: 0.85rem; }}
        /* Buttons */
        .stButton>button, .stDownloadButton>button {{
            border-radius: 10px;
            border: 1px solid {NAVY};
            background: linear-gradient(135deg, {NAVY}, #1B2150);
            color: white; font-weight: 700;
            transition: transform 120ms ease, box-shadow 120ms ease, filter 120ms ease;
            box-shadow: 0 10px 24px -16px rgba(39,46,95,0.6);
        }}
        .stButton>button:hover, .stDownloadButton>button:hover {{
            transform: translateY(-1px); filter: brightness(1.05);
            background: linear-gradient(135deg, {RED}, #8C0A0F);
            border-color: {RED};
        }}
        /* Tabs */
        button[data-baseweb="tab"] {{
            font-weight: 700 !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {RED} !important;
        }}
        div[data-testid="stMetricValue"] {{ color: {NAVY}; }}
        /* Program cards */
        .prog-grid {{
            display:grid; grid-template-columns: repeat(auto-fit, minmax(310px,1fr));
            gap: 0.9rem; margin: 0.5rem 0 1rem 0;
        }}
        .prog-card {{
            background:#FFFFFF; border:1px solid #EEF0F6; border-radius: 16px;
            overflow:hidden; display:flex; flex-direction:column; height:100%;
            box-shadow: 0 20px 40px -28px rgba(39,46,95,0.30);
            transition: transform 200ms ease, box-shadow 200ms ease;
        }}
        .prog-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 30px 56px -28px rgba(39,46,95,0.45);
        }}
        .prog-cover {{
            padding: 1rem 1.1rem; color:white; position: relative;
        }}
        .prog-cover::after {{
            content:''; position:absolute; inset:0;
            background: radial-gradient(220px 110px at 90% 20%, rgba(255,255,255,0.22), transparent 60%);
        }}
        .prog-name {{
            font-size: 1.05rem; font-weight: 800; line-height: 1.2; position: relative; z-index: 2;
        }}
        .prog-op {{
            font-size: 0.78rem; opacity: 0.88; margin-top: 0.15rem; position: relative; z-index: 2;
        }}
        .prog-body {{
            padding: 0.85rem 1.1rem 1rem 1.1rem;
            display:flex; flex-direction:column; gap: 0.55rem; flex:1;
        }}
        .prog-summary {{ font-size:0.86rem; line-height:1.45; color:{INK}; margin: 0; }}
        .prog-meta {{
            display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.4rem;
            border-top: 1px dashed #EEF0F6; border-bottom: 1px dashed #EEF0F6;
            padding: 0.45rem 0;
        }}
        .prog-meta > div {{ display:flex; flex-direction:column; gap: 0.05rem; }}
        .prog-meta .l {{ font-size: 0.68rem; color: {MUTED}; letter-spacing: 0.3px; text-transform: uppercase; }}
        .prog-meta .v {{ font-size: 0.84rem; color: {INK}; font-weight: 700; }}
        .prog-partners {{ display:flex; flex-wrap:wrap; gap: 0.3rem; }}
        .prog-chip {{
            background: #F4F5FA; color: {NAVY}; border:1px solid #E5E7EB;
            padding: 0.15rem 0.55rem; border-radius: 999px;
            font-size: 0.72rem; font-weight: 700;
        }}
        .prog-hl {{ margin: 0; padding-left: 1.05rem; color: {INK}; font-size: 0.82rem; line-height: 1.4; }}
        .prog-link {{
            margin-top: auto; align-self:flex-start;
            color: {RED}; font-weight: 800; font-size: 0.86rem; text-decoration: none;
        }}
        .prog-link:hover {{ text-decoration: underline; }}
        /* News cards */
        .news-grid {{
            display:grid; grid-template-columns: repeat(auto-fit, minmax(300px,1fr));
            gap: 0.9rem; margin: 0.5rem 0 1rem 0;
        }}
        .news-card {{
            background:#FFFFFF; border:1px solid #EEF0F6; border-radius: 16px;
            overflow:hidden; display:flex; flex-direction:column;
            box-shadow: 0 18px 40px -28px rgba(39,46,95,0.30);
            transition: transform 200ms ease, box-shadow 200ms ease;
        }}
        .news-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 26px 52px -28px rgba(39,46,95,0.45);
        }}
        .news-cover {{
            padding: 0.9rem 1rem; color: white;
            display:flex; justify-content:space-between; align-items:center;
            position: relative; min-height: 70px;
        }}
        .news-cover::after {{
            content:''; position:absolute; inset:0;
            background: radial-gradient(180px 90px at 90% 25%, rgba(255,255,255,0.26), transparent 60%);
        }}
        .news-tag {{
            background: rgba(255,255,255,0.18); border:1px solid rgba(255,255,255,0.32);
            padding: 0.18rem 0.6rem; border-radius: 999px; font-size: 0.72rem;
            font-weight: 700; position: relative; z-index: 2;
        }}
        .news-date {{
            font-size: 0.78rem; font-weight: 700; opacity: 0.95; position: relative; z-index: 2;
        }}
        .news-body {{
            padding: 0.85rem 1rem 1rem 1rem;
            display:flex; flex-direction:column; gap: 0.4rem;
        }}
        .news-source {{
            font-size: 0.74rem; color: {RED}; font-weight: 800; letter-spacing: 0.3px; text-transform: uppercase;
        }}
        .news-body h4 {{
            margin: 0; color: {NAVY}; font-size: 1rem; font-weight: 800; line-height: 1.25;
        }}
        .news-body p {{ margin: 0; color: {INK}; font-size: 0.85rem; line-height: 1.45; }}
        .news-body a {{
            margin-top: 0.2rem; color: {RED}; font-weight: 800; font-size: 0.83rem;
            text-decoration: none;
        }}
        .news-body a:hover {{ text-decoration: underline; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


_TONE_GRADIENTS: dict[str, tuple[str, str]] = {
    "navy": (NAVY, "#1B2150"),
    "red": (RED, "#8C0A0F"),
    "gold": (GOLD, "#8C7415"),
    "teal": (TEAL, "#067067"),
    "violet": (VIOLET, "#4C1D95"),
    "rose": (ROSE, "#831238"),
    "blue": (BLUE, "#1E3A8A"),
    "amber": (AMBER, "#7A4F0F"),
    "green": (GREEN, "#0E4A2A"),
}


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


HERO_QUOTES_EN: list[tuple[str, str, str]] = [
    ("Speed is the moat. Compounding execution beats clever strategy on a slide deck.",
     "Founder mindset", "FM"),
    ("Capital is patient when traction is loud. Make the numbers talk before you do.",
     "Investor wisdom", "IW"),
    ("The best Tunisian startups don't ask for permission. They ship, measure, iterate.",
     "Builder's law", "BL"),
    ("Distribution is the new innovation. Pick a wedge and own it before anyone notices.",
     "Operator's playbook", "OP"),
    ("Funding is a milestone, not a finish line. Welcome to the control room.",
     "CDC LaunchPAD", "CL"),
]
HERO_QUOTES_FR: list[tuple[str, str, str]] = [
    ("La vitesse est votre rempart. L'execution composee bat la strategie sur slide.",
     "Mental de fondateur", "MF"),
    ("Le capital est patient quand la traction est bruyante. Faites parler les chiffres avant vous.",
     "Sagesse d'investisseur", "SI"),
    ("Les meilleures startups tunisiennes ne demandent pas la permission. Elles livrent, mesurent, iterent.",
     "Loi du builder", "LB"),
    ("La distribution est la nouvelle innovation. Choisissez un creneau et dominez-le.",
     "Manuel de l'operateur", "MO"),
    ("Le financement est une etape, pas une ligne d'arrivee. Bienvenue dans la salle de controle.",
     "CDC LaunchPAD", "CL"),
]


def _header(lang: str) -> None:
    import base64
    import streamlit as st

    media_html = ""
    if os.path.exists(LOGO_ANIMATION):
        with open(LOGO_ANIMATION, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("ascii")
        media_html = (
            f"<video autoplay loop muted playsinline preload='auto' "
            f"poster=''><source src='data:video/mp4;base64,{encoded}' type='video/mp4'></video>"
        )
    elif os.path.exists(LOGO_FILE):
        with open(LOGO_FILE, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("ascii")
        media_html = f"<img src='data:image/png;base64,{encoded}' alt='CDC'>"

    badges = (("Smart Capital", RED), ("Startup Act", NAVY))
    badge_html = "".join(
        f"<span class='badge'><span class='dot' style='background:{c}'></span>{label}</span>"
        for label, c in badges
    )
    quotes = HERO_QUOTES_FR if lang == "FR" else HERO_QUOTES_EN
    avatar_palette = [(NAVY, "#1B2150"), (RED, "#8C0A0F"),
                      (NAVY, RED), (RED, NAVY), (NAVY, "#1B2150")]
    quote_html = "".join(
        f"<div class='cdc-quote'>"
        f"  <div class='avatar' style='background:linear-gradient(135deg,{c1},{c2})'>{initials}</div>"
        f"  <div><div class='text'>\"{text}\"</div>"
        f"  <div class='who'>{who}</div></div>"
        f"</div>"
        for (text, who, initials), (c1, c2) in zip(quotes, avatar_palette)
    )
    tagline = (
        "From funding to breakout, welcome to Tunisia's next "
        "<span class='accent'>game-changers'</span> control room."
        if lang == "EN"
        else "Du financement a la rupture, bienvenue dans la salle de "
             "controle des <span class='accent'>game-changers</span> tunisiens."
    )

    cdc_letters = "".join(f"<span class='pad' style='animation-delay:{i*0.18}s'>{ch}</span>"
                          for i, ch in enumerate("CDC LaunchPAD"))
    st.markdown(
        f"""
        <div class="cdc-hero">
            <div class="cdc-hero-row">
                <div class="cdc-hero-logo">{media_html}</div>
                <div>
                    <h1 class="cdc-title-anim">{cdc_letters}</h1>
                    <p class="cdc-tag-line">{tagline}</p>
                    <div class="badges">{badge_html}</div>
                    <div class="cdc-quote-wrap">{quote_html}</div>
                </div>
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
        import base64 as _b64
        if os.path.exists(LOGO_FILE):
            with open(LOGO_FILE, "rb") as _fh:
                _enc = _b64.b64encode(_fh.read()).decode("ascii")
            st.markdown(
                f"<div style='text-align:center;padding:0.4rem 0 0.6rem 0'>"
                f"<img src='data:image/png;base64,{_enc}' style='max-width:140px;border-radius:10px'/>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            "<div style='background:linear-gradient(135deg,#272E5F,#1B2150);"
            "color:white;padding:0.7rem 0.85rem;border-radius:12px;margin-bottom:0.7rem'>"
            "<div style='font-size:0.72rem;opacity:.8;letter-spacing:.5px'>PLATEFORME</div>"
            "<div style='font-weight:800;font-size:1.05rem'>CDC LAUNCHPAD</div>"
            "<div style='font-size:0.78rem;opacity:.85;margin-top:.15rem'>Decision IA - VAIR / FMVA</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        session.lang = st.radio(
            "Langue / Language",
            ["EN", "FR"],
            index=0 if session.lang == "EN" else 1,
            horizontal=True,
        )
        st.divider()
        if session.auth:
            email = session.get("auth_email", "")
            st.markdown(
                f"<div style='background:rgba(34,122,74,.12);border:1px solid rgba(34,122,74,.35);"
                f"border-radius:10px;padding:0.55rem 0.7rem;color:{INK};font-size:0.85rem'>"
                f"<b style='color:{GREEN}'>Connecte</b><br>"
                f"<span style='color:{MUTED};font-size:0.78rem'>{email or 'utilisateur autorise'}</span></div>",
                unsafe_allow_html=True,
            )
            if st.button(t("logout", session.lang), use_container_width=True):
                session.auth = False
                session["auth_issued"] = None
                st.rerun()
        else:
            st.info("Acces restreint - code par email")
        st.divider()
        st.caption("Support")
        st.markdown(
            f"<div style='font-size:0.82rem;color:{MUTED}'>"
            f"Administrateur : <a href='mailto:{ADMIN_EMAIL}' style='color:{RED}'>{ADMIN_EMAIL}</a></div>",
            unsafe_allow_html=True,
        )

    lang = session.lang
    _header(lang)

    if not session.auth:
        st.markdown(
            "<div class='section-h'><span class='pill'>Acces securise</span>"
            "<h3>Authentification a deux etapes</h3></div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Saisissez votre email professionnel pour recevoir un code a usage unique "
            "(6 chiffres, valide 20 minutes). Aucun code n'est affiche a l'ecran."
            if lang == "FR"
            else "Enter your professional email to receive a one-time 6-digit code "
                 "(valid 20 minutes). No code is shown on screen."
        )
        left, right = st.columns([1, 1])
        with left:
            email_value = st.text_input(t("email", lang),
                                        value=session.get("auth_email", ""),
                                        placeholder="name@cdc.tn",
                                        key="auth_email_input")
            if st.button(t("request", lang), use_container_width=True, key="req_code_btn"):
                email_clean = email_value.strip().lower()
                if "@" not in email_clean or "." not in email_clean.split("@")[-1]:
                    st.error("Adresse email invalide." if lang == "FR" else "Invalid email address.")
                else:
                    issued = issue_access_code(email_clean)
                    session["auth_issued"] = issued
                    session["auth_email"] = email_clean
                    if issued["ok"] and issued["channel"] == "email":
                        st.success(
                            "Code envoye par email. Verifiez votre boite de reception (et vos spams)."
                            if lang == "FR"
                            else "Code sent by email. Check your inbox (and spam folder)."
                        )
                    elif issued["ok"] and issued["channel"] == "admin_log":
                        if lang == "FR":
                            st.warning(
                                "SMTP non configure (mode developpement). "
                                "L'administrateur peut recuperer le code de l'une des facons suivantes :"
                            )
                            st.markdown(
                                "- **Streamlit Cloud** : ouvrir *Manage app* en bas a droite, "
                                "onglet *Logs*, chercher la banniere `DEV-MODE ACCESS CODE`.\n"
                                "- **Local** : voir le terminal ou tourne `streamlit run`, ou ouvrir le "
                                f"fichier `{os.path.basename(ACCESS_LOG)}` a cote de `app.py`.\n"
                                "- **Production** : ajouter les secrets `CDC_SMTP_HOST`, "
                                "`CDC_SMTP_USER`, `CDC_SMTP_PASS`, `CDC_SMTP_FROM` "
                                "(Streamlit Cloud : *Settings -> Secrets*) puis redemander un code."
                            )
                        else:
                            st.warning(
                                "SMTP not configured (dev mode). Admin can retrieve the code one of these ways:"
                            )
                            st.markdown(
                                "- **Streamlit Cloud**: open *Manage app* (bottom-right), *Logs* tab, "
                                "look for the `DEV-MODE ACCESS CODE` banner.\n"
                                "- **Local**: see the terminal running `streamlit run`, or open "
                                f"`{os.path.basename(ACCESS_LOG)}` next to `app.py`.\n"
                                "- **Production**: add secrets `CDC_SMTP_HOST`, `CDC_SMTP_USER`, "
                                "`CDC_SMTP_PASS`, `CDC_SMTP_FROM` (Streamlit Cloud: *Settings -> Secrets*), "
                                "then request a new code."
                            )
                    else:
                        st.error("Envoi impossible. Contactez l'administrateur."
                                 if lang == "FR" else "Could not send code. Contact admin.")
        with right:
            code = st.text_input(t("code", lang), type="password", key="auth_code_input",
                                 placeholder="6 chiffres")
            if st.button(t("enter", lang), use_container_width=True, key="enter_btn"):
                issued = session.get("auth_issued")
                if not issued:
                    st.error("Demandez d'abord un code." if lang == "FR" else "Request a code first.")
                elif verify_access_code(session.get("auth_email", ""), code.strip(), issued):
                    session.auth = True
                    session["auth_issued"] = None
                    st.rerun()
                else:
                    st.error(t("bad_code", lang))
        st.caption(
            f"Administrateur : {ADMIN_EMAIL} - support@cdc.tn"
            if lang == "FR"
            else f"Administrator: {ADMIN_EMAIL} - support@cdc.tn"
        )
        st.stop()

    tabs = st.tabs(
        [
            t("tab_ecosystem", lang),
            t("tab_programs", lang),
            t("tab_portfolio", lang),
            t("tab_assessment", lang),
            t("tab_news", lang),
            t("tab_reports", lang),
        ]
    )

    with tabs[0]:
        kpi_values = compute_impact_kpis(df)
        kpi_html_parts = []
        for kpi in IMPACT_KPI_DEFINITIONS:
            v = kpi_values.get(kpi["key"], {"value": 0, "secondary": ""})
            c1, c2 = _TONE_GRADIENTS.get(kpi["tone"], (NAVY, "#1B2150"))
            kpi_html_parts.append(
                f"<div class='kpi' style='--c1:{c1}; --c2:{c2}'>"
                f"<span class='bar'></span>"
                f"<div class='icon'>{kpi['icon']}</div>"
                f"<div class='kpi-label'>{kpi['label']}</div>"
                f"<div class='kpi-value'>{v['value']:,}</div>"
                f"<div class='kpi-sub'>{v['secondary']}</div>"
                f"</div>"
            )
        st.markdown(
            "<div class='section-h'><span class='pill'>A la une</span>"
            "<h3>Pouls de l'ecosysteme tunisien</h3></div>"
            f"<div class='kpi-strip'>{''.join(kpi_html_parts)}</div>",
            unsafe_allow_html=True,
        )

        article_cards = []
        for art in ECOSYSTEM_ARTICLES:
            c1, c2 = _TONE_GRADIENTS.get(art["color"], (NAVY, "#1B2150"))
            article_cards.append(
                f"<div class='alaune'>"
                f"  <div class='cover' style='--c1:{c1}; --c2:{c2}'>"
                f"    <span class='tag'>{art['tag']}</span>"
                f"    <span class='src'>{art['source']}</span>"
                f"  </div>"
                f"  <div class='body'>"
                f"    <h4>{art['title']}</h4>"
                f"    <p>{art['summary']}</p>"
                f"    <a href='{art['url']}' target='_blank' rel='noopener'>Ouvrir la source &nbsp;&rsaquo;</a>"
                f"  </div>"
                f"</div>"
            )
        st.markdown(
            "<div class='section-h'><span class='pill' style='background:linear-gradient(135deg,#0FB5A6,#067067)'>Veille</span>"
            "<h3>Initiatives, fonds et programmes a suivre</h3></div>"
            f"<div class='alaune-grid'>{''.join(article_cards)}</div>",
            unsafe_allow_html=True,
        )

        spotlight = beneficiary_spotlight(df, k=4)
        if spotlight:
            cards_html = []
            for b in spotlight:
                c1, c2 = _TONE_GRADIENTS.get(b["color"], (NAVY, "#1B2150"))
                label_html = "<span class='label'>Startup Act</span>" if b["labelled"] else ""
                year = b["year"] if b["year"] else "n/d"
                cards_html.append(
                    f"<div class='spot' style='--c1:{c1}; --c2:{c2}'>"
                    f"  <div class='row'>"
                    f"    <span class='sector'>{b['sector'][:22]}</span>"
                    f"    {label_html}"
                    f"  </div>"
                    f"  <h4>{b['name'][:38]}</h4>"
                    f"  <div class='meta'>{b['region']}</div>"
                    f"  <div class='stats'>"
                    f"    <div class='stat'><div class='l'>Fondee</div><div class='v'>{year}</div></div>"
                    f"    <div class='stat'><div class='l'>Statut</div><div class='v'>Financee</div></div>"
                    f"  </div>"
                    f"</div>"
                )
            st.markdown(
                "<div class='section-h'><span class='pill' style='background:linear-gradient(135deg,#7C3AED,#4C1D95)'>Beneficiaires</span>"
                "<h3>Coups de projecteur sur les startups financees</h3></div>"
                f"<div class='spot-grid'>{''.join(cards_html)}</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div class='section-h'><span class='pill' style='background:linear-gradient(135deg,#C9A227,#8C7415)'>Carte sectorielle</span>"
            "<h3>Cartographie des secteurs et dynamique annuelle</h3></div>",
            unsafe_allow_html=True,
        )
        c_left, c_right = st.columns([1.4, 1])
        with c_left:
            top = df["sector"].value_counts().head(14)
            st.plotly_chart(plotly_treemap(top, "Top secteurs (taille = nombre de startups)"),
                            use_container_width=True, config={"displayModeBar": False})
        with c_right:
            st.markdown("**Dynamique annuelle**")
            years = pd.to_numeric(df["founding_year"], errors="coerce").dropna().astype(int)
            yearly = years.value_counts().sort_index().tail(15)
            st.line_chart(yearly, color=NAVY, height=320)

    with tabs[1]:
        _render_programs_tab(lang)

    with tabs[2]:
        st.markdown(
            "<div class='section-h'><span class='pill'>Portefeuille</span>"
            "<h3>Cartographie interactive du portefeuille</h3></div>",
            unsafe_allow_html=True,
        )
        kpi_values = compute_impact_kpis(df)
        kpi_html_parts = []
        for kpi in IMPACT_KPI_DEFINITIONS[:4]:
            v = kpi_values.get(kpi["key"], {"value": 0, "secondary": ""})
            c1, c2 = _TONE_GRADIENTS.get(kpi["tone"], (NAVY, "#1B2150"))
            kpi_html_parts.append(
                f"<div class='kpi' style='--c1:{c1}; --c2:{c2}'>"
                f"<span class='bar'></span><div class='icon'>{kpi['icon']}</div>"
                f"<div class='kpi-label'>{kpi['label']}</div>"
                f"<div class='kpi-value'>{v['value']:,}</div>"
                f"<div class='kpi-sub'>{v['secondary']}</div></div>"
            )
        st.markdown(f"<div class='kpi-strip'>{''.join(kpi_html_parts)}</div>",
                    unsafe_allow_html=True)

        f1, f2, f3 = st.columns([1.2, 1, 1])
        sector_filter = f1.multiselect(
            "Filtrer par secteur",
            sorted(df["sector"].dropna().astype(str).unique()),
            placeholder="Tous secteurs",
        )
        region_filter = f2.multiselect(
            "Region",
            sorted(df.get("Region", pd.Series([], dtype=str)).dropna().astype(str).unique()) if "Region" in df.columns else [],
            placeholder="Toutes regions",
        )
        only_funded = f3.toggle("Financees uniquement", value=False)

        filtered = segmented_df.copy()
        if sector_filter and "Secteur" in filtered.columns:
            filtered = filtered[filtered["Secteur"].astype(str).isin(sector_filter)]
        elif sector_filter and "sector" in filtered.columns:
            filtered = filtered[filtered["sector"].astype(str).isin(sector_filter)]
        if region_filter and "Region" in filtered.columns:
            filtered = filtered[filtered["Region"].astype(str).isin(region_filter)]
        if only_funded and "funded" in filtered.columns:
            filtered = filtered[filtered["funded"] == 1]

        is_fr_pf = (lang == "FR")
        map_title = "Carte des startups par gouvernorat" if is_fr_pf else "Startup map by governorate"
        st.markdown(
            f"<div class='section-h'><span class='pill' style='background:linear-gradient(135deg,{NAVY},{RED})'>Carte</span>"
            f"<h3>{map_title}</h3></div>",
            unsafe_allow_html=True,
        )
        if "Region" in filtered.columns:
            region_counts = filtered["Region"].dropna().astype(str).value_counts()
        else:
            region_counts = pd.Series(dtype=int)
        tn_map = plotly_tunisia_map(region_counts, filtered)
        if tn_map is not None:
            st.plotly_chart(tn_map, use_container_width=True,
                            config={"displayModeBar": False, "scrollZoom": False})
        else:
            st.info(
                "Aucune region exploitable dans les filtres actuels."
                if is_fr_pf else "No region data with current filters."
            )

        c_left, c_right = st.columns([1.4, 1])
        with c_left:
            st.markdown("**Repartition sectorielle (filtree)**")
            if "Secteur" in filtered.columns:
                top = filtered["Secteur"].value_counts().head(16)
            else:
                top = filtered["sector"].value_counts().head(16)
            if not top.empty:
                st.plotly_chart(plotly_treemap(top, ""),
                                use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Aucune startup avec ces filtres.")
        with c_right:
            st.markdown("**Segments comportementaux**")
            st.dataframe(
                segment_summary.rename(columns={
                    "profile": "Segment",
                    "startups": "Startups",
                    "funded_rate": "Funded %",
                    "avg_age": "Age moyen",
                }),
                use_container_width=True, hide_index=True, height=320,
            )

        st.markdown("**Cartes startups (top 24 filtrees)**")
        name_col = "Nom" if "Nom" in filtered.columns else None
        secteur_col = "Secteur" if "Secteur" in filtered.columns else "sector"
        cards_html = []
        rows = filtered.head(24)
        for _, r in rows.iterrows():
            name = str(r.get(name_col, "(sans nom)")) if name_col else "(sans nom)"
            sector = str(r.get(secteur_col, ""))
            region = str(r.get("Region", "")) if "Region" in filtered.columns else ""
            year = r.get("founding_year", "")
            try:
                year_str = f"{int(year)}" if pd.notna(year) and year else "-"
            except Exception:
                year_str = "-"
            chips = []
            if sector:
                chips.append(f"<span class='chip'>{sector[:24]}</span>")
            if region:
                chips.append(f"<span class='chip'>{region[:18]}</span>")
            if "funded" in filtered.columns and int(r.get("funded", 0)) == 1:
                chips.append("<span class='chip funded'>Financee</span>")
            if "profile" in filtered.columns and r.get("profile"):
                chips.append(f"<span class='chip'>{str(r.get('profile'))[:20]}</span>")
            cards_html.append(
                f"<div class='pf'>"
                f"<div class='name'>{name}</div>"
                f"<div class='meta'>Fondee en {year_str}</div>"
                f"<div class='chips'>{''.join(chips)}</div>"
                f"</div>"
            )
        st.markdown(f"<div class='pf-grid'>{''.join(cards_html)}</div>",
                    unsafe_allow_html=True)

        with st.expander("Table detaillee (jusqu'a 500 lignes filtrees)", expanded=False):
            display_cols = [
                col for col in ["Nom", "Secteur", "Region", "founding_year", "funded", "profile"]
                if col in filtered.columns
            ]
            st.dataframe(filtered[display_cols].head(500),
                         use_container_width=True, hide_index=True, height=400)

    with tabs[3]:
        is_fr = (lang == "FR")
        inner_tabs = st.tabs([
            "Evaluer" if is_fr else "Run",
            "Scoring comite" if is_fr else "Committee scoring",
            "Valorisation" if is_fr else "Valuation",
            "Diligence financiere" if is_fr else "Financial diligence",
            "Capitaliser" if is_fr else "Capitalize",
        ])
    with inner_tabs[0]:
        run_title = "Lancer l'evaluation" if is_fr else "Run the assessment"
        st.markdown(
            "<div class='section-h'><span class='pill' style='background:linear-gradient(135deg,"
            f"{NAVY},{RED})'>Pilote</span>"
            f"<h3>{run_title}</h3></div>",
            unsafe_allow_html=True,
        )
        query = st.text_input(
            "Look up - nom de startup ou identifiant RNE" if is_fr
            else "Look up - startup name or registry id"
        )
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
            assess_inputs = {
                "sector": sector,
                "founding_year": year,
                "n_founders": founders,
                "is_labelled": labelled,
                "has_email": has_email,
                "has_web": has_web,
                "stage": ["Idea", "MVP", "Revenue-generating"].index(stage),
                "team": team,
                "market": market,
                "product": product,
                "competition": competition,
                "revenue_tnd": revenue,
                "growth": growth,
            }
            result = assess_one(bundle, assess_inputs)
            valuation = valuation_engine(assess_inputs)
            db_alert = lookup_startup(df, name)["title"]
            session["scoring_inputs"] = auto_score_grid(assess_inputs)
            session["fmva_inputs"] = auto_fmva_inputs(assess_inputs)
            session["scoring_meta"] = {
                "name": name,
                "sector": sector,
                "region": region,
                "evaluator": "",
            }
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

    with inner_tabs[1]:
        st.markdown("### Grille de scoring - VAIR Greentech")
        st.caption(
            "Auto-prerempli depuis l'evaluation. Les membres du comite ajustent chaque "
            "critere (0-5), la note d'axe et la note globale sont recalculees en direct."
        )
        if "scoring_inputs" not in session:
            session["scoring_inputs"] = auto_score_grid(
                {"stage": 1, "team": 0.65, "market": 0.65, "product": 0.60,
                 "competition": 0.50, "revenue_tnd": 0, "growth": 0.40,
                 "is_labelled": False, "has_email": False, "has_web": False,
                 "n_founders": 2}
            )
            session["scoring_meta"] = {"name": "", "sector": "", "region": "", "evaluator": ""}

        meta = session["scoring_meta"]
        m1, m2, m3 = st.columns(3)
        meta["name"] = m1.text_input("Startup", meta.get("name", ""), key="grid_name")
        meta["sector"] = m2.text_input("Secteur", meta.get("sector", ""), key="grid_sector")
        meta["evaluator"] = m3.text_input("Membre du comité", meta.get("evaluator", ""), key="grid_eval")

        scores_state = session["scoring_inputs"]
        for block in SCORING_GRID:
            with st.expander(f"**{block['axis']}**", expanded=False):
                st.caption(block["guidance"])
                axis_scores = scores_state.setdefault(block["axis"], {})
                for crit in block["criteria"]:
                    st.markdown(f"**{crit['name']}**")
                    st.caption(f"_{crit['angle']}_ — _Champs : {crit['fields']}_")
                    current = int(axis_scores.get(crit["name"], 0))
                    new = st.slider(
                        "Note (0-5)",
                        min_value=0, max_value=5, value=current, step=1,
                        format="%d",
                        key=f"score_{block['axis']}_{crit['name']}",
                        help=" | ".join(f"{i}: {a}" for i, a in enumerate(crit["anchors"])),
                    )
                    axis_scores[crit["name"]] = new
                    st.caption(f"→ {crit['anchors'][new]}")

        scorecard = committee_scorecard(scores_state)
        rationales = {ax["axis"]: axis_rationale(
            ax["axis"], {c["name"]: c["score"] for c in ax["criteria"]}
        ) for ax in scorecard["axes"]}
        last_score = float((session.get("last_assessment") or {}).get("score", 0.0))
        last_fmva = session.get("last_fmva")
        overall = overall_recommendation(last_score, scorecard, last_fmva)
        session["last_overall"] = overall
        session["last_rationales"] = rationales

        st.markdown(
            "<div class='section-h'><span class='pill' style='background:linear-gradient(135deg,#D10A11,#8C0A0F)'>Synthese</span>"
            "<h3>Verdict et justification du comite</h3></div>",
            unsafe_allow_html=True,
        )
        c1_g, c2_g = _TONE_GRADIENTS["green" if overall["tone"] == "ok" else ("amber" if overall["tone"] == "warn" else "red")]
        rationale_html = "".join(f"<div>- {x}</div>" for x in overall["rationale"])
        st.markdown(
            f"<div class='rec-banner' style='background:linear-gradient(135deg,{c1_g},{c2_g})'>"
            f"<div><div style='opacity:.85;font-size:.85rem'>Recommandation</div>"
            f"<div class='verdict'>{overall['action']}</div></div>"
            f"<div style='font-size:.88rem;max-width:60%'>{rationale_html}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        sc1, sc2, sc3 = st.columns([1, 1, 1])
        sc1.metric("Note globale", f"{scorecard['global_note']:.2f} / 5")
        sc2.metric("Score modele", f"{last_score:.1f}/100" if last_score else "n/d")
        sc3.metric("FMVA IQR",
                   f"{last_fmva['iqr_ratio']:.0%}" if last_fmva else "n/d",
                   "Revue" if last_fmva and last_fmva["review_flag"] else None)

        st.markdown("**Profil par axe (radar 0-5)**")
        st.plotly_chart(plotly_radar(scorecard),
                        use_container_width=True, config={"displayModeBar": False})

        st.markdown("**Justification axe par axe**")
        for ax in scorecard["axes"]:
            rat = rationales[ax["axis"]]
            with st.expander(f"{ax['axis']} - {ax['note']}/5", expanded=False):
                st.write(rat["summary"])
                if rat["strengths"]:
                    st.markdown("**Points forts**")
                    for s in rat["strengths"]:
                        st.markdown(f"- {s}")
                if rat["gaps"]:
                    st.markdown("**Points faibles**")
                    for g in rat["gaps"]:
                        st.markdown(f"- {g}")
                st.info(f"Conseil : {rat['advice']}")

        st.markdown("**Prochaines etapes proposees**")
        for step in overall["next_steps"]:
            st.markdown(f"- {step}")

        grid_payload = {**meta}
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Grille de scoring (Excel)",
                scoring_grid_excel(grid_payload, scorecard),
                file_name=f"Grille_{(meta.get('name') or 'startup').replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="dl_grid_committee_tab",
            )
        with d2:
            st.download_button(
                "Rapport comite (PDF)",
                committee_pdf(grid_payload, scorecard, rationales, overall),
                file_name=f"Rapport_comite_{(meta.get('name') or 'startup').replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="dl_pdf_committee_tab",
            )
        session["last_scorecard"] = scorecard

    with inner_tabs[2]:
        st.markdown("### Valorisation - Triangulation 5 methodes")
        st.caption(
            "Berkus / Scorecard (Payne) / Risk Factor Summation / Venture Capital / Hybrid DCF "
            "- ponderes en ensemble avec controle qualite (IQR > 60% = revue requise)."
        )
        if "fmva_inputs" not in session:
            session["fmva_inputs"] = auto_fmva_inputs(
                {"stage": 1, "team": 0.65, "market": 0.65, "product": 0.60,
                 "competition": 0.50, "revenue_tnd": 0, "growth": 0.40,
                 "is_labelled": False}
            )
        fmva = session["fmva_inputs"]

        h1, h2 = st.columns(2)
        fmva["baseline_usd"] = h1.number_input(
            "Tunisia baseline pre-money (USD)",
            min_value=100_000, max_value=20_000_000,
            value=int(fmva.get("baseline_usd", TUNISIA_BASELINE_USD)),
            step=100_000,
        )
        fmva["tnd_per_usd"] = h2.number_input(
            "TND per USD", min_value=0.5, max_value=10.0,
            value=float(fmva.get("tnd_per_usd", TND_PER_USD)), step=0.05,
        )

        with st.expander("Berkus — 5 factors × USD 500k max", expanded=False):
            for factor in BERKUS_FACTORS:
                fmva["berkus"][factor] = st.number_input(
                    factor, min_value=0, max_value=500_000,
                    value=int(fmva["berkus"].get(factor, 0)),
                    step=25_000, key=f"berkus_{factor}",
                )

        with st.expander("Scorecard (Payne) — multipliers 0.5–1.5", expanded=False):
            for factor in SCORECARD_WEIGHTS:
                fmva["scorecard"][factor] = st.slider(
                    f"{factor} (weight {SCORECARD_WEIGHTS[factor]:.0%})",
                    0.5, 1.5, float(fmva["scorecard"].get(factor, 1.0)), 0.05,
                    key=f"sc_{factor}",
                )

        with st.expander("Risk Factor Summation — 12 dimensions (-2 to +2)", expanded=False):
            cols = st.columns(2)
            for i, dim in enumerate(RFS_DIMENSIONS):
                with cols[i % 2]:
                    fmva["rfs"][dim] = st.slider(
                        dim, -2, 2, int(fmva["rfs"].get(dim, 0)), 1,
                        key=f"rfs_{dim}",
                    )

        with st.expander("Venture Capital Method", expanded=False):
            v1, v2, v3 = st.columns(3)
            fmva["vc"]["current_revenue_tnd"] = v1.number_input(
                "Current revenue (TND)", min_value=0, max_value=50_000_000,
                value=int(fmva["vc"].get("current_revenue_tnd", 850_000)), step=50_000,
            )
            fmva["vc"]["growth"] = v2.slider(
                "Growth", 0.0, 2.0, float(fmva["vc"].get("growth", 0.5)), 0.05,
                key="vc_growth",
            )
            fmva["vc"]["exit_year"] = v3.number_input(
                "Exit year", min_value=2, max_value=10,
                value=int(fmva["vc"].get("exit_year", 5)),
            )
            v4, v5, v6 = st.columns(3)
            fmva["vc"]["exit_multiple"] = v4.slider(
                "Exit revenue multiple", 1.0, 15.0,
                float(fmva["vc"].get("exit_multiple", 4.5)), 0.5,
            )
            fmva["vc"]["target_return"] = v5.slider(
                "Target VC return (x)", 2.0, 30.0,
                float(fmva["vc"].get("target_return", 10.0)), 1.0,
            )
            fmva["vc"]["round_size_usd"] = v6.number_input(
                "Round size (USD)", min_value=0, max_value=20_000_000,
                value=int(fmva["vc"].get("round_size_usd", 500_000)), step=50_000,
            )

        with st.expander("Hybrid DCF — 5-year explicit + terminal multiple", expanded=False):
            d1, d2, d3, d4 = st.columns(4)
            fmva["dcf"]["starting_ebitda_tnd"] = d1.number_input(
                "Starting EBITDA (TND)", min_value=0, max_value=20_000_000,
                value=int(fmva["dcf"].get("starting_ebitda_tnd", 120_000)), step=20_000,
            )
            fmva["dcf"]["ebitda_growth"] = d2.slider(
                "EBITDA growth", 0.0, 2.0,
                float(fmva["dcf"].get("ebitda_growth", 0.45)), 0.05,
                key="dcf_growth",
            )
            fmva["dcf"]["wacc"] = d3.slider(
                "WACC", 0.05, 0.60, float(fmva["dcf"].get("wacc", 0.275)), 0.005,
            )
            fmva["dcf"]["terminal_multiple"] = d4.slider(
                "Terminal multiple", 2.0, 15.0,
                float(fmva["dcf"].get("terminal_multiple", 5.0)), 0.5,
            )

        result = fmva_valuation(fmva)
        session["last_fmva"] = result
        last_score = float((session.get("last_assessment") or {}).get("score", 0.0))
        sc_for_overall = session.get("last_scorecard") or committee_scorecard(
            session.get("scoring_inputs") or auto_score_grid(
                {"stage": 1, "team": 0.65, "market": 0.65, "product": 0.6,
                 "competition": 0.5, "revenue_tnd": 0, "growth": 0.4,
                 "is_labelled": False, "has_email": False, "has_web": False,
                 "n_founders": 2}
            )
        )
        overall = overall_recommendation(last_score, sc_for_overall, result)

        st.markdown(
            "<div class='section-h'><span class='pill' style='background:linear-gradient(135deg,#272E5F,#D10A11)'>Triangulation</span>"
            "<h3>Synthese FMVA et recommandation</h3></div>",
            unsafe_allow_html=True,
        )
        c1_g, c2_g = _TONE_GRADIENTS["green" if overall["tone"] == "ok" else ("amber" if overall["tone"] == "warn" else "red")]
        rationale_html = "".join(f"<div>- {x}</div>" for x in overall["rationale"])
        st.markdown(
            f"<div class='rec-banner' style='background:linear-gradient(135deg,{c1_g},{c2_g})'>"
            f"<div><div style='opacity:.85;font-size:.85rem'>Recommandation IA</div>"
            f"<div class='verdict'>{overall['action']}</div></div>"
            f"<div style='font-size:.88rem;max-width:60%'>{rationale_html}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Ensemble (USD)", f"${result['ensemble_usd']/1e6:.2f} M")
        e2.metric("Ensemble (TND)", f"{result['ensemble_tnd']/1e6:.2f} M")
        e3.metric("Low - High",
                  f"${result['low_usd']/1e6:.2f}-${result['high_usd']/1e6:.2f} M")
        e4.metric("IQR ratio", f"{result['iqr_ratio']:.0%}",
                  "Revue requise" if result["review_flag"] else "OK")
        if result["review_flag"]:
            st.warning("Dispersion entre methodes > 60% : retester les hypotheses VC/DCF avant decision.")
        if not result["berkus_cap_ok"]:
            st.warning("Total Berkus depasse le plafond USD 2.5M - reduire un ou plusieurs facteurs.")

        st.markdown("**Composition de l'ensemble et comparatif des methodes**")
        v1, v2 = st.columns([1, 1.4])
        with v1:
            st.plotly_chart(plotly_donut_methods(result),
                            use_container_width=True, config={"displayModeBar": False})
        with v2:
            st.plotly_chart(plotly_method_bars(result),
                            use_container_width=True, config={"displayModeBar": False})

        rows = pd.DataFrame({
            "Methode": list(result["methods_usd"].keys()),
            "USD": [f"${v:,.0f}" for v in result["methods_usd"].values()],
            "TND": [f"{v:,.0f}" for v in result["methods_tnd"].values()],
            "Poids": [f"{ENSEMBLE_WEIGHTS[m]:.0%}" for m in result["methods_usd"]],
        })
        st.dataframe(rows, use_container_width=True, hide_index=True)

        st.markdown("**Justification methode par methode**")
        for m, v in result["methods_usd"].items():
            with st.expander(f"{m} - ${v:,.0f}", expanded=False):
                st.write(method_rationale(m, v, fmva, result))

        last = session.get("last_assessment") or {}
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Workbook FMVA (Excel)",
                fmva_workbook_excel(last, fmva, result),
                file_name=f"FMVA_{(last.get('name') or 'startup').replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="dl_xlsx_fmva_tab",
            )
        with d2:
            st.download_button(
                "Rapport FMVA (PDF)",
                fmva_pdf(last, fmva, result, overall),
                file_name=f"FMVA_{(last.get('name') or 'startup').replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="dl_pdf_fmva_tab",
            )
        session["last_fmva_overall"] = overall

    with inner_tabs[3]:
        is_fr = (lang == "FR")
        st.markdown(
            f"<div class='section-h'><span class='pill' style='background:linear-gradient(135deg,{NAVY},{RED})'>"
            f"{'Due diligence' if is_fr else 'Due diligence'}</span>"
            f"<h3>{'Analyser des etats financiers' if is_fr else 'Analyse financial statements'}</h3></div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Glissez un fichier .xlsx ou .csv contenant le P&L et/ou le bilan. "
            "Le moteur extrait les lignes standard, calcule 12 ratios cles et "
            "produit un memo d'investissement (PDF) avec recommandation argumentee."
            if is_fr
            else "Drop a .xlsx or .csv with the P&L and/or balance sheet. The engine "
                 "extracts the standard lines, computes 12 key ratios and produces an "
                 "investment memo (PDF) with an argued recommendation."
        )
        last = session.get("last_assessment") or {}
        startup_name = st.text_input(
            "Nom de la startup" if is_fr else "Startup name",
            value=last.get("name", "Demo"),
            key="fin_startup_name",
        )
        uploaded = st.file_uploader(
            "Etats financiers (.xlsx / .csv)" if is_fr else "Financial statements (.xlsx / .csv)",
            type=["xlsx", "xlsm", "xls", "csv"],
            key="fin_upload",
        )
        if uploaded is not None:
            parsed = parse_financial_statement(uploaded.getvalue(), uploaded.name)
            if not parsed["ok"]:
                st.error(parsed.get("error", "Erreur de lecture."))
            else:
                data = parsed["extracted"]
                st.success(
                    f"{len(data)} lignes reconnues / {parsed['n_rows']} lignes du fichier."
                    if is_fr
                    else f"{len(data)} lines recognised / {parsed['n_rows']} lines in the file."
                )
                extracted_df = pd.DataFrame(
                    [{"Ligne": k, "Valeur": f"{v:,.0f}"} for k, v in data.items()]
                )
                with st.expander("Lignes extraites", expanded=False):
                    st.dataframe(extracted_df, use_container_width=True, hide_index=True)

                ratios = compute_financial_ratios(data)
                memo = financial_memo(data, ratios)
                session["last_financial"] = {
                    "data": data, "ratios": ratios, "memo": memo,
                    "name": startup_name,
                    "sector": last.get("sector", ""),
                }

                # Recommendation banner
                tone_c1, tone_c2 = _TONE_GRADIENTS[
                    "green" if memo["tone"] == "ok" else ("amber" if memo["tone"] == "warn" else "red")
                ]
                rationale_html = "".join(f"<div>- {t}</div>" for t in memo["thesis"])
                st.markdown(
                    f"<div class='rec-banner' style='background:linear-gradient(135deg,{tone_c1},{tone_c2})'>"
                    f"<div><div style='opacity:.85;font-size:.85rem'>"
                    f"{'Recommandation IA' if is_fr else 'AI recommendation'}</div>"
                    f"<div class='verdict'>{memo['action']}</div></div>"
                    f"<div style='font-size:.88rem;max-width:60%'>{rationale_html}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.metric(
                    "Score de sante financiere" if is_fr else "Financial health score",
                    f"{memo['health_score']*100:.0f}/100",
                )

                ratio_df = pd.DataFrame([
                    {
                        "Ratio": name,
                        "Valeur": _format_ratio(r["value"], r["fmt"]),
                        "Drapeau": {"ok": "Vert", "warn": "Orange", "bad": "Rouge", "na": "n/d"}[r["flag"]],
                        "Lecture": r["explanation"],
                    }
                    for name, r in ratios.items()
                ])
                st.markdown("**Tableau des ratios**" if is_fr else "**Ratio table**")
                st.dataframe(ratio_df, use_container_width=True, hide_index=True)

                if memo["strengths"]:
                    st.markdown("**Points forts**" if is_fr else "**Strengths**")
                    for s in memo["strengths"]:
                        st.markdown(f"- {s}")
                if memo["watch"]:
                    st.markdown("**Vigilance**" if is_fr else "**Watch**")
                    for w in memo["watch"]:
                        st.markdown(f"- {w}")
                if memo["risks"]:
                    st.markdown("**Risques**" if is_fr else "**Risks**")
                    for r_line in memo["risks"]:
                        st.markdown(f"- {r_line}")

                st.markdown("**Prochaines etapes**" if is_fr else "**Next steps**")
                for step in memo["next_steps"]:
                    st.markdown(f"- {step}")

                pdf_buf = financial_pdf(
                    {"name": startup_name, "sector": last.get("sector", "")},
                    data, ratios, memo,
                )
                st.download_button(
                    "Telecharger le memo PDF" if is_fr else "Download memo PDF",
                    pdf_buf,
                    file_name=f"Memo_{(startup_name or 'startup').replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="dl_financial_pdf",
                )
        else:
            st.info(
                "Format attendu : colonne A = libelles, colonnes B+ = annees. "
                "Les libelles reconnus incluent : revenue/chiffre d'affaires, COGS, gross profit, "
                "OpEx, EBITDA, EBIT, net income, interest expense, depreciation, total assets, "
                "current assets, cash, receivables, inventory, current liabilities, total liabilities, "
                "equity, long-term debt."
                if is_fr
                else "Expected format: column A = labels, columns B+ = years. Recognised labels "
                     "include: revenue, COGS, gross profit, OpEx, EBITDA, EBIT, net income, "
                     "interest expense, depreciation, total assets, current assets, cash, "
                     "receivables, inventory, current liabilities, total liabilities, equity, "
                     "long-term debt."
            )

    with inner_tabs[4]:
        is_fr = (lang == "FR")
        st.markdown(
            f"<div class='section-h'><span class='pill' style='background:linear-gradient(135deg,{NAVY},{RED})'>"
            f"{'Apprendre' if is_fr else 'Capitalize'}</span>"
            f"<h3>{'Ajouter une startup et reentrainer le moteur' if is_fr else 'Add a startup, retrain the engine'}</h3></div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Chaque cas que vous labellisez ici alimente la boucle d'apprentissage : "
            "le modele est reentraine immediatement et toutes les recommandations futures en profitent."
            if is_fr
            else "Every case you label here feeds the learning loop: the model retrains "
                 "immediately and every future recommendation benefits."
        )
        with st.form("learning_form"):
            l1, l2, l3 = st.columns(3)
            new_name = l1.text_input(
                "Nom de la startup" if is_fr else "Startup name",
                "NewCo Tunisia",
            )
            new_sector = l2.selectbox(
                "Secteur" if is_fr else "Sector",
                sorted(df["sector"].dropna().astype(str).unique()),
                key="learn_sector",
            )
            new_year = l3.number_input(
                "Annee de creation" if is_fr else "Founding year",
                2000, ANALYSIS_YEAR, 2022, key="learn_year",
            )
            l4, l5, l6 = st.columns(3)
            new_founders = l4.number_input(
                "Fondateurs" if is_fr else "Founders",
                1, 12, 3, key="learn_founders",
            )
            outcome = l5.selectbox(
                "Resultat reel" if is_fr else "Actual outcome",
                (["finance", "non finance"] if is_fr else ["funded", "not funded"]),
            )
            new_labelled = l6.checkbox(
                "Label Startup Act", True, key="learn_label",
            )
            append = st.form_submit_button(
                "Ajouter et reentrainer" if is_fr else "Add and retrain",
                use_container_width=True,
            )
        if append:
            funded_value = 1 if outcome in ("finance", "funded") else 0
            total = append_record({
                "name": new_name, "sector": new_sector, "year": new_year,
                "founders": new_founders, "labelled": new_labelled,
                "funded": funded_value,
            })
            st.cache_resource.clear()
            st.success(
                f"{total} lignes capitalisees. Modele en cours de reentrainement."
                if is_fr
                else f"{total} cases captured. Engine retraining..."
            )
            st.rerun()

    with tabs[4]:
        _render_newsroom_tab(lang)

    with tabs[5]:
        st.markdown(
            "<div class='section-h'><span class='pill'>Centre de rapports</span>"
            "<h3>Tous les livrables generes par la plateforme</h3></div>",
            unsafe_allow_html=True,
        )
        last = session.get("last_assessment")
        scorecard = session.get("last_scorecard")
        rationales = session.get("last_rationales")
        overall = session.get("last_overall") or session.get("last_fmva_overall")
        fmva = session.get("fmva_inputs")
        fmva_result = session.get("last_fmva")

        if not last:
            st.info(
                "Lancez d'abord une evaluation dans l'onglet Assessment pour generer "
                "tous les rapports. Le rapport portefeuille reste disponible ci-dessous."
            )

        slug = (last.get("name") if last else "startup").replace(" ", "_") or "startup"
        st.markdown("#### Dossier startup")
        r1, r2, r3 = st.columns(3)
        if last:
            r1.download_button(
                "Assessment - PDF",
                assessment_pdf(last),
                file_name=f"Assessment_{slug}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="rep_assess_pdf",
            )
            r2.download_button(
                "Assessment - Excel",
                assessment_excel(last),
                file_name=f"Assessment_{slug}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="rep_assess_xlsx",
            )
        else:
            r1.button("Assessment - PDF", disabled=True, use_container_width=True, key="rep_a_pdf_disabled")
            r2.button("Assessment - Excel", disabled=True, use_container_width=True, key="rep_a_xls_disabled")
        if last and scorecard:
            r3.download_button(
                "Comite - PDF",
                committee_pdf(
                    {**(session.get("scoring_meta") or {}),
                     "name": last.get("name"), "sector": last.get("sector"),
                     "region": last.get("region")},
                    scorecard, rationales, overall,
                ),
                file_name=f"Rapport_comite_{slug}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="rep_comm_pdf",
            )
        else:
            r3.button("Comite - PDF", disabled=True, use_container_width=True, key="rep_c_pdf_disabled")

        r4, r5, r6 = st.columns(3)
        if scorecard:
            r4.download_button(
                "Comite - Grille Excel",
                scoring_grid_excel(
                    {**(session.get("scoring_meta") or {}),
                     "name": (last or {}).get("name", ""),
                     "sector": (last or {}).get("sector", ""),
                     "region": (last or {}).get("region", "")},
                    scorecard,
                ),
                file_name=f"Grille_{slug}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="rep_grid_xlsx",
            )
        else:
            r4.button("Comite - Grille Excel", disabled=True, use_container_width=True, key="rep_c_xls_disabled")
        if last and fmva and fmva_result:
            r5.download_button(
                "FMVA - PDF",
                fmva_pdf(last, fmva, fmva_result, overall),
                file_name=f"FMVA_{slug}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="rep_fmva_pdf",
            )
            r6.download_button(
                "FMVA - Workbook Excel",
                fmva_workbook_excel(last, fmva, fmva_result),
                file_name=f"FMVA_{slug}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="rep_fmva_xlsx",
            )
        else:
            r5.button("FMVA - PDF", disabled=True, use_container_width=True, key="rep_f_pdf_disabled")
            r6.button("FMVA - Workbook Excel", disabled=True, use_container_width=True, key="rep_f_xls_disabled")

        st.markdown("#### Portefeuille global")
        p1, _ = st.columns([1, 2])
        with p1:
            st.download_button(
                "Portefeuille - PDF",
                portfolio_pdf(df, segment_summary),
                file_name="CDC_Portfolio_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="rep_pf_pdf",
            )

        if last and scorecard and fmva and fmva_result:
            st.success(
                "Tous les livrables generes : Assessment (PDF+Excel), Comite (PDF+Excel), "
                "FMVA (PDF+Excel), Portefeuille (PDF). Telechargeables ci-dessus."
            )


if __name__ == "__main__":
    run_app()
