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
    "tab_committee": {"EN": "Committee Scoring", "FR": "Grille de scoring"},
    "tab_valuation": {"EN": "Valuation (FMVA)", "FR": "Valorisation (FMVA)"},
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
            t("tab_committee", lang),
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

    with tabs[3]:
        st.markdown("### Committee scoring — VAIR Greentech grid")
        st.caption(
            "Auto-pré-rempli depuis l'évaluation. Les membres du comité ajustent chaque "
            "critère (0-5), la note d'axe et la note globale sont recalculées en direct."
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
        st.markdown("### Synthèse")
        sc1, sc2, sc3 = st.columns([1, 1, 2])
        tone_color = {"ok": GREEN, "warn": AMBER, "bad": RED}[scorecard["tone"]]
        sc1.metric("Note globale", f"{scorecard['global_note']:.2f} / 5")
        sc2.markdown(
            f"<h3 style='color:{tone_color};margin-top:0'>{scorecard['recommendation']}</h3>",
            unsafe_allow_html=True,
        )
        synth_rows = pd.DataFrame(
            [
                {"Axe": ax["axis"], "Pondération": f"{ax['weight']:.0%}", "Note": ax["note"]}
                for ax in scorecard["axes"]
            ]
        )
        st.dataframe(synth_rows, use_container_width=True, hide_index=True)

        grid_payload = {**meta}
        st.download_button(
            "Télécharger la grille remplie (Excel)",
            scoring_grid_excel(grid_payload, scorecard),
            file_name=f"Grille_{(meta.get('name') or 'startup').replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        session["last_scorecard"] = scorecard

    with tabs[4]:
        st.markdown("### Valorisation FMVA — Triangulation 5 méthodes")
        st.caption(
            "Berkus / Scorecard (Payne) / Risk Factor Summation / Venture Capital / Hybrid DCF "
            "— pondérés en ensemble avec contrôle qualité (IQR > 60% = revue requise)."
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
        st.markdown("### Ensemble")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Ensemble (USD)", f"${result['ensemble_usd']/1e6:.2f} M")
        e2.metric("Ensemble (TND)", f"{result['ensemble_tnd']/1e6:.2f} M TND")
        e3.metric("Low – High (USD)",
                  f"${result['low_usd']/1e6:.2f}–${result['high_usd']/1e6:.2f} M")
        e4.metric("IQR ratio", f"{result['iqr_ratio']:.0%}",
                  "Review required" if result["review_flag"] else "OK")
        if result["review_flag"]:
            st.warning("Spread between methods exceeds 60% of the ensemble — review recommended.")
        if not result["berkus_cap_ok"]:
            st.warning("Berkus total exceeds the USD 2.5M cap — reduce one or more factors.")

        rows = pd.DataFrame(
            {
                "Method": list(result["methods_usd"].keys()),
                "USD": [f"${v:,.0f}" for v in result["methods_usd"].values()],
                "TND": [f"{v:,.0f}" for v in result["methods_tnd"].values()],
                "Weight": [f"{ENSEMBLE_WEIGHTS[m]:.0%}" for m in result["methods_usd"]],
            }
        )
        st.dataframe(rows, use_container_width=True, hide_index=True)

        last = session.get("last_assessment") or {}
        st.download_button(
            "Télécharger le workbook FMVA (Excel)",
            fmva_workbook_excel(last, fmva, result),
            file_name=f"FMVA_{(last.get('name') or 'startup').replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        session["last_fmva"] = result

    with tabs[5]:
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

    with tabs[6]:
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
