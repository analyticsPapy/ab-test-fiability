# app.py — Calculateur de fiabilité d'un A/B test (FR)
# Exécutez:  streamlit run app.py
# Dépendances: streamlit, numpy, scipy, pandas
# -------------------------------------------------------------
# BUT DE L'APP
# -------------------------------------------------------------
# Cette application aide à évaluer la "fiabilité" d'un test A/B :
# - Pour un **taux de conversion** (succès/échec) : test Z sur la différence de proportions
# - Pour une **métrique continue** (ex: panier moyen) : test t de Welch
# Elle affiche : p-valeur, intervalle de confiance, lift, puissance post hoc, et taille d'échantillon.
# Le front inclut des guides de lecture et des messages d'aide pour un public non-statisticien.

import math
import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st

# -------------------------------------------------------------
# CONFIG STREAMLIT
# -------------------------------------------------------------
st.set_page_config(page_title="Calculateur A/B test", page_icon="📊", layout="centered")

# Petit thème visuel simple lisible (auto-adapté au dark/light mode Streamlit)
st.markdown(
    """
    <style>
    /* Améliore la lisibilité générale */
    .block-container {max-width: 960px;}

    /* KPI : suivent automatiquement le thème clair/sombre */
    .stMetric { background: var(--background-color-secondary); color: var(--text-color); }

    /* Badges adaptatifs (thème clair/sombre) */
    .help-badge, .warn-badge, .ok-badge {
      display:inline-block; padding:2px 8px; border-radius:8px; font-size:12px;
      background: var(--background-color-secondary); color: var(--text-color);
    }

    /* Couleurs personnalisées pour résultats significatifs / non significatifs */
    .result-significant { display:inline-block; padding:4px 10px; border-radius:10px; font-weight:bold; background: #dcfce7; color: #166534; }
    .result-nonsignificant { display:inline-block; padding:4px 10px; border-radius:10px; font-weight:bold; background: #fee2e2; color: #991b1b; }

    /* 🆕 Cartes de résumé visuel faciles à lire */
    .summary-card {padding:12px 16px; border-radius:14px; margin:12px 0; border:1px solid transparent;}
    .summary-card .summary-title {font-size:16px; font-weight:700; margin-bottom:4px;}
    .summary-card .summary-subtitle {font-size:13px; opacity:0.9;}
    .summary-card.ok {background:#dcfce7; color:#166534; border-color:#86efac;}
    .summary-card.warn {background:#fef9c3; color:#713f12; border-color:#fde68a;}
    .summary-card.bad {background:#fee2e2; color:#991b1b; border-color:#fecaca;}
  </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# HELPERS (fonctions utilitaires)
# -------------------------------------------------------------

def fmt_pct(x: float, digits: int = 2) -> str:
    """Formate un ratio (0–1) en pourcentage lisible.
    Retourne '—' si x est NaN/inf.
    """
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x*100:.{digits}f}%"


def safe_div(a: float, b: float) -> float:
    """Division protégée : retourne NaN si b est nul ou None."""
    return a / b if b not in (0, None) else np.nan

# -------- Tests binomiaux 2-arm --------

def wald_ci_diff_proportions(x_a: int, n_a: int, x_b: int, n_b: int, alpha: float = 0.05, alternative: str = "two-sided"):
    """Test Z (Wald) pour la différence de proportions pB - pA.
    - Utilise l'écart-type *poolé* sous H0 pour la stat de test.
    - Retourne taux A/B, diff, erreur-type, z, p-valeur et IC sur la différence.
    """
    p_a = safe_div(x_a, n_a)
    p_b = safe_div(x_b, n_b)
    diff = p_b - p_a
    pooled = safe_div(x_a + x_b, n_a + n_b)
    se = math.sqrt(pooled * (1 - pooled) * (1/n_a + 1/n_b)) if pooled not in (None, np.nan) else np.nan

    # Seuil critique selon l'alternative
    if alternative == "two-sided":
        z_crit = stats.norm.ppf(1 - alpha/2)
        tail = 2
    else:
        z_crit = stats.norm.ppf(1 - alpha)
        tail = 1

    if se and se > 0:
        ci = (diff - z_crit*se, diff + z_crit*se)
        z = diff / se
        # p-valeur selon le sens du test
        if alternative == "two-sided":
            pval = 2*(1 - stats.norm.cdf(abs(z)))
        elif alternative == "larger":
            pval = 1 - stats.norm.cdf(z)
        else:
            pval = stats.norm.cdf(z)
    else:
        ci, z, pval = (np.nan, np.nan), np.nan, np.nan
    return {
        "p_a": p_a, "p_b": p_b, "diff": diff,
        "se": se, "z": z, "p_value": pval, "ci": ci, "tails": tail
    }

# 🔧 Wilson/Newcombe pour allocations inégales ou petits n

def wilson_interval(x: int, n: int, alpha: float = 0.05):
    if n <= 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha/2)
    p = x / n
    denom = 1 + z**2 / n
    center = (p + z**2/(2*n)) / denom
    half = (z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def newcombe_ci_diff(x_a: int, n_a: int, x_b: int, n_b: int, alpha: float = 0.05):
    """IC sur pB−pA via Newcombe (Wilson par bras)."""
    la, ua = wilson_interval(x_a, n_a, alpha)
    lb, ub = wilson_interval(x_b, n_b, alpha)
    return (lb - ua, ub - la)

# 🔁 Ajustement "peeking" (lecture séquentielle)

def adjust_peeking_p(p: float, looks: int, method: str = "sidak") -> float:
    """Ajuste une p-valeur pour m lectures (peeks). Méthodes : 'bonferroni' ou 'sidak'."""
    m = max(1, int(looks))
    p = float(p)
    if method == "bonferroni":
        return min(1.0, p * m)
    if method == "sidak":
        return min(1.0, 1 - (1 - p)**m)
    return p


def posthoc_power_proportions(p_a: float, p_b: float, n_a: int, n_b: int, alpha: float = 0.05, alternative: str = "two-sided") -> float:
    """Puissance post hoc (approx. normale) pour proportions.
    Interprétation : probabilité de détecter un effet au moins aussi grand que l'observé
    si l'effet observé était la vérité. Indicatif seulement.
    """
    if any(v in (None, np.nan) for v in [p_a, p_b]) or n_a <= 0 or n_b <= 0:
        return np.nan
    se_h1 = math.sqrt(p_a*(1-p_a)/n_a + p_b*(1-p_b)/n_b)
    diff = p_b - p_a
    if se_h1 == 0:
        return np.nan

    if alternative == "two-sided":
        z_alpha = stats.norm.ppf(1 - alpha/2)
        beta = stats.norm.cdf(z_alpha - abs(diff)/se_h1) - stats.norm.cdf(-z_alpha - abs(diff)/se_h1)
        power = 1 - beta
    elif alternative == "larger":
        z_alpha = stats.norm.ppf(1 - alpha)
        power = 1 - stats.norm.cdf(z_alpha - diff/se_h1)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
        power = stats.norm.cdf(-z_alpha - diff/se_h1)
    return max(0.0, min(1.0, power))

# 50/50 : taille par variante

def sample_size_proportions(p0: float, mde_rel: float, alpha: float = 0.05, beta: float = 0.2, alternative: str = "two-sided") -> int:
    """Taille d'échantillon **par variante** (répartition 50/50) pour détecter un MDE relatif
    autour d'un taux baseline p0 au niveau α et avec puissance 1-β.
    """
    p1 = p0 * (1 + mde_rel)
    p1 = min(max(p1, 1e-9), 1-1e-9)
    z_alpha = stats.norm.ppf(1 - (alpha/2 if alternative == "two-sided" else alpha))
    z_beta = stats.norm.ppf(1 - beta)
    se_term = p0*(1-p0) + p1*(1-p1)
    n = ((z_alpha + z_beta)**2 * se_term) / ((p1 - p0)**2)
    return math.ceil(n)

# 🔥 Tailles d'échantillon avec **répartition inégale** (binomiale)

def sample_size_proportions_unequal(p0: float, mde_rel: float, f_B: float, alpha: float = 0.05, beta: float = 0.2, alternative: str = "two-sided"):
    """Calcule N_total et (n_A, n_B) cibles pour un test sur proportions avec répartition **f_B** vers B.
    Formule (approx. normale sous H1) :
      N_total = ((zα + zβ)^2 * ( p0(1-p0)/(1-f) + p1(1-p1)/f )) / (Δ^2)
    """
    f = max(1e-6, min(1-1e-6, f_B))
    p1 = p0 * (1 + mde_rel)
    p1 = min(max(p1, 1e-9), 1-1e-9)
    delta = p1 - p0
    z_alpha = stats.norm.ppf(1 - (alpha/2 if alternative == "two-sided" else alpha))
    z_beta = stats.norm.ppf(1 - beta)
    var_term = p0*(1-p0)/(1 - f) + p1*(1-p1)/f
    N_total = ((z_alpha + z_beta)**2 * var_term) / (delta**2)
    N_total = math.ceil(N_total)
    n_B = math.ceil(f * N_total)
    n_A = N_total - n_B
    return {"n_A": n_A, "n_B": n_B, "N_total": N_total}

# -------- Tests sur moyennes --------

def welch_test_and_ci(mean_a: float, sd_a: float, n_a: int, mean_b: float, sd_b: float, n_b: int, alpha: float = 0.05, alternative: str = "two-sided"):
    """Test t de Welch + IC pour différence de moyennes (B - A)."""
    diff = mean_b - mean_a
    se = math.sqrt((sd_a**2)/n_a + (sd_b**2)/n_b)
    # Degrés de liberté (Welch–Satterthwaite)
    df_num = ((sd_a**2)/n_a + (sd_b**2)/n_b)**2
    df_den = ((sd_a**2/n_a)**2)/(n_a-1) + ((sd_b**2/n_b)**2)/(n_b-1)
    df = df_num/df_den if df_den > 0 else np.nan
    if se == 0 or np.isnan(df):
        return {"diff": diff, "se": se, "df": df, "t": np.nan, "p_value": np.nan, "ci": (np.nan, np.nan)}

    t_stat = diff / se
    if alternative == "two-sided":
        pval = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        t_crit = stats.t.ppf(1 - alpha/2, df)
    elif alternative == "larger":
        pval = 1 - stats.t.cdf(t_stat, df)
        t_crit = stats.t.ppf(1 - alpha, df)
    else:
        pval = stats.t.cdf(t_stat, df)
        t_crit = stats.t.ppf(1 - alpha, df)

    ci = (diff - t_crit*se, diff + t_crit*se)
    return {"diff": diff, "se": se, "df": df, "t": t_stat, "p_value": pval, "ci": ci}


def posthoc_power_means(mean_a: float, sd_a: float, n_a: int, mean_b: float, sd_b: float, n_b: int, alpha: float = 0.05, alternative: str = "two-sided") -> float:
    """Puissance post hoc (approx. normale) pour la différence de moyennes."""
    diff = mean_b - mean_a
    se_h1 = math.sqrt((sd_a**2)/n_a + (sd_b**2)/n_b)
    if se_h1 == 0:
        return np.nan
    if alternative == "two-sided":
        z_alpha = stats.norm.ppf(1 - alpha/2)
        beta = stats.norm.cdf(z_alpha - abs(diff)/se_h1) - stats.norm.cdf(-z_alpha - abs(diff)/se_h1)
        power = 1 - beta
    elif alternative == "larger":
        z_alpha = stats.norm.ppf(1 - alpha)
        power = 1 - stats.norm.cdf(z_alpha - diff/se_h1)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
        power = stats.norm.cdf(-z_alpha - diff/se_h1)
    return max(0.0, min(1.0, power))

# 50/50 : taille par variante

def sample_size_means(sd_pooled: float, mde_abs: float, alpha: float = 0.05, beta: float = 0.2, alternative: str = "two-sided") -> int:
    """Taille d'échantillon **par variante** (50/50) pour détecter une différence absolue (mde_abs)
    avec un écart-type attendu sd_pooled.
    """
    z_alpha = stats.norm.ppf(1 - (alpha/2 if alternative == "two-sided" else alpha))
    z_beta = stats.norm.ppf(1 - beta)
    n = 2 * ((z_alpha + z_beta) * sd_pooled / mde_abs)**2
    return math.ceil(n)

# 🔥 Tailles d'échantillon pour moyennes avec répartition inégale

def sample_size_means_unequal(sd_a: float, sd_b: float, mde_abs: float, f_B: float, alpha: float = 0.05, beta: float = 0.2, alternative: str = "two-sided"):
    """Calcule N_total et (n_A, n_B) cibles pour un test sur moyennes avec répartition **f_B** vers B.
    Formule (approx. normale) : N_total = ((zα+zβ)^2 * (sd_a^2/(1-f) + sd_b^2/f)) / (Δ^2)
    """
    f = max(1e-6, min(1-1e-6, f_B))
    z_alpha = stats.norm.ppf(1 - (alpha/2 if alternative == "two-sided" else alpha))
    z_beta = stats.norm.ppf(1 - beta)
    var_term = (sd_a**2)/(1 - f) + (sd_b**2)/f
    N_total = ((z_alpha + z_beta)**2 * var_term) / (mde_abs**2)
    N_total = math.ceil(N_total)
    n_B = math.ceil(f * N_total)
    n_A = N_total - n_B
    return {"n_A": n_A, "n_B": n_B, "N_total": N_total}

# --------- Ajustements multi-tests ---------

def p_adjust(pvals, method: str = "holm"):
    p = np.array(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p
    if method == "none":
        return p
    if method == "bonferroni":
        return np.minimum(1.0, p * m)
    if method == "holm":
        order = np.argsort(p)
        p_sorted = p[order]
        adj = np.empty_like(p_sorted)
        for i, pv in enumerate(p_sorted):
            adj[i] = (m - i) * pv
        # monotone non-decreasing
        for i in range(1, m):
            adj[i] = max(adj[i], adj[i-1])
        # map back
        out = np.empty_like(p)
        out[order] = np.minimum(1.0, adj)
        return out
    if method in ("bh", "fdr_bh"):
        order = np.argsort(p)
        p_sorted = p[order]
        adj = np.empty_like(p_sorted)
        prev = 1.0
        for i in range(m-1, -1, -1):
            adj[i] = min(prev, p_sorted[i] * m / (i+1))
            prev = adj[i]
        out = np.empty_like(p)
        out[order] = np.minimum(1.0, adj)
        return out
    return p

# --------- 📐 Plan d'échantillonnage multi-variantes (FWER) ---------

def plan_multi_proportions(p0: float, mde_rel: float, m_variants: int, alpha: float = 0.05, beta: float = 0.2, allocation: str = "optimal", method: str = "bonferroni"):
    """Plan par paires vs contrôle sous contrôle FWER.
    - m_variants : nombre de bras variantes (hors contrôle)
    - allocation : "optimal" ⇒ n_C ≈ √m × n_T ; "egalitaire" ⇒ tous ≈ égaux
    - method : "bonferroni" (dimensionnement avec α/m) ; "holm" (≈ utilise α/m aussi, plus conservateur)
    Retourne dict avec n_control, n_variant, N_total et tableau par bras.
    """
    m = max(1, int(m_variants))
    p1 = min(max(p0 * (1 + mde_rel), 1e-9), 1-1e-9)
    delta = p1 - p0
    if method not in ("bonferroni", "holm"):
        method = "bonferroni"
    alpha_star = alpha / m  # conservateur; Holm donnera au moins autant de puissance
    z_alpha = stats.norm.ppf(1 - alpha_star/2)
    z_beta = stats.norm.ppf(1 - beta)

    if allocation == "optimal":
        r = math.sqrt(m)  # n_C = r * n_T
    else:
        r = 1.0

    var_term = p0*(1-p0)/r + p1*(1-p1)
    n_var = math.ceil(((z_alpha + z_beta)**2 * var_term) / (delta**2))
    n_ctrl = math.ceil(r * n_var)

    arms = {"A (contrôle)": n_ctrl}
    for i in range(m):
        arms[f"Variante {chr(66+i)}"] = n_var
    N_total = n_ctrl + m * n_var
    return {"n_control": n_ctrl, "n_variant": n_var, "N_total": N_total, "arms": arms, "ratio_ctrl_per_var": r, "alpha_star": alpha_star}


def plan_multi_means(sd_pooled: float, mde_abs: float, m_variants: int, alpha: float = 0.05, beta: float = 0.2, allocation: str = "optimal", method: str = "bonferroni"):
    """Plan par paires vs contrôle pour moyennes (Welch approx, sd≈constante)."""
    m = max(1, int(m_variants))
    if method not in ("bonferroni", "holm"):
        method = "bonferroni"
    alpha_star = alpha / m
    z_alpha = stats.norm.ppf(1 - alpha_star/2)
    z_beta = stats.norm.ppf(1 - beta)

    if allocation == "optimal":
        r = math.sqrt(m)
    else:
        r = 1.0

    var_term = (1 + 1/r) * (sd_pooled**2)
    n_var = math.ceil(((z_alpha + z_beta)**2 * var_term) / (mde_abs**2))
    n_ctrl = math.ceil(r * n_var)

    arms = {"A (contrôle)": n_ctrl}
    for i in range(m):
        arms[f"Variante {chr(66+i)}"] = n_var
    N_total = n_ctrl + m * n_var
    return {"n_control": n_ctrl, "n_variant": n_var, "N_total": N_total, "arms": arms, "ratio_ctrl_per_var": r, "alpha_star": alpha_star}

# -------------------------------------------------------------
# UI PRINCIPALE
# -------------------------------------------------------------

st.title("📊 Calculateur de fiabilité d'un A/B test")

# 🧭 Aide à choisir Binomiale vs Moyenne continue (accueil)
with st.expander("🤔 Quand choisir *Binomiale* vs *Moyenne continue* ?", expanded=True):
    st.markdown(
        """
        - **Binomiale (taux de conversion)** → votre métrique vaut **0/1** (ex. : a converti / n'a pas converti).
          *Exemples* : inscription, achat, clic, ajout au panier.
        - **Moyenne continue (Welch)** → votre métrique est **numérique continue** (ex. : panier moyen, revenu, pages vues, durée).
          *Exemples* : panier moyen (€), nombre de pages, temps passé (s/min).

        👉 Règle simple : si vous comptez des **succès** sur un **nombre d'essais**, c'est *binomiale* ; sinon, si c'est une **valeur mesurée**, c'est *moyenne continue*.
        """
    )

# Bandeau d'aide rapide pour non-statisticien
with st.expander("🧭 Guide de lecture rapide (recommandé)", expanded=False):
    st.markdown(
        """
        **Ce que vous verrez :**
        - **p-valeur** : plus elle est **petite** (ex. < 0,05), plus on peut dire que **la différence est réelle**.
        - **IC (intervalle de confiance)** : la **fourchette** dans laquelle l'effet réel a de bonnes chances de se trouver.
        - **Lift** : l'amélioration **en %** de B par rapport à A.
        - **Puissance** : la **capacité** du test à détecter un effet. On vise **80 % ou plus**.
        - **Peeking** : si on **regarde souvent** les résultats, on **durcit** le seuil (p-valeur ajustée) pour éviter les faux positifs.
        """,
        unsafe_allow_html=True,
    )

st.caption("Choisissez **taux de conversion (binomiale)** ou **moyenne continue**. L'app calcule p-valeur, IC, lift, puissance et taille d'échantillon, avec explications.")

# Panneau latéral : paramètres globaux
with st.sidebar:
    st.header("Paramètres globaux")
    alpha = st.number_input("Niveau de risque α", value=0.05, min_value=0.0001, max_value=0.2, step=0.005, format="%.3f")
    alt_choice = st.selectbox(
        "Hypothèse alternative",
        options=[("two-sided", "Bilatéral (par défaut)"), ("larger", "A < B (on teste B>A)"), ("smaller", "A > B (on teste B<A)")],
        index=0,
        format_func=lambda x: x[1]
    )
    alternative = alt_choice[0]
    metric_type = st.radio("Type de métrique", ["Taux de conversion (binomiale)", "Moyenne continue"], index=0)
    multi_mode = st.checkbox("Activer mode multi-variantes (A/B/n)", value=False)
    mt_correction = st.selectbox("Correction multi-tests", options=["holm", "bonferroni", "bh", "none"], index=0, help="Appliquée aux comparaisons vs contrôle en mode multi-variantes")

    st.caption("👉 Pour tester **plus de 2 variantes**, cochez *Mode multi-variantes (A/B/n)* puis **ajoutez/supprimez des lignes** dans le tableau.")

    st.divider()
    st.header("⏱️ Durée & monitoring (peeking)")
    dur_jours = st.number_input("Durée du test (jours)", min_value=0.1, value=14.0, step=0.5)
    freq = st.selectbox("Fréquence d'analyse", ["Analyse unique (pas de peeking)", "Quotidienne", "Horaire"], index=1)
    if freq.startswith("Analyse unique"):
        looks_auto = 1
    elif freq == "Quotidienne":
        looks_auto = int(math.ceil(dur_jours))
    else:
        looks_auto = int(math.ceil(dur_jours * 24))
    looks = st.number_input("Nombre de lectures effectuées (peeks)", min_value=1, value=int(looks_auto), step=1, help="Nombre de fois où vous avez regardé les résultats pendant le test")
    peek_method = st.selectbox("Ajustement peeking", ["sidak", "bonferroni", "aucun"], index=0, help="Ajuste la p-valeur pour m lectures séquentielles")

st.markdown("### 1) Renseigner les données de l'A/B test")

# ======================================================================================
# MODE MULTI-VARIANTES (analyses k>2)
# ======================================================================================
if multi_mode and metric_type == "Taux de conversion (binomiale)":
    st.subheader("Mode multi-variantes — Binomiale")

    # Table éditable : label, n, x
    df_init = pd.DataFrame({
        "Variante": ["A (contrôle)", "B", "C"],
        "n": [1000, 800, 800],
        "x": [100, 96, 92],
    })
    df_user = st.data_editor(df_init, num_rows="dynamic", use_container_width=True, key="multi_binom")

    # Nettoyage
    df_user = df_user.dropna()
    df_user["n"] = df_user["n"].astype(int)
    df_user["x"] = df_user["x"].astype(int)

    if len(df_user) < 2:
        st.info("Ajoutez au moins 2 variantes (dont un contrôle).")
    else:
        # Global: chi2 d'indépendance 2xk (succès vs échec par variante)
        table = np.vstack([df_user["x"].values, df_user["n"].values - df_user["x"].values])
        chi2, p_glob, dof = None, None, None
        try:
            from scipy.stats import chi2_contingency
            chi2, p_glob, dof, _ = chi2_contingency(table.T)
        except Exception:
            pass

        st.markdown(f"**Test global (chi²)** — p = `{p_glob:.4g}`" if p_glob is not None else "Test global indisponible (SciPy)")

        # Pairwise vs contrôle (ligne 0)
        base_x, base_n = int(df_user.loc[df_user.index[0], "x"]), int(df_user.loc[df_user.index[0], "n"])
        results = []
        for idx in range(1, len(df_user)):
            x_i = int(df_user.loc[df_user.index[idx], "x"])
            n_i = int(df_user.loc[df_user.index[idx], "n"])
            r = wald_ci_diff_proportions(base_x, base_n, x_i, n_i, alpha=alpha, alternative=alternative)
            pA, pB = r["p_a"], r["p_b"]
            diff, ci, pval = r["diff"], r["ci"], r["p_value"]
            lift = safe_div(diff, pA)
            f_i = n_i / (n_i + base_n)
            eff_i = 4 * f_i * (1 - f_i)
            results.append({
                "Variante": str(df_user.loc[df_user.index[idx], "Variante"]),
                "n": n_i,
                "x": x_i,
                "p": pB,
                "f_B": f_i,
                "eff_alloc": eff_i,
                "Diff vs A": diff,
                "Lift vs A": lift,
                "p_raw": pval,
            })
        if results:
            # Multi-tests entre variantes
            p_multi = p_adjust([r["p_raw"] for r in results], method=mt_correction)
            # Ajustement peeking ensuite
            p_final = [adjust_peeking_p(p, looks, peek_method) if peek_method != "aucun" else p for p in p_multi]
            for i, (pm, pf) in enumerate(zip(p_multi, p_final)):
                results[i]["p_adj_multi"] = pm
                results[i]["p_adj_peeking"] = pf
                results[i]["Signif (finale)"] = (pf < alpha)
            df_res = pd.DataFrame(results)
            st.dataframe(df_res, use_container_width=True)
            st.caption(f"Ajustement peeking : méthode={peek_method}, lectures m={int(looks)}. Les p-valeurs affichées tiennent compte des multi-tests (\"{mt_correction}\") et du peeking.")
            # Barres des taux
            df_rates = pd.DataFrame({
                "Variante": df_user["Variante"],
                "Taux": df_user["x"] / df_user["n"],
            }).set_index("Variante")
            st.bar_chart(df_rates)

            # 🧾 Résumé visuel multi-variantes (binomiale)
            is_higher_better = (alternative != "smaller")
            if results:
                sig = [r for r in results if r.get("p_adj_peeking", r.get("p_adj_multi", 1.0)) < alpha and ((r["Diff vs A"] > 0) if is_higher_better else (r["Diff vs A"] < 0))]
                if sig:
                    winner = max(sig, key=lambda r: r["Diff vs A"]) if is_higher_better else min(sig, key=lambda r: r["Diff vs A"])
                    title = f"✅ Gagnant : {winner['Variante']}"
                    subtitle = f"p (ajustée) = {winner.get('p_adj_peeking', winner.get('p_adj_multi', float('nan'))):.4g} • Δ vs A = {winner['Diff vs A']*100:.2f} points"
                    status = "ok"
                else:
                    leader = max(results, key=lambda r: r["Diff vs A"]) if is_higher_better else min(results, key=lambda r: r["Diff vs A"])
                    best_p = min([r.get('p_adj_peeking', r.get('p_adj_multi', 1.0)) for r in results])
                    title = "Aucun gagnant (non significatif)"
                    subtitle = f"Leader provisoire : {leader['Variante']} • Δ={leader['Diff vs A']*100:.2f} points • meilleure p (ajustée)={best_p:.4g}"
                    status = "warn"
                st.markdown(f"""
                <div class=\"summary-card {status}\">
                  <div class=\"summary-title\">{title}</div>
                  <div class=\"summary-subtitle\">{subtitle}</div>
                </div>
                """, unsafe_allow_html=True)

                # 🏆 Top 3 des variantes (lecture simple)
                is_desc = True if is_higher_better else False
                top = sorted(results, key=lambda r: r["Diff vs A"], reverse=is_desc)[:3]
                st.markdown("**Top 3 (vs contrôle)** :")
                for i, t in enumerate(top, 1):
                    st.markdown(f"{i}. **{t['Variante']}** — Δ={t['Diff vs A']*100:.2f} points • p (ajustée)={t.get('p_adj_peeking', t.get('p_adj_multi', float('nan'))):.4g}")

        st.markdown("#### 📐 Plan d'échantillonnage multi-variantes (FWER)")
        with st.expander("Configurer le plan (proportions)", expanded=False):
            p0_mv = st.number_input("Taux baseline p₀", min_value=0.0, max_value=1.0, value=float((df_user.loc[df_user.index[0],"x"]) / max(1, df_user.loc[df_user.index[0],"n"])) if len(df_user)>0 else 0.1, step=0.001, format="%.3f", key="p0_mv")
            mde_rel_mv = st.number_input("MDE relatif (ex: 0.05 = +5%)", min_value=0.0001, max_value=1.0, value=0.05, step=0.005, format="%.3f", key="mde_rel_mv")
            m_mv = st.number_input("Nombre de variantes (hors contrôle)", min_value=1, value=max(1, len(df_user)-1), step=1, key="m_mv")
            alloc_mv = st.selectbox("Allocation", options=["optimal", "egalitaire"], index=0, help="Optimal ≈ contrôle √m fois plus grand que chaque variante", key="alloc_mv")
            beta_mv = st.number_input("β (1−puissance)", min_value=0.01, max_value=0.5, value=0.2, step=0.01, format="%.2f", key="beta_mv")
            meth_mv = st.selectbox("Méthode FWER pour le dimensionnement", options=["bonferroni", "holm"], index=0, key="meth_mv")
            plan_mv = plan_multi_proportions(p0_mv, mde_rel_mv, m_mv, alpha=alpha, beta=beta_mv, allocation=alloc_mv, method=meth_mv)
            st.write(f"**n contrôle** = {plan_mv['n_control']:,} • **n par variante** = {plan_mv['n_variant']:,} • **N total** = {plan_mv['N_total']:,}")
            st.caption(f"Ajustement conservateur α* = {plan_mv['alpha_star']:.4f} (m = {int(m_mv)} comparaisons). Ratio optimal contrôle/variante ≈ √m = {plan_mv['ratio_ctrl_per_var']:.3f}.")
            df_arms = pd.DataFrame({"Bras": list(plan_mv["arms"].keys()), "n": list(plan_mv["arms"].values())}).set_index("Bras")
            st.bar_chart(df_arms)

elif multi_mode and metric_type == "Moyenne continue":
    st.subheader("Mode multi-variantes — Moyenne continue")

    df_init = pd.DataFrame({
        "Variante": ["A (contrôle)", "B", "C"],
        "n": [200, 180, 180],
        "mean": [100.0, 104.0, 102.0],
        "sd": [15.0, 15.0, 15.0],
    })
    df_user = st.data_editor(df_init, num_rows="dynamic", use_container_width=True, key="multi_means")

    df_user = df_user.dropna()
    df_user["n"] = df_user["n"].astype(int)
    df_user["mean"] = df_user["mean"].astype(float)
    df_user["sd"] = df_user["sd"].astype(float)

    if len(df_user) < 2:
        st.info("Ajoutez au moins 2 variantes (dont un contrôle).")
    else:
        # Pairwise Welch vs contrôle
        base = df_user.iloc[0]
        results = []
        for idx in range(1, len(df_user)):
            row = df_user.iloc[idx]
            r = welch_test_and_ci(base.mean, base.sd, int(base.n), row.mean, row.sd, int(row.n), alpha=alpha, alternative=alternative)
            f_i = int(row.n) / (int(row.n) + int(base.n))
            eff_i = 4 * f_i * (1 - f_i)
            results.append({
                "Variante": row.Variante,
                "n": int(row.n),
                "mean": float(row.mean),
                "sd": float(row.sd),
                "f_B": f_i,
                "eff_alloc": eff_i,
                "Diff vs A": r["diff"],
                "CI_low": r["ci"][0],
                "CI_high": r["ci"][1],
                "p_raw": r["p_value"],
            })
        p_multi = p_adjust([r["p_raw"] for r in results], method=mt_correction)
        p_final = [adjust_peeking_p(p, looks, peek_method) if peek_method != "aucun" else p for p in p_multi]
        for i, (pm, pf) in enumerate(zip(p_multi, p_final)):
            results[i]["p_adj_multi"] = pm
            results[i]["p_adj_peeking"] = pf
            results[i]["Signif (finale)"] = (pf < alpha)
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        # Barres des moyennes
        st.bar_chart(df_user.set_index("Variante")["mean"])

        # 🧾 Résumé visuel multi-variantes (moyennes)
        is_higher_better = (alternative != "smaller")
        if results:
            sig = [r for r in results if r.get("p_adj_peeking", r.get("p_adj_multi", 1.0)) < alpha and ((r["Diff vs A"] > 0) if is_higher_better else (r["Diff vs A"] < 0))]
            if sig:
                winner = max(sig, key=lambda r: r["Diff vs A"]) if is_higher_better else min(sig, key=lambda r: r["Diff vs A"])
                title = f"✅ Gagnant : {winner['Variante']}"
                subtitle = f"p (ajustée) = {winner.get('p_adj_peeking', winner.get('p_adj_multi', float('nan'))):.4g} • Δ vs A = {winner['Diff vs A']:.2f}"
                status = "ok"
            else:
                leader = max(results, key=lambda r: r["Diff vs A"]) if is_higher_better else min(results, key=lambda r: r["Diff vs A"])
                best_p = min([r.get('p_adj_peeking', r.get('p_adj_multi', 1.0)) for r in results])
                title = "Aucun gagnant (non significatif)"
                subtitle = f"Leader provisoire : {leader['Variante']} • Δ={leader['Diff vs A']:.2f} • meilleure p (ajustée)={best_p:.4g}"
                status = "warn"
            st.markdown(f"""
            <div class=\"summary-card {status}\">
              <div class=\"summary-title\">{title}</div>
              <div class=\"summary-subtitle\">{subtitle}</div>
            </div>
            """, unsafe_allow_html=True)

            # 🏆 Top 3
            is_desc = True if is_higher_better else False
            top = sorted(results, key=lambda r: r["Diff vs A"], reverse=is_desc)[:3]
            st.markdown("**Top 3 (vs contrôle)** :")
            for i, t in enumerate(top, 1):
                st.markdown(f"{i}. **{t['Variante']}** — Δ={t['Diff vs A']:.2f} • p (ajustée)={t.get('p_adj_peeking', t.get('p_adj_multi', float('nan'))):.4g}")

        st.markdown("#### 📐 Plan d'échantillonnage multi-variantes (FWER)")
        with st.expander("Configurer le plan (moyennes)", expanded=False):
            sd_mv = st.number_input("Écart-type estimé (sd)", min_value=0.0001, value=float(df_user["sd"].mean() if len(df_user)>0 else 15.0), step=0.1, format="%.4f", key="sd_mv")
            mde_abs_mv = st.number_input("MDE absolu (différence vs contrôle)", min_value=0.0001, value=2.0, step=0.1, format="%.4f", key="mde_abs_mv")
            m_mv = st.number_input("Nombre de variantes (hors contrôle)", min_value=1, value=max(1, len(df_user)-1), step=1, key="m_mv_means")
            alloc_mv = st.selectbox("Allocation", options=["optimal", "egalitaire"], index=0, key="alloc_mv_means")
            beta_mv = st.number_input("β (1−puissance)", min_value=0.01, max_value=0.5, value=0.2, step=0.01, format="%.2f", key="beta_mv_means")
            meth_mv = st.selectbox("Méthode FWER pour le dimensionnement", options=["bonferroni", "holm"], index=0, key="meth_mv_means")
            plan_mv = plan_multi_means(sd_mv, mde_abs_mv, m_mv, alpha=alpha, beta=beta_mv, allocation=alloc_mv, method=meth_mv)
            st.write(f"**n contrôle** = {plan_mv['n_control']:,} • **n par variante** = {plan_mv['n_variant']:,} • **N total** = {plan_mv['N_total']:,}")
            st.caption(f"Ajustement conservateur α* = {plan_mv['alpha_star']:.4f} (m = {int(m_mv)} comparaisons). Ratio optimal contrôle/variante ≈ √m = {plan_mv['ratio_ctrl_per_var']:.3f}.")
            df_arms = pd.DataFrame({"Bras": list(plan_mv["arms"].keys()), "n": list(plan_mv["arms"].values())}).set_index("Bras")
            st.bar_chart(df_arms)

# ======================================================================================
# MODE 2-VARIANTES (comme avant) + COMPARATIF VISUEL 50/50 vs ALLOCATION CHOISIE
# ======================================================================================
else:
    if metric_type == "Taux de conversion (binomiale)":
        col1, col2 = st.columns(2)
        with col1:
            n_a = st.number_input("Visiteurs A (nA)", min_value=1, value=1000, step=1, help="Nombre total d'utilisateurs exposés à A")
            x_a = st.number_input("Conversions A (xA)", min_value=0, value=100, step=1, help="Nombre d'utilisateurs ayant converti en A")
        with col2:
            n_b = st.number_input("Visiteurs B (nB)", min_value=1, value=1000, step=1, help="Nombre total d'utilisateurs exposés à B")
            x_b = st.number_input("Conversions B (xB)", min_value=0, value=120, step=1, help="Nombre d'utilisateurs ayant converti en B")

        # Sanity checks — on protège contre des valeurs incohérentes
        x_a = min(x_a, n_a)
        x_b = min(x_b, n_b)

        res = wald_ci_diff_proportions(x_a, n_a, x_b, n_b, alpha=alpha, alternative=alternative)
        p_a, p_b = res["p_a"], res["p_b"]
        diff, ci_wald = res["diff"], res["ci"]
        z, p_value = res["z"], res["p_value"]

        # Badges de qualité d'échantillon (lisibilité)
        f_obs = n_b / (n_a + n_b)
        low_counts = (x_a < 5 or x_b < 5)
        extreme_rates = (p_a < 0.01 or p_a > 0.99 or p_b < 0.01 or p_b > 0.99)
        if low_counts or extreme_rates or (abs(f_obs-0.5) > 0.15):
            st.warning("Effectifs/ratios extrêmes ou allocation inégale marquée : préférez IC **Wilson/Newcombe** et prudence sur le Wald.")

        st.markdown("### 2) Que disent les résultats ? — taux de conversion (binomiale)")

        # Tuiles KPI synthétiques (plus lisibles pour non-experts)
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric("Taux A", fmt_pct(p_a, 2))
        with kpi2:
            st.metric("Taux B", fmt_pct(p_b, 2))
        with kpi3:
            st.metric("Lift relatif B vs A", fmt_pct(safe_div(diff, p_a), 2))

        # IC : Wald ou Newcombe (auto/option)
        use_newcombe = st.checkbox("Utiliser IC robustes Newcombe/Wilson", value=(low_counts or extreme_rates or abs(f_obs-0.5)>0.15))
        ci = newcombe_ci_diff(x_a, n_a, x_b, n_b, alpha) if use_newcombe else ci_wald
        ci_label = "IC (Newcombe/Wilson)" if use_newcombe else "IC (Wald)"

        # Ajustement peeking
        p_peek = adjust_peeking_p(p_value, looks, peek_method) if peek_method != "aucun" else p_value

        # Affichage
        st.write(f"**Différence absolue (B−A)** = {diff:.4f}")
        st.write(f"**z** = {z:.3f}  •  **p-valeur** = {p_value:.4g}  •  **p-ajustée peeking** = {p_peek:.4g}  •  Hypothèse = `{alternative}`")
        st.write(f"**{ci_label} {(1-alpha)*100:.1f}%** sur (B−A) : [{ci[0]:.4f}, {ci[1]:.4f}]")

        # 🧾 Résumé visuel clair (A/B binomiale)
        sig = (p_peek < alpha) if not np.isnan(p_peek) else False
        is_higher_better = (alternative != "smaller")
        leader = "B" if ((p_b > p_a) if is_higher_better else (p_b < p_a)) else "A"
        eff_abs = (diff if leader == "B" else -diff)
        eff_rel = safe_div(eff_abs, p_a)
        status = "ok" if sig else "warn"
        title = f"✅ Gagnant : Variante {leader}" if sig else "Aucun gagnant (non significatif)"
        subtitle = f"p (ajustée) = {p_peek:.4g} • Écart ≈ {eff_abs*100:.2f} points ({fmt_pct(eff_rel)}) • {ci_label} : [{ci[0]*100:.2f}; {ci[1]*100:.2f}]"
        st.markdown(f"""
        <div class=\"summary-card {status}\">
          <div class=\"summary-title\">{title}</div>
          <div class=\"summary-subtitle\">{subtitle}</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📘 Explication simple"):
            st.markdown("""
            - **Gagnant** : la variante avec le **meilleur taux** (si l'objectif est d'augmenter).
            - **p (ajustée)** : on tient compte du **peeking** (regarder souvent) ⇒ seuil plus **strict**.
            - **Écart** : différence en **points de pourcentage** et en **%** vs A (lift).
            - **Fourchette plausible** : la vraie différence a de bonnes chances d'être **dans l'IC**.
            *100:.1f}%** sur (B−A) : [{ci[0]:.4f}, {ci[1]:.4f}]")""")

        significant_final = (p_peek < alpha) if not np.isnan(p_peek) else False
        eff = 4 * f_obs * (1 - f_obs)
        infl = 1 / eff if eff > 0 else np.nan

        qual = "excellente" if eff >= 0.9 else ("bonne" if eff >= 0.75 else ("moyenne" if eff >= 0.6 else "faible"))
        if significant_final:
            st.success(f"✅ **Fiabilité** : *Significatif après peeking* (p_adj={p_peek:.4g}). Allocation {qual} (eff={eff:.2f}, inflation vs 50/50 ×{infl:.2f}).")
        else:
            st.info(f"ℹ️ **Fiabilité** : *Non significatif après peeking* (p_adj={p_peek:.4g}). Allocation {qual} (eff={eff:.2f}, inflation ×{infl:.2f}).")

        # 📎 Panel allocation & efficacité
        st.markdown("#### 📎 Allocation observée & efficacité statistique")
        a1, a2, a3 = st.columns(3)
        with a1:
            st.metric("Part B", fmt_pct(f_obs))
        with a2:
            st.metric("Efficacité d'allocation", f"{eff:.3f}", help="1.00 au mieux (50/50). ≈ 4 f (1-f)")
        with a3:
            st.metric("Inflation vs 50/50", f"×{infl:.3f}")
        st.bar_chart(pd.DataFrame({"Bras": ["A", "B"], "Visiteurs": [n_a, n_b]}).set_index("Bras"))
        st.caption(f"Réglages peeking : méthode={peek_method}, lectures m={int(looks)}, durée≈{dur_jours} j — fréquence='{freq}'.")

        st.divider()
        st.markdown("### 3) Le test était-il assez puissant ? (post hoc) (≈)")
        power = posthoc_power_proportions(p_a, p_b, n_a, n_b, alpha=alpha, alternative=alternative)
        st.write(f"Puissance post hoc ≈ {power:.3f}")

        st.divider()
        st.markdown("### 4) Combien de données faut-il pour la prochaine fois ?")
        st.info(
                "💡 **Binomiale** : nombre minimal par variante pour un MDE donné. Un écart à 50/50 augmente N_total; surveiller trop souvent augmente la sévérité (p-vals ajustées)."
        )
        col3, col4 = st.columns(2)
        with col3:
            p0 = st.number_input("Taux baseline attendu p₀", min_value=0.0, max_value=1.0, value=float(p_a if not np.isnan(p_a) else 0.1), step=0.001, format="%.3f")
            mde_rel = st.number_input("MDE relatif (ex: 0.05 = +5%)", min_value=0.0001, max_value=1.0, value=0.05, step=0.005, format="%.3f")
        with col4:
            beta_target = st.number_input("β (1−puissance)", min_value=0.01, max_value=0.5, value=0.2, step=0.01, format="%.2f")

        # Cas 50/50 (comme avant)
        n_per_group = sample_size_proportions(p0, mde_rel, alpha=alpha, beta=beta_target, alternative=alternative)
        N_equal = 2 * n_per_group
        st.write(f"**n par variante (≈, 50/50)** : {n_per_group:,}")

        # 🆕 Répartition inégale (ex. 80/20) + 📊 comparatif visuel
        st.markdown("#### 🆕 Plan avec répartition inégale (40/60, 80/20, etc.)")
        f_B = st.slider("Part de trafic vers B", min_value=0.05, max_value=0.95, value=float(round(f_obs,2)), step=0.05, help="Ex. 0.60 = 60 % des utilisateurs en B (40 % en A)")
        plan_uneq = sample_size_proportions_unequal(p0, mde_rel, f_B, alpha=alpha, beta=beta_target, alternative=alternative)
        infl_plan = plan_uneq["N_total"] / N_equal if N_equal > 0 else np.nan

        colu1, colu2, colu3 = st.columns(3)
        with colu1:
            st.metric("Cible n_A", f"{plan_uneq['n_A']:,}")
        with colu2:
            st.metric("Cible n_B", f"{plan_uneq['n_B']:,}")
        with colu3:
            st.metric("N total", f"{plan_uneq['N_total']:,}")

        st.caption(
            f"Inflation plan vs **50/50** ≈ ×{infl_plan:.3f} (min. théorique ≈ ×{1/(4*f_B*(1-f_B)):.3f}). Répartition courante B={fmt_pct(f_obs)}."
        )

        # 📊 Comparatif visuel 50/50 vs plan choisi
        st.markdown("##### 📊 Comparatif : 50/50 vs allocation choisie")
        df_chart = pd.DataFrame({
            "Plan": ["50/50", f"{int(round((1-f_B)*100))}/{int(round(f_B*100))}"],
            "N_total_cible": [N_equal, plan_uneq["N_total"]],
        }).set_index("Plan")
        st.bar_chart(df_chart)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Inflation ×", f"{infl_plan:.3f}")
        with c2:
            st.metric("Δ N total", f"{plan_uneq['N_total'] - N_equal:,}")

        # Export CSV récap
        df = pd.DataFrame({
            "metrique": ["binomiale"],
            "n_A": [n_a], "x_A": [x_a], "p_A": [p_a],
            "n_B": [n_b], "x_B": [x_b], "p_B": [p_b],
            "diff_B-A": [diff], "p_value": [p_value], "p_adj_peeking": [p_peek],
            "IC_low": [ci[0]], "IC_high": [ci[1]],
            "alpha": [alpha], "alternative": [alternative],
            "puissance_posthoc": [power],
            "allocation_obs_B": [f_obs], "effic_alloc_obs": [eff], "inflation_obs": [infl],
            "looks": [looks], "peek_method": [peek_method], "duree_jours": [dur_jours], "freq": [freq],
            "n_par_variante_pour_MDE_50_50": [n_per_group],
            "plan_unequal_n_A": [plan_uneq['n_A']],
            "plan_unequal_n_B": [plan_uneq['n_B']],
            "plan_unequal_N_total": [plan_uneq['N_total']],
            "plan_unequal_f_B": [f_B]
        })
        st.download_button("💾 Exporter résumé (CSV)", df.to_csv(index=False).encode("utf-8"), file_name="abtest_resume_binomiale.csv", mime="text/csv")

    # -------------------------------------------------------------
    # BRANCHE CONTINUE (moyenne)
    # -------------------------------------------------------------
    else:
        col1, col2 = st.columns(2)
        with col1:
            n_a = st.number_input("Taille A (nA)", min_value=2, value=200, step=1, help="Nombre d'observations en A")
            mean_a = st.number_input("Moyenne A", value=100.0, step=1.0, format="%.4f")
            sd_a = st.number_input("Écart-type A (sdA)", min_value=0.0, value=15.0, step=0.5, format="%.4f")
        with col2:
            n_b = st.number_input("Taille B (nB)", min_value=2, value=200, step=1, help="Nombre d'observations en B")
            mean_b = st.number_input("Moyenne B", value=104.0, step=1.0, format="%.4f")
            sd_b = st.number_input("Écart-type B (sdB)", min_value=0.0, value=15.0, step=0.5, format="%.4f")

        res = welch_test_and_ci(mean_a, sd_a, n_a, mean_b, sd_b, n_b, alpha=alpha, alternative=alternative)
        diff, ci, t_stat, p_value, df_w = res["diff"], res["ci"], res["t"], res["p_value"], res["df"]

        st.markdown("### 2) Que disent les résultats ? — moyenne continue (Welch)")

        # Tuiles KPI
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Moyenne A", f"{mean_a:.2f}")
        with k2:
            st.metric("Moyenne B", f"{mean_b:.2f}")
        with k3:
            st.metric("Diff. B−A", f"{diff:.2f}")

        # Ajustement peeking
        p_peek = adjust_peeking_p(p_value, looks, peek_method) if peek_method != "aucun" else p_value

        st.write(f"**t** = {t_stat:.3f}  (df≈{df_w:.1f})  •  **p-valeur** = {p_value:.4g}  •  **p-ajustée peeking** = {p_peek:.4g}  •  Hypothèse = `{alternative}`")
        st.write(f"**IC {(1-alpha)*100:.1f}%** sur (B−A) : [{ci[0]:.4f}, {ci[1]:.4f}]")

        # 🧾 Résumé visuel clair (A/B moyennes)
        sig = (p_peek < alpha) if not np.isnan(p_peek) else False
        is_higher_better = (alternative != "smaller")
        leader = "B" if ((mean_b > mean_a) if is_higher_better else (mean_b < mean_a)) else "A"
        eff_abs = (diff if leader == "B" else -diff)
        eff_rel = safe_div(eff_abs, mean_a)
        status = "ok" if sig else "warn"
        title = f"✅ Gagnant : Variante {leader}" if sig else "Aucun gagnant (non significatif)"
        subtitle = f"p (ajustée) = {p_peek:.4g} • Écart ≈ {eff_abs:.2f} ({fmt_pct(eff_rel)}) • IC : [{ci[0]:.2f}; {ci[1]:.2f}]"
        st.markdown(f"""
        <div class=\"summary-card {status}\">
          <div class=\"summary-title\">{title}</div>
          <div class=\"summary-subtitle\">{subtitle}</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📘 Explication simple"):
            st.markdown("""
            - **Gagnant** : la variante avec la **meilleure moyenne** (si l'objectif est d'augmenter).
            - **p (ajustée)** : on tient compte du **peeking** ⇒ seuil plus **strict**.
            - **Écart** : différence **absolue** et **relative** vs A.
            - **Fourchette plausible** : la vraie différence a de bonnes chances d'être **dans l'IC**.
            *100:.1f}%** sur (B−A) : [{ci[0]:.4f}, {ci[1]:.4f}]")""")

        significant_final = (p_peek < alpha) if not np.isnan(p_peek) else False

        # Info sur la répartition effective
        f_obs = n_b / (n_a + n_b)
        eff = 4 * f_obs * (1 - f_obs)
        infl = 1 / eff if eff > 0 else np.nan
        qual = "excellente" if eff >= 0.9 else ("bonne" if eff >= 0.75 else ("moyenne" if eff >= 0.6 else "faible"))
        if significant_final:
            st.success(f"✅ **Fiabilité** : *Significatif après peeking* (p_adj={p_peek:.4g}). Allocation {qual} (eff={eff:.2f}, inflation ×{infl:.2f}).")
        else:
            st.info(f"ℹ️ **Fiabilité** : *Non significatif après peeking* (p_adj={p_peek:.4g}). Allocation {qual} (eff={eff:.2f}, inflation ×{infl:.2f}).")

        st.caption(f"Réglages peeking : méthode={peek_method}, lectures m={int(looks)}, durée≈{dur_jours} j — fréquence='{freq}'.")

        with st.expander("📝 Comment lire ces résultats ?"):
            st.markdown(
                f"""
                - **Significativité** : p_adj (peeking) = `{p_peek:.4g}` {"< α ⇒ résultat significatif." if significant_final else ">= α ⇒ pas de preuve suffisante de différence."}
                - **Effet** : la différence estimée est **{diff:.2f}** (B − A).
                - **Incertitude** : la vraie différence est probablement entre **{ci[0]:.2f}** et **{ci[1]:.2f}**.
                - **Allocation** : B = **{fmt_pct(f_obs)}** (efficacité≈{eff:.3f}). Un éloignement du 50/50 augmente l'échantillon total nécessaire.
                """
            )

        st.divider()
        st.markdown("### 3) Le test était-il assez puissant ? (post hoc) (≈)")
        power = posthoc_power_means(mean_a, sd_a, n_a, mean_b, sd_b, n_b, alpha=alpha, alternative=alternative)
        st.write(f"Puissance post hoc ≈ {power:.3f}")

        st.divider()
        st.markdown("### 4) Combien de données faut-il pour la prochaine fois ?")
        st.info(
            "💡 **Moyenne continue** : volume requis par groupe pour détecter le MDE. Un monitoring fréquent entraîne un ajustement des seuils (p-vals)."
        )
        col3, col4 = st.columns(2)
        with col3:
            sd_pooled = st.number_input("Écart-type *pooled* attendu", min_value=0.0001, value=float(np.sqrt((sd_a**2 + sd_b**2)/2)), step=0.1, format="%.4f")
            mde_abs = st.number_input("MDE absolu (différence à détecter)", min_value=0.0001, value=2.0, step=0.1, format="%.4f")