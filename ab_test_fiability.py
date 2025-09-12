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
    .block-container {max-width: 920px;}

    /* KPI : suivent automatiquement le thème clair/sombre */
    .stMetric {
      background: var(--background-color-secondary);
      color: var(--text-color);
    }

    /* Badges adaptatifs (thème clair/sombre) */
    .help-badge, .warn-badge, .ok-badge {
      display:inline-block; padding:2px 8px; border-radius:8px; font-size:12px;
      background: var(--background-color-secondary);
      color: var(--text-color);
    }

    /* Couleurs personnalisées pour résultats significatifs / non significatifs */
    .result-significant {
      display:inline-block; padding:4px 10px; border-radius:10px; font-weight:bold;
      background: #dcfce7;   /* vert clair */
      color: #166534;        /* texte vert foncé */
    }
    .result-nonsignificant {
      display:inline-block; padding:4px 10px; border-radius:10px; font-weight:bold;
      background: #fee2e2;   /* rouge clair */
      color: #991b1b;        /* texte rouge foncé */
    }
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
    else:
        z_crit = stats.norm.ppf(1 - alpha)

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
        "se": se, "z": z, "p_value": pval, "ci": ci
    }


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


def sample_size_proportions(p0: float, mde_rel: float, alpha: float = 0.05, beta: float = 0.2, alternative: str = "two-sided") -> int:
    """Taille d'échantillon **par variante** pour détecter un MDE relatif
    autour d'un taux baseline p0 au niveau α et avec puissance 1-β.
    """
    p1 = p0 * (1 + mde_rel)
    p1 = min(max(p1, 1e-9), 1-1e-9)
    z_alpha = stats.norm.ppf(1 - (alpha/2 if alternative == "two-sided" else alpha))
    z_beta = stats.norm.ppf(1 - beta)
    se_term = p0*(1-p0) + p1*(1-p1)
    n = ((z_alpha + z_beta)**2 * se_term) / ((p1 - p0)**2)
    return math.ceil(n)


def welch_test_and_ci(mean_a: float, sd_a: float, n_a: int, mean_b: float, sd_b: float, n_b: int, alpha: float = 0.05, alternative: str = "two-sided"):
    """Test t de Welch + IC pour différence de moyennes (B - A).
    Robuste aux variances différentes.
    """
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
    """Puissance post hoc (approx. normale) pour la différence de moyennes.
    Utile pour juger si le test était suffisamment armé pour l'effet observé.
    """
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


def sample_size_means(sd_pooled: float, mde_abs: float, alpha: float = 0.05, beta: float = 0.2, alternative: str = "two-sided") -> int:
    """Taille d'échantillon **par variante** pour détecter une différence absolue (mde_abs)
    avec un écart-type attendu sd_pooled.
    """
    z_alpha = stats.norm.ppf(1 - (alpha/2 if alternative == "two-sided" else alpha))
    z_beta = stats.norm.ppf(1 - beta)
    n = 2 * ((z_alpha + z_beta) * sd_pooled / mde_abs)**2
    return math.ceil(n)

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
        - **p-valeur** : probabilité d'observer un écart au moins aussi grand **si** A et B étaient en réalité identiques. Si p < α, la différence est **significative**.
        - **IC (intervalle de confiance)** : fourchette plausible de la vraie différence. S'il contient 0, l'effet peut être nul.
        - **Lift** : amélioration relative de B vs A (utile pour lire l'impact en %).
        - **Puissance** *(indicatif)* : capacité du test à détecter l'effet observé. <span class="warn-badge">Faible</span> si < 0.8.
        - **Taille d'échantillon** : combien de visiteurs/observations **par variante** viser pour un MDE donné.
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

st.markdown("### 1) Renseigner les données de l'A/B test")

# -------------------------------------------------------------
# BRANCHE BINOMIALE (taux de conversion)
# -------------------------------------------------------------
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
    diff, ci = res["diff"], res["ci"]
    z, p_value = res["z"], res["p_value"]

    # Badges de qualité d'échantillon (lisibilité)
    low_counts = (x_a < 5 or x_b < 5)
    extreme_rates = (p_a < 0.01 or p_a > 0.99 or p_b < 0.01 or p_b > 0.99)
    if low_counts or extreme_rates:
        st.warning("Les effectifs/ratios sont extrêmes (très peu de conversions ou ~0%/~100%). Les IC/Wald peuvent être fragiles. Envisagez Wilson/Newcombe ou un test exact.")

    st.markdown("### 2) Que disent les résultats ? — taux de conversion (binomiale)")

    # Tuiles KPI synthétiques (plus lisibles pour non-experts)
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("Taux A", fmt_pct(p_a, 2))
    with kpi2:
        st.metric("Taux B", fmt_pct(p_b, 2))
    with kpi3:
        st.metric("Lift relatif B vs A", fmt_pct(safe_div(diff, p_a), 2))

    # Détails chiffrés
    st.write(f"**Différence absolue (B−A)** = {diff:.4f}")
    st.write(f"**z** = {z:.3f}  •  **p-valeur** = {p_value:.4g}  •  Hypothèse = `{alternative}`")
    st.write(f"**IC {(1-alpha)*100:.1f}%** sur (B−A) : [{ci[0]:.4f}, {ci[1]:.4f}]")

    significant = (p_value < alpha) if not np.isnan(p_value) else False
    if significant:
        st.success("✅ Différence **significative** (on rejette H₀: pA = pB)")
    else:
        st.info("ℹ️ Différence **non significative** (on ne rejette pas H₀)")

    # Guide d'interprétation contextuel
    with st.expander("📝 Comment lire ces résultats ?"):
        st.markdown(
            f"""
            - **Significativité** : p = `{p_value:.4g}` {"< α ⇒ résultat significatif." if significant else ">= α ⇒ pas de preuve suffisante de différence."}
            - **Effet** : la meilleure estimation de l'écart est **{fmt_pct(diff)}** (absolu), soit **{fmt_pct(safe_div(diff, p_a))}** de lift.
            - **Incertitude** : la vraie différence est probablement entre **{fmt_pct(ci[0])}** et **{fmt_pct(ci[1])}**.
            - **Décision produit** : préférez la variante dont l'**IC** est majoritairement > 0 si votre objectif est d'augmenter le taux.
            """
        )

    st.divider()
    st.markdown("### 3) Le test était-il assez puissant ? (post hoc) (≈)")
    power = posthoc_power_proportions(p_a, p_b, n_a, n_b, alpha=alpha, alternative=alternative)

    if not np.isnan(power):
        if power >= 0.8:  # seuil classique de 80 %
            st.markdown(
                f"<span class='result-significant'>Puissance ≈ {power:.3f} (OK, suffisante)</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<span class='result-nonsignificant'>Puissance ≈ {power:.3f} (insuffisante)</span>",
                unsafe_allow_html=True,
            )
    else:
        st.write("Puissance non calculable (données insuffisantes)")

    with st.expander("ℹ️ Aide à la lecture de la puissance post hoc", expanded=False):
        st.markdown(
            """
            - La **puissance** est la probabilité de détecter un effet réel (**1−β**).
            - On vise souvent **≥ 80 %** pour considérer un test suffisamment armé.
            - Ici, il s'agit d'une **puissance post hoc** : calculée *après coup* à partir de l'effet **observé** et des **volumes saisis**.
              Si elle est faible (< 80 %), l'effet est peut‑être trop petit ou l'échantillon trop réduit.
            - Cette valeur est **approximative** (approx. normale) : prudence si les échantillons sont petits ou si les taux sont très proches de 0 % / 100 %.
            """
        )

    st.divider()

    st.markdown("### 4) Combien de données faut-il pour la prochaine fois ?")
    st.info(
            "💡 **Aide à la lecture (binomiale)** : Ici on estime le nombre minimal "
            "le volume **par échantillons** nécessaire pour détecter un MDE donné. "
            "Si ton volume réel est plus petit, tu risques un **faux négatif**. "
    )
    col3, col4 = st.columns(2)
    with col3:
        p0 = st.number_input("Taux baseline attendu p₀", min_value=0.0, max_value=1.0, value=float(p_a if not np.isnan(p_a) else 0.1), step=0.001, format="%.3f")
        mde_rel = st.number_input("MDE relatif (ex: 0.05 = +5%)", min_value=0.0001, max_value=1.0, value=0.05, step=0.005, format="%.3f")
    with col4:
        beta_target = st.number_input("β (1−puissance)", min_value=0.01, max_value=0.5, value=0.2, step=0.01, format="%.2f")

    n_per_group = sample_size_proportions(p0, mde_rel, alpha=alpha, beta=beta_target, alternative=alternative)
    st.write(f"**n par variante (≈)** : {n_per_group:,}")

    st.info(
    f"Avec p₀ = {p0:.2%} et MDE = {mde_rel:.1%}, viser ≈ **{n_per_group:,}** obs/variante "
    f"pour α = {alpha:.2f} et puissance ≈ {1 - beta_target:.0%}."
    )
    # Export CSV récap
    df = pd.DataFrame({
        "metrique": ["binomiale"],
        "n_A": [n_a], "x_A": [x_a], "p_A": [p_a],
        "n_B": [n_b], "x_B": [x_b], "p_B": [p_b],
        "diff_B-A": [diff], "p_value": [p_value],
        "IC_low": [ci[0]], "IC_high": [ci[1]],
        "alpha": [alpha], "alternative": [alternative],
        "puissance_posthoc": [power],
        "n_par_variante_pour_MDE": [n_per_group]
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
    diff, ci, t_stat, p_value, df = res["diff"], res["ci"], res["t"], res["p_value"], res["df"]

    st.markdown("### 2) Que disent les résultats ? — moyenne continue (Welch)")

    # Tuiles KPI
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Moyenne A", f"{mean_a:.2f}")
    with k2:
        st.metric("Moyenne B", f"{mean_b:.2f}")
    with k3:
        st.metric("Diff. B−A", f"{diff:.2f}")

    st.write(f"**t** = {t_stat:.3f}  (df≈{df:.1f})  •  **p-valeur** = {p_value:.4g}  •  Hypothèse = `{alternative}`")
    st.write(f"**IC {(1-alpha)*100:.1f}%** sur (B−A) : [{ci[0]:.4f}, {ci[1]:.4f}]")

    significant = (p_value < alpha) if not np.isnan(p_value) else False
    if significant:
        st.success("✅ Différence **significative** (on rejette H₀: μA = μB)")
    else:
        st.info("ℹ️ Différence **non significative** (on ne rejette pas H₀)")

    with st.expander("📝 Comment lire ces résultats ?"):
        st.markdown(
            f"""
            - **Significativité** : p = `{p_value:.4g}` {"< α ⇒ résultat significatif." if significant else ">= α ⇒ pas de preuve suffisante de différence."}
            - **Effet** : la différence estimée est **{diff:.2f}** (B − A).
            - **Incertitude** : la vraie différence est probablement entre **{ci[0]:.2f}** et **{ci[1]:.2f}**.
            - **Décision produit** : regardez si l'IC est majoritairement > 0 quand vous visez une hausse.
            """
        )

    st.divider()
    st.markdown("### 3) Le test était-il assez puissant ? (post hoc) (≈)")
    power = posthoc_power_means(mean_a, sd_a, n_a, mean_b, sd_b, n_b, alpha=alpha, alternative=alternative)

    # 🔴🟢 Badge rouge/vert selon la puissance
    if not np.isnan(power):
        if power >= 0.8:  # seuil classique de 80 %
            st.markdown(
                f"<span class='result-significant'>Puissance ≈ {power:.3f} (OK, suffisante)</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<span class='result-nonsignificant'>Puissance ≈ {power:.3f} (insuffisante)</span>",
                unsafe_allow_html=True,
            )
    else:
        st.write("Puissance non calculable (données insuffisantes)")

    with st.expander("ℹ️ Aide à la lecture de la puissance post hoc", expanded=False):
        st.markdown(
            """
            - La **puissance** est la probabilité de détecter un effet réel (**1−β**).
            - Cible habituelle : **≥ 80 %**.
            - Ici on calcule une **puissance post hoc** basée sur l'effet **observé** (différence de moyennes)
              et les **volumes saisis**. Si elle est faible (< 80 %), soit l'effet est trop petit, soit il faut plus d'observations.
            - Valeur **indicative** (approx. normale), moins fiable si n est petit ou si les distributions s'éloignent des hypothèses.
            """
        )

    st.divider()
    st.markdown("### 4) Combien de données faut-il pour la prochaine fois ?)")
    st.info(
        "💡 **Aide à la lecture (moyenne continue)** : Volume d’utilisateurs requis par groupe pour "
        "que le test ait de bonnes chances de repérer une différence au moins aussi grande que celle que vous jugez importante (MDE absolu).  "
        "Si le volume réel est insuffisant, le test peut manquer une vraie différence "
        "(**faux négatif**)."
    )
    col3, col4 = st.columns(2)
    with col3:
        sd_pooled = st.number_input("Écart-type *pooled* attendu", min_value=0.0001, value=float(np.sqrt((sd_a**2 + sd_b**2)/2)), step=0.1, format="%.4f")
        mde_abs = st.number_input("MDE absolu (différence à détecter)", min_value=0.0001, value=2.0, step=0.1, format="%.4f")
    with col4:
        beta_target = st.number_input("β (1−puissance)", min_value=0.01, max_value=0.5, value=0.2, step=0.01, format="%.2f")

    n_per_group = sample_size_means(sd_pooled, mde_abs, alpha=alpha, beta=beta_target, alternative=alternative)
    st.write(f"**n par variante (≈)** : {n_per_group:,}")
    st.info(
        f"Avec p₀ = {sd_pooled:.2%} et MDE = {mde_abs:.1%}, viser ≈ **{n_per_group:,}** obs/variante "
        f"pour α = {alpha:.2f} et puissance ≈ {1 - beta_target:.0%}."
    )
    # Export CSV
    df = pd.DataFrame({
        "metrique": ["continue"],
        "n_A": [n_a], "mean_A": [mean_a], "sd_A": [sd_a],
        "n_B": [n_b], "mean_B": [mean_b], "sd_B": [sd_b],
        "diff_B-A": [diff], "p_value": [p_value],
        "IC_low": [ci[0]], "IC_high": [ci[1]],
        "alpha": [alpha], "alternative": [alternative],
        "puissance_posthoc": [power],
        "n_par_variante_pour_MDE": [n_per_group]
    })
    st.download_button("💾 Exporter résumé (CSV)", df.to_csv(index=False).encode("utf-8"), file_name="abtest_resume_continue.csv", mime="text/csv")

# -------------------------------------------------------------
# NOTES MÉTHODOLOGIQUES (pédagogie)
# -------------------------------------------------------------

st.divider()
st.markdown(
    """
    #### ℹ️ Notes méthodologiques
    - **Binomiale** : test Z de différence de proportions (Wald) avec proportion *poolée* pour l'écart-type sous H₀.
    - **Continue** : test t de Welch (variances potentiellement différentes), IC basé sur la loi t.
    - **Puissance post hoc** : approximation normale sous l'effet observé (indicatif, ne remplace pas un plan a priori).
    - **Taille d'échantillon** : formules classiques (approx. normale). Pour des taux extrêmes ou de petits n, privilégiez Wilson/Newcombe, tests exacts ou des simulations.
    - **Bonnes pratiques** : durée d'expo suffisante, randomisation, absence de contamination, contrôles de saisonnalité et de multiples comparaisons.
    """
)

st.caption("Développé par un data analyst. Code source : Léo Combe")
