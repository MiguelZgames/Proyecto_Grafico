"""
MÓDULO CENTRAL DE LÓGICA Y ANÁLISIS DE AGENTES
==============================================

Este módulo consolida la lógica de análisis (scoring, categorización, crédito, etc.)
y, de forma CRÍTICA, mantiene como fuente única de verdad la lógica original de
cálculo de las 11 métricas (0-10) del archivo `dashboard_agentes_limpio1.py`.

Regla:
- NO reimplementar fórmulas/umbrales/guardas de métricas en este módulo.
- El cálculo mensual se deriva SOLO filtrando datos y llamando a la función original.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ============================================================================
# FUENTE ORIGINAL: MÉTRICAS + HELPERS (COPIA EXACTA DEL DASHBOARD)
# ----------------------------------------------------------------------------
# Nota: esta sección se copia exactamente desde `dashboard_agentes_limpio1.py`
# para evitar dependencias de import en despliegues donde el dashboard no esté.
# ============================================================================

# PESOS DE LAS MÉTRICAS (según .tex)
PESOS_METRICAS = {
    'rentabilidad': 0.12,
    'volumen': 0.15,
    'fidelidad': 0.15,
    'estabilidad': 0.12,
    'crecimiento': 0.10,
    'eficiencia_casino': 0.08,
    'eficiencia_deportes': 0.08,
    'eficiencia_conversion': 0.11,
    'tendencia': 0.04,
    'diversificacion': 0.03,
    'calidad_jugadores': 0.02
}

def calcular_percentil_25(historial):
    """Calcula el percentil 25 del historial"""
    if len(historial) == 0:
        return 0
    return np.percentile(historial, 25)

def calcular_coeficiente_variacion(historial):
    """Calcula el coeficiente de variación (CV)"""
    if len(historial) < 2:
        return 0
    media = np.mean(historial)
    if media == 0:
        return 0
    desv_std = np.std(historial, ddof=1)
    return desv_std / abs(media)

def calcular_factor_volatilidad(cv_log):
    """Calcula el factor de volatilidad basado en el CV logarítmico"""
    if cv_log < 0.2:
        return 1.00, "Baja volatilidad"
    elif cv_log < 0.4:
        return 0.85, "Moderada"
    elif cv_log < 0.6:
        return 0.70, "Alta"
    elif cv_log < 0.8:
        return 0.55, "Muy alta"
    else:
        return 0.40, "Extrema"

def calcular_tendencia_lineal(historial):
    """Calcula la tendencia usando regresión lineal"""
    if len(historial) < 2:
        return 0
    
    n = len(historial)
    t = np.arange(1, n + 1)
    
    numerador = n * np.sum(t * historial) - np.sum(t) * np.sum(historial)
    denominador = n * np.sum(t**2) - (np.sum(t))**2
    
    if denominador == 0:
        return 0
    
    return numerador / denominador

def calcular_factor_tendencia(tendencia):
    """Calcula el factor de tendencia"""
    if tendencia > 5000:
        return 1.15, "Crecimiento fuerte"
    elif tendencia > 0:
        return 1.05, "Crecimiento moderado"
    elif tendencia >= -5000:
        return 0.95, "Estancamiento"
    else:
        return 0.80, "Decrecimiento"

def calcular_metricas_agente(df_agente, total_jugadores_global=1):
    """Calcula las 11 métricas para un agente usando las columnas del dashboard"""
    metricas = {}
    
    # Convertir fecha y agrupar por mes
    if 'creado' not in df_agente.columns:
        return {k: 0 for k in PESOS_METRICAS.keys()}, pd.DataFrame()
    
    df_agente = df_agente.copy()
    df_agente['creado'] = pd.to_datetime(df_agente['creado'], errors='coerce')
    df_agente = df_agente.dropna(subset=['creado'])
    
    if len(df_agente) == 0:
        return {k: 0 for k in PESOS_METRICAS.keys()}, pd.DataFrame()
    
    df_agente['mes'] = df_agente['creado'].dt.to_period('M')
    
    # Agrupar por mes
    df_mensual = df_agente.groupby('mes').agg({
        'calculo_ngr': 'sum',  # NGR = comisión
        'num_depositos': 'sum',  # Número de depósitos
        'num_retiros': 'sum',  # Número de retiros
        'total_depositos': 'sum',
        'total_retiros': 'sum',
        'apuestas_deportivas_ggr': 'sum',
        'casino_ggr': 'sum',
        'jugador_id': 'nunique'
    }).reset_index()
    
    total_ngr = df_mensual['calculo_ngr'].sum()
    total_depositos = df_mensual['total_depositos'].sum()
    total_num_depositos = df_mensual['num_depositos'].sum()  # Suma de n_deposito
    total_ggr_deportes = df_mensual['apuestas_deportivas_ggr'].sum()
    total_ggr_casino = df_mensual['casino_ggr'].sum()
    total_ggr = total_ggr_deportes + total_ggr_casino
    
    # Calcular apuestas aproximadas (GGR / margen estimado)
    margen_estimado = 0.05  # 5% margen promedio
    total_apuestas_deportes = total_ggr_deportes / margen_estimado if total_ggr_deportes > 0 else 0
    total_apuestas_casino = total_ggr_casino / margen_estimado if total_ggr_casino > 0 else 0
    total_apuestas = total_apuestas_deportes + total_apuestas_casino
    
    # 1. Rentabilidad de Comisión (20%) - Combina % de rentabilidad con volumen absoluto
    if total_depositos > 0:
        rentabilidad_pct = (total_ngr / total_depositos) * 100
        
        # Score base por porcentaje (0-7 puntos)
        if rentabilidad_pct >= 8.0:
            score_pct = 7.0
        elif rentabilidad_pct >= 6.0:
            score_pct = 5.5
        elif rentabilidad_pct >= 4.0:
            score_pct = 4.0
        else:
            score_pct = max(0, min(7, rentabilidad_pct * 1.75))
        
        # Bonus por volumen absoluto de comisión (0-3 puntos)
        # $1K=0.5, $5K=1.5, $10K=2.0, $20K=2.5, $50K=3.0
        if total_ngr > 0:
            score_volumen = min(3.0, np.log10(total_ngr + 1) * 0.75)
        else:
            score_volumen = 0
        
        metricas['rentabilidad'] = score_pct + score_volumen
    else:
        metricas['rentabilidad'] = 0
    
    # 2. Volumen de Negocio (15%) - Basado en número de transacciones (n_deposito + n_retiro)
    total_transacciones = df_mensual['num_depositos'].sum() + df_mensual['num_retiros'].sum()
    if total_transacciones > 0:
        # Escala logarítmica: 10 trans=2.3, 50=3.9, 100=4.6, 500=6.2, 1000=6.9, 5000=8.5
        volumen = np.log10(total_transacciones + 1) * 2.3
        metricas['volumen'] = min(10, max(0, volumen))
    else:
        metricas['volumen'] = 0
    
    # 3. Fidelidad de Jugadores (15%) - Proporción de jugadores del agente respecto al total
    jugadores_agente = df_agente['jugador_id'].nunique()
    if total_jugadores_global > 0:
        proporcion_jugadores = (jugadores_agente / total_jugadores_global) * 100
        # Escala: 1%=2.5, 2%=5.0, 5%=7.5, 10%=10.0
        metricas['fidelidad'] = min(10, proporcion_jugadores * 2.5)
    else:
        metricas['fidelidad'] = 0
    
    # 4. Estabilidad Financiera (10%) - EF_log = 1 - (σ_log / μ_log)
    # Usar comisiones mensuales (calculo_ngr = comis_calculada) con transformación logarítmica
    comisiones_mensuales = df_mensual['calculo_ngr'].values
    if len(comisiones_mensuales) > 1:
        # Transformación logarítmica: x'_i = ln(x_i + |min(x)| + 1)
        min_comision = np.min(comisiones_mensuales)
        comisiones_log = np.log(comisiones_mensuales + abs(min_comision) + 1)
        
        # Calcular CV con datos transformados
        cv_log = calcular_coeficiente_variacion(comisiones_log)
        # EF_log = 1 - CV_log
        ef = 1 - cv_log
        
        # Convertir EF a escala 0-10 según rangos
        # EF ≥ 0.8: Alta estabilidad → 8-10 puntos
        # 0.6 ≤ EF < 0.8: Estabilidad moderada → 6-8 puntos
        # 0.4 ≤ EF < 0.6: Estabilidad baja → 4-6 puntos
        # EF < 0.4: Inestabilidad alta → 0-4 puntos
        
        if ef >= 0.8:
            # Mapear [0.8, 1.0] → [8, 10]
            metricas['estabilidad'] = 8.0 + ((ef - 0.8) / 0.2) * 2.0
        elif ef >= 0.6:
            # Mapear [0.6, 0.8) → [6, 8]
            metricas['estabilidad'] = 6.0 + ((ef - 0.6) / 0.2) * 2.0
        elif ef >= 0.4:
            # Mapear [0.4, 0.6) → [4, 6]
            metricas['estabilidad'] = 4.0 + ((ef - 0.4) / 0.2) * 2.0
        elif ef >= 0:
            # Mapear [0, 0.4) → [0, 4]
            metricas['estabilidad'] = (ef / 0.4) * 4.0
        else:
            # CV > 1, muy inestable
            metricas['estabilidad'] = 0
    else:
        metricas['estabilidad'] = 5.0
    
    # 5. Crecimiento de Depósitos (10%) - Basado en número de depósitos (n_deposito)
    # Fórmula: (suma_n_deposito_mes_actual - suma_n_deposito_mes_anterior) / suma_n_deposito_mes_anterior
    # Comparar último mes con mes anterior
    if len(df_mensual) >= 2:
        depositos_actual = df_mensual.iloc[-1]['num_depositos']
        depositos_anterior = df_mensual.iloc[-2]['num_depositos']
        
        if depositos_anterior > 0:
            crecimiento_pct = ((depositos_actual - depositos_anterior) / depositos_anterior) * 100
            
            # Escala ajustada para crecimiento de número de depósitos
            if crecimiento_pct >= 20:
                metricas['crecimiento'] = 10.0
            elif crecimiento_pct >= 10:
                metricas['crecimiento'] = 8.0
            elif crecimiento_pct >= 5:
                metricas['crecimiento'] = 6.5
            elif crecimiento_pct >= 0:
                metricas['crecimiento'] = 5.0
            elif crecimiento_pct >= -10:
                metricas['crecimiento'] = 3.5
            elif crecimiento_pct >= -20:
                metricas['crecimiento'] = 2.0
            else:
                metricas['crecimiento'] = 1.0
        elif depositos_actual > 0:
            # Mes anterior sin depósitos pero actual sí tiene = crecimiento máximo
            metricas['crecimiento'] = 10.0
        else:
            # Ambos meses sin depósitos
            metricas['crecimiento'] = 0
    else:
        metricas['crecimiento'] = 5.0
    
    # 6. Eficiencia Casino (8%) - EC_casino = (n_deposito / GGR_casino) * 100
    # Menor valor = mejor eficiencia (menos depósitos necesarios por unidad de GGR)
    if total_ggr_casino > 0:
        efic_casino = (total_num_depositos / total_ggr_casino) * 100
        # Escala invertida: menor es mejor
        # < 5.5 = 10.0 (excelente), 5.5-10 = 7.5, 10-14 = 5.0, 14-20 = 3.0, > 20 = 2.0
        if efic_casino > 5.5:
            metricas['eficiencia_casino'] = 10.0
        elif efic_casino > 10:
            metricas['eficiencia_casino'] = 7.5
        elif efic_casino > 14:
            metricas['eficiencia_casino'] = 5.0
        elif efic_casino > 20:
            metricas['eficiencia_casino'] = 3.0
        elif efic_casino > 33:
            metricas['eficiencia_casino'] = 2.0
        else:
            metricas['eficiencia_casino'] = max(0, 10 - (efic_casino / 10))
    else:
        metricas['eficiencia_casino'] = 0
    
    # 7. Eficiencia Deportes (8%) - EC_deportes = (n_deposito / GGR_deportes) * 100
    # Menor valor = mejor eficiencia (menos depósitos necesarios por unidad de GGR)
    if total_ggr_deportes > 0:
        efic_deportes = (total_num_depositos / total_ggr_deportes) * 100
        # Escala invertida: menor es mejor
        # < 5.5 = 10.0 (excelente), 5.5-10 = 7.5, 10-14 = 5.0, 14-20 = 3.0, > 20 = 2.0
        if efic_deportes > 5.5:
            metricas['eficiencia_deportes'] = 10.0
        elif efic_deportes > 10:
            metricas['eficiencia_deportes'] = 7.5
        elif efic_deportes > 14:
            metricas['eficiencia_deportes'] = 5.0
        elif efic_deportes > 20:
            metricas['eficiencia_deportes'] = 3.0
        elif efic_deportes > 33:
            metricas['eficiencia_deportes'] = 2.0
        else:
            metricas['eficiencia_deportes'] = max(0, 10 - (efic_deportes / 10))
    else:
        metricas['eficiencia_deportes'] = 0
    
    # 8. Eficiencia de Conversión (5%) - EC = (GGR / Depósitos) * 100
    if total_depositos > 0:
        conversion_pct = (total_ggr / total_depositos) * 100
        if conversion_pct >= 15:
            metricas['eficiencia_conversion'] = 10.0
        elif conversion_pct >= 10:
            metricas['eficiencia_conversion'] = 7.5
        elif conversion_pct >= 7:
            metricas['eficiencia_conversion'] = 5.0
        elif conversion_pct >= 5:
            metricas['eficiencia_conversion'] = 3.0
        else:
            metricas['eficiencia_conversion'] = max(0, conversion_pct * 2)
    else:
        metricas['eficiencia_conversion'] = 0
    
    # 9. Tendencia Técnica (4%)
    if len(comisiones_mensuales) >= 3:
        tendencia = calcular_tendencia_lineal(comisiones_mensuales)
        if tendencia > 1000:
            metricas['tendencia'] = 8.0
        elif tendencia > 0:
            metricas['tendencia'] = 6.0
        elif tendencia > -1000:
            metricas['tendencia'] = 4.0
        else:
            metricas['tendencia'] = 2.0
    else:
        metricas['tendencia'] = 5.0
    
    # 10. Diversificación de Productos (3%)
    if total_apuestas > 0:
        p_casino = total_apuestas_casino / total_apuestas
        p_deportes = total_apuestas_deportes / total_apuestas
        hhi = p_casino**2 + p_deportes**2
        diversificacion = (1 - hhi) * 10
        metricas['diversificacion'] = diversificacion
    else:
        metricas['diversificacion'] = 0
    
    # 11. Calidad de Jugadores (2%)
    if jugadores_agente > 0:
        apuesta_promedio = total_apuestas / jugadores_agente
        if apuesta_promedio > 10000:
            metricas['calidad_jugadores'] = 8.0
        elif apuesta_promedio > 5000:
            metricas['calidad_jugadores'] = 6.0
        elif apuesta_promedio > 1000:
            metricas['calidad_jugadores'] = 4.0
        else:
            metricas['calidad_jugadores'] = 2.0
    else:
        metricas['calidad_jugadores'] = 0
    
    return metricas, df_mensual

def calcular_score_total(metricas):
    """Calcula el score total ponderado"""
    score = 0
    for metrica, valor in metricas.items():
        if metrica in PESOS_METRICAS:
            score += valor * PESOS_METRICAS[metrica]
    return score

def categorizar_agente(score):
    """Categoriza al agente según su score total"""
    if score >= 9.0:
        return "A+++", "Excelencia excepcional - Líderes absolutos"
    elif score >= 8.5:
        return "A++", "Excelencia alta - Top tier sobresaliente"
    elif score >= 8.0:
        return "A+", "Excelencia - Muy alto desempeño"
    elif score >= 7.0:
        return "B+++", "Consolidado superior - Buen track record"
    elif score >= 6.5:
        return "B++", "Consolidado alto - Desempeño sólido"
    elif score >= 6.0:
        return "B+", "Consolidado - Estable y confiable"
    elif score >= 5.0:
        return "C+++", "En desarrollo avanzado - Progreso visible"
    elif score >= 4.0:
        return "C++", "En desarrollo medio - Requiere mejoras"
    elif score >= 3.0:
        return "C+", "Principiante - Necesita atención"
    else:
        return "C", "Crítico - Intervención urgente"


# ============================================================================
# DERIVADOS: MÉTRICAS MENSUALES SIN DUPLICAR LÓGICA
# ============================================================================

def calcular_metricas_mensuales(df_agente: pd.DataFrame, total_jugadores_global: int = 1, mode: str = "snapshot") -> pd.DataFrame:
    """
    Construye un DataFrame mensual con las 11 métricas (0-10) y score_global,
    SIN reimplementar ninguna fórmula.

    mode:
      - "snapshot": usa SOLO filas del mes (mes == M)
      - "cumulative": usa filas hasta el mes (mes <= M)
    """
    if mode not in ("snapshot", "cumulative"):
        raise ValueError("mode debe ser 'snapshot' o 'cumulative'")

    if df_agente is None or len(df_agente) == 0 or 'creado' not in df_agente.columns:
        return pd.DataFrame(columns=["mes", *PESOS_METRICAS.keys(), "score_global"])

    df = df_agente.copy()
    df['creado'] = pd.to_datetime(df['creado'], errors='coerce')
    df = df.dropna(subset=['creado'])
    if len(df) == 0:
        return pd.DataFrame(columns=["mes", *PESOS_METRICAS.keys(), "score_global"])

    df['mes'] = df['creado'].dt.to_period('M')
    meses = sorted(df['mes'].unique())

    filas = []
    for mes in meses:
        if mode == "snapshot":
            df_f = df[df['mes'] == mes]
        else:
            df_f = df[df['mes'] <= mes]

        metricas_mes, _ = calcular_metricas_agente(df_f, total_jugadores_global)
        score_mes = calcular_score_total(metricas_mes)

        fila = {"mes": mes}
        for k in PESOS_METRICAS.keys():
            fila[k] = metricas_mes.get(k, 0)
        fila["score_global"] = score_mes
        filas.append(fila)

    return pd.DataFrame(filas)


def calcular_metricas_agente_refactor(
    df_agente: pd.DataFrame,
    total_jugadores_global: int = 1,
    monthly_mode: str = "snapshot",
    debug_validate: bool = False
):
    """
    Wrapper oficial para este módulo.

    Devuelve:
      1) metricas_globales: métricas (0-10) calculadas con la lógica original sobre df completo
      2) df_mensual_original: agregación mensual que retorna la lógica original
      3) df_metricas_mensuales: métricas por mes (snapshot/cumulative) derivadas SOLO por filtrado

    Importante: NO hay una segunda lógica.
    """
    metricas_globales, df_mensual_original = calcular_metricas_agente(df_agente, total_jugadores_global)

    df_metricas_mensuales = calcular_metricas_mensuales(
        df_agente=df_agente,
        total_jugadores_global=total_jugadores_global,
        mode=monthly_mode
    )

    if debug_validate:
        # Validación mínima: resultado global debe coincidir con la llamada directa (misma función).
        metricas_directo, _ = calcular_metricas_agente(df_agente, total_jugadores_global)
        if metricas_directo != metricas_globales:
            raise AssertionError("Validación fallida: métricas globales difieren de la fuente original.")

    return metricas_globales, df_mensual_original, df_metricas_mensuales


# Alias para compatibilidad con código existente que esperaba mensualidad.
def calcular_metricas_agente_con_mensual(df_agente, total_jugadores_global=1, monthly_mode="snapshot", debug_validate=False):
    return calcular_metricas_agente_refactor(df_agente, total_jugadores_global, monthly_mode, debug_validate)


# ============================================================================
# CRÉDITO Y PREDICCIÓN (se mantiene lo existente en este módulo)
# ============================================================================

def calcular_credito_sugerido(df_mensual, score, metricas):
    # (Lógica idéntica a metricas_agente.py)
    detalles = {'credito': 0, 'razon': 'Insuficiente data'}
    if len(df_mensual) == 0: return 0, detalles
    
    ngr = df_mensual['calculo_ngr'].values
    validos = ngr[ngr > 0]
    if len(validos) == 0: return 0, detalles
    
    p25 = np.percentile(validos, 25)
    
    # Factores
    f_score = 0.5 + 0.05 * ((score + metricas.get('estabilidad', 0))/2)
    
    norm_log = np.log(validos + abs(validos.min()) + 1)
    cv = calcular_coeficiente_variacion(norm_log)
    f_volatilidad, _ = calcular_factor_volatilidad(cv)
    
    tend = calcular_tendencia_lineal(validos)
    f_tendencia, _ = calcular_factor_tendencia(tend)
    
    total_com = validos.sum()
    if total_com >= 50000: f_vol = 1.5
    elif total_com >= 30000: f_vol = 1.3
    elif total_com >= 15000: f_vol = 1.15
    elif total_com >= 5000: f_vol = 1.0
    else: f_vol = 0.85
    
    credito = p25 * f_score * f_volatilidad * f_tendencia * f_vol
    
    # Límites
    mediana = np.median(validos)
    if p25 < 100: credito = 0
    else: credito = min(credito, 3 * mediana * f_vol)
    
    if len(validos) < 3: credito *= 0.5
    
    return round(credito, 2), {
        'p25': p25, 'f_score': f_score, 'f_volatilidad': f_volatilidad,
        'f_tendencia': f_tendencia, 'f_volumen': f_vol
    }

def predecir_ggr(df_mensual):
    if len(df_mensual) == 0: return 0
    ggr = (df_mensual['apuestas_deportivas_ggr'] + df_mensual['casino_ggr']).values
    if len(ggr) >= 3:
        return ggr[-1]*0.5 + ggr[-2]*0.3 + ggr[-3]*0.2
    elif len(ggr) == 2:
        return ggr[-1]*0.6 + ggr[-2]*0.4
    else:
        return ggr[-1]


# Alias de nombre usado en el dashboard original
def predecir_ggr_proximo_mes(df_mensual: pd.DataFrame) -> float:
    return float(predecir_ggr(df_mensual))
