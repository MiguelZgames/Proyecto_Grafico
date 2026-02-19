
"""
MÓDULO CENTRAL DE LÓGICA Y ANÁLISIS DE AGENTES
==============================================

Este módulo consolida toda la lógica de cálculo de métricas, scoring, 
categorización, predicción y análisis profundo (retención, crecimiento orgánico)
del proyecto.

Reemplaza a:
- src/metricas_agente.py
- src/dashboard_agentes_limpio1.py (lógica de backend)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURACIÓN Y PESOS
# ============================================================================

PESOS_METRICAS = {
    'rentabilidad': 0.12,           # Rentabilidad de Comisión
    'volumen': 0.15,                # Volumen de Negocio
    'fidelidad': 0.15,              # Fidelidad de Jugadores
    'estabilidad': 0.12,            # Estabilidad Financiera
    'crecimiento': 0.10,            # Crecimiento de Depósitos
    'eficiencia_casino': 0.08,      # Eficiencia Casino
    'eficiencia_deportes': 0.08,    # Eficiencia Deportes
    'eficiencia_conversion': 0.11,  # Eficiencia de Conversión
    'tendencia': 0.04,              # Tendencia Técnica
    'diversificacion': 0.03,        # Diversificación de Productos
    'calidad_jugadores': 0.02       # Calidad de Jugadores
}

# ============================================================================
# FUNCIONES AUXILIARES DE CÁLCULO
# ============================================================================

def calcular_percentil_25(historial):
    if len(historial) == 0: return 0
    return np.percentile(historial, 25)

def calcular_coeficiente_variacion(historial):
    if len(historial) < 2: return 0
    media = np.mean(historial)
    if media == 0: return 0
    desv_std = np.std(historial, ddof=1)
    return desv_std / abs(media)

def calcular_factor_volatilidad(cv_log):
    if cv_log < 0.2: return 1.00, "Baja volatilidad"
    elif cv_log < 0.4: return 0.85, "Moderada"
    elif cv_log < 0.6: return 0.70, "Alta"
    elif cv_log < 0.8: return 0.55, "Muy alta"
    else: return 0.40, "Extrema"

def calcular_tendencia_lineal(historial):
    if len(historial) < 2: return 0
    n = len(historial)
    t = np.arange(1, n + 1)
    numerador = n * np.sum(t * historial) - np.sum(t) * np.sum(historial)
    denominador = n * np.sum(t**2) - (np.sum(t))**2
    if denominador == 0: return 0
    return numerador / denominador

def calcular_factor_tendencia(tendencia):
    if tendencia > 5000: return 1.15, "Crecimiento fuerte"
    elif tendencia > 0: return 1.05, "Crecimiento moderado"
    elif tendencia >= -5000: return 0.95, "Estancamiento"
    else: return 0.80, "Decrecimiento"

# ============================================================================
# LÓGICA CORE: 11 MÉTRICAS (De metricas_agente.py)
# ============================================================================

def calcular_metricas_agente(df_agente, total_jugadores_global=1):
    """Calcula las 11 métricas principales para un agente (0-10)."""
    metricas = {k: 0 for k in PESOS_METRICAS.keys()}
    
    if 'creado' not in df_agente.columns:
        return metricas, pd.DataFrame()

    df_agente = df_agente.copy()
    df_agente['creado'] = pd.to_datetime(df_agente['creado'], errors='coerce')
    df_agente = df_agente.dropna(subset=['creado'])

    if len(df_agente) == 0:
        return metricas, pd.DataFrame()

    df_agente['mes'] = df_agente['creado'].dt.to_period('M')

    # Agrupar por mes
    df_mensual = df_agente.groupby('mes').agg({
        'calculo_ngr': 'sum',
        'calculo_comision': 'sum', # Add commission aggregation
        'num_depositos': 'sum',
        'num_retiros': 'sum',
        'total_depositos': 'sum',
        'total_retiros': 'sum',
        'apuestas_deportivas_ggr': 'sum',
        'casino_ggr': 'sum',
        'jugador_id': 'nunique'
    }).reset_index().rename(columns={'jugador_id': 'active_players'})

    # Totales
    total_ngr = df_mensual['calculo_ngr'].sum()
    total_depositos = df_mensual['total_depositos'].sum()
    total_num_depositos = df_mensual['num_depositos'].sum()
    total_ggr_deportes = df_mensual['apuestas_deportivas_ggr'].sum()
    total_ggr_casino = df_mensual['casino_ggr'].sum()
    total_ggr = total_ggr_deportes + total_ggr_casino

    # Apuestas estimadas
    margen_estimado = 0.05
    total_apuestas = (total_ggr_deportes + total_ggr_casino) / margen_estimado if margen_estimado else 0
    total_apuestas_deportes = total_ggr_deportes / margen_estimado
    total_apuestas_casino = total_ggr_casino / margen_estimado

    # 1. Rentabilidad
    if total_depositos > 0:
        rentabilidad_pct = (total_ngr / total_depositos) * 100
        score_pct = 7.0 if rentabilidad_pct >= 8 else (5.5 if rentabilidad_pct >= 6 else (4.0 if rentabilidad_pct >= 4 else max(0, min(7, rentabilidad_pct * 1.75))))
        score_volumen = min(3.0, np.log10(total_ngr + 1) * 0.75) if total_ngr > 0 else 0
        metricas['rentabilidad'] = score_pct + score_volumen

    # 2. Volumen
    total_transacciones = total_num_depositos + df_mensual['num_retiros'].sum()
    metricas['volumen'] = min(10, max(0, np.log10(total_transacciones + 1) * 2.3)) if total_transacciones > 0 else 0

    # 3. Fidelidad
    jugadores_agente = df_agente['jugador_id'].nunique()
    if total_jugadores_global > 0:
        metricas['fidelidad'] = min(10, (jugadores_agente / total_jugadores_global) * 100 * 2.5)

    # 4. Estabilidad
    comisiones = df_mensual['calculo_ngr'].values
    if len(comisiones) > 1:
        min_c = np.min(comisiones)
        c_log = np.log(comisiones + abs(min_c) + 1)
        ef = 1 - calcular_coeficiente_variacion(c_log)
        if ef >= 0.8: metricas['estabilidad'] = 8 + ((ef-0.8)/0.2)*2
        elif ef >= 0.6: metricas['estabilidad'] = 6 + ((ef-0.6)/0.2)*2
        elif ef >= 0.4: metricas['estabilidad'] = 4 + ((ef-0.4)/0.2)*2
        elif ef >= 0: metricas['estabilidad'] = (ef/0.4)*4
    else:
        metricas['estabilidad'] = 5.0

    # 5. Crecimiento (Últimos 2 meses)
    if len(df_mensual) >= 2:
        dep_act = df_mensual.iloc[-1]['num_depositos']
        dep_ant = df_mensual.iloc[-2]['num_depositos']
        if dep_ant > 0:
            pct = ((dep_act - dep_ant)/dep_ant)*100
            if pct >= 20: metricas['crecimiento'] = 10
            elif pct >= 10: metricas['crecimiento'] = 8
            elif pct >= 5: metricas['crecimiento'] = 6.5
            elif pct >= 0: metricas['crecimiento'] = 5
            elif pct >= -10: metricas['crecimiento'] = 3.5
            elif pct >= -20: metricas['crecimiento'] = 2
            else: metricas['crecimiento'] = 1
        elif dep_act > 0: metricas['crecimiento'] = 10
    else:
        metricas['crecimiento'] = 5.0

    # 6. Eficiencia Casino
    if total_ggr_casino > 0:
        val = (total_num_depositos / total_ggr_casino) * 100
        if val < 5.5: metricas['eficiencia_casino'] = 10
        elif val < 10: metricas['eficiencia_casino'] = 7.5
        elif val < 14: metricas['eficiencia_casino'] = 5
        elif val < 20: metricas['eficiencia_casino'] = 3
        elif val < 33: metricas['eficiencia_casino'] = 2
        else: metricas['eficiencia_casino'] = max(1.0, 10 - (val/10)) # Min 1.0 if active

    # 7. Eficiencia Deportes
    if total_ggr_deportes > 0:
        val = (total_num_depositos / total_ggr_deportes) * 100
        if val < 5.5: metricas['eficiencia_deportes'] = 10
        elif val < 10: metricas['eficiencia_deportes'] = 7.5
        elif val < 14: metricas['eficiencia_deportes'] = 5
        elif val < 20: metricas['eficiencia_deportes'] = 3
        elif val < 33: metricas['eficiencia_deportes'] = 2
        else: metricas['eficiencia_deportes'] = max(1.0, 10 - (val/10)) # Min 1.0 if active

    # 8. Conversión
    if total_depositos > 0:
        val = (total_ggr / total_depositos) * 100
        if val >= 15: metricas['eficiencia_conversion'] = 10
        elif val >= 10: metricas['eficiencia_conversion'] = 7.5
        elif val >= 7: metricas['eficiencia_conversion'] = 5
        elif val >= 5: metricas['eficiencia_conversion'] = 3
        else: metricas['eficiencia_conversion'] = max(0, val*2)

    # 9. Tendencia
    if len(comisiones) >= 3:
        t = calcular_tendencia_lineal(comisiones)
        if t > 1000: metricas['tendencia'] = 8
        elif t > 0: metricas['tendencia'] = 6
        elif t > -1000: metricas['tendencia'] = 4
        else: metricas['tendencia'] = 2
    else:
        metricas['tendencia'] = 5

    # 10. Diversificación
    if total_apuestas > 0:
        hhi = (total_apuestas_casino/total_apuestas)**2 + (total_apuestas_deportes/total_apuestas)**2
        metricas['diversificacion'] = (1 - hhi) * 10

    # 11. Calidad Jugadores
    if jugadores_agente > 0:
        avg = total_apuestas / jugadores_agente
        if avg > 10000: metricas['calidad_jugadores'] = 8
        elif avg > 5000: metricas['calidad_jugadores'] = 6
        elif avg > 1000: metricas['calidad_jugadores'] = 4
        else: metricas['calidad_jugadores'] = 2

    # --- CÁLCULO DE SCORE MENSUAL (HISTÓRICO) ---
    scores_mensuales = []
    all_monthly_metrics = []  # Store per-month metric dicts
    
    # Pre-calculate rolling/history context if needed, but for simplicity we treat each month as a snapshot
    # for most metrics, and use simple windows for growth.
    
    for i, row in df_mensual.iterrows():
        m_score = {}
        
        # Data Points
        ngr = row['calculo_ngr']
        deps = row['total_depositos']
        num_deps = row['num_depositos']
        num_rets = row['num_retiros']
        ggr_c = row['casino_ggr']
        ggr_d = row['apuestas_deportivas_ggr']
        ggr_total = ggr_c + ggr_d
        players = row['active_players']
        
        # 1. Rentabilidad
        if deps > 0:
            rent_pct = (ngr / deps) * 100
            s_pct = 7.0 if rent_pct >= 8 else (5.5 if rent_pct >= 6 else (4.0 if rent_pct >= 4 else max(0, min(7, rent_pct * 1.75))))
            s_vol = min(3.0, np.log10(ngr + 1) * 0.75) if ngr > 0 else 0
            m_score['rentabilidad'] = s_pct + s_vol
        else:
            m_score['rentabilidad'] = 0
            
        # 2. Volumen
        txs = num_deps + num_rets
        m_score['volumen'] = min(10, max(0, np.log10(txs + 1) * 2.3)) if txs > 0 else 0
        
        # 3. Fidelidad (Vs Global Total passed to function)
        if total_jugadores_global > 0:
            m_score['fidelidad'] = min(10, (players / total_jugadores_global) * 100 * 2.5)
        else:
            m_score['fidelidad'] = 0
            
        # 4. Estabilidad (Snapshot = Neutral 5.0)
        m_score['estabilidad'] = 5.0
        
        # 5. Crecimiento (Vs Prev Month)
        if i > 0:
            prev_deps = df_mensual.iloc[i-1]['num_depositos']
            if prev_deps > 0:
                pct = ((num_deps - prev_deps) / prev_deps) * 100
                if pct >= 20: m_score['crecimiento'] = 10
                elif pct >= 10: m_score['crecimiento'] = 8
                elif pct >= 5: m_score['crecimiento'] = 6.5
                elif pct >= 0: m_score['crecimiento'] = 5
                elif pct >= -10: m_score['crecimiento'] = 3.5
                elif pct >= -20: m_score['crecimiento'] = 2
                else: m_score['crecimiento'] = 1
            elif num_deps > 0:
                m_score['crecimiento'] = 10
            else:
                m_score['crecimiento'] = 5
        else:
            m_score['crecimiento'] = 5.0
            
        # 6. Eficiencia Casino
        if ggr_c > 0:
            val = (num_deps / ggr_c) * 100
            if val > 5.5: m_score['eficiencia_casino'] = 10
            elif val > 10: m_score['eficiencia_casino'] = 7.5
            elif val > 14: m_score['eficiencia_casino'] = 5
            elif val > 20: m_score['eficiencia_casino'] = 3
            elif val > 33: m_score['eficiencia_casino'] = 2
            else: m_score['eficiencia_casino'] = max(1.0, 10 - (val/10))
        else:
             m_score['eficiencia_casino'] = 0
             
        # 7. Eficiencia Deportes
        if ggr_d > 0:
            val = (num_deps / ggr_d) * 100
            if val > 5.5: m_score['eficiencia_deportes'] = 10
            elif val > 10: m_score['eficiencia_deportes'] = 7.5
            elif val > 14: m_score['eficiencia_deportes'] = 5
            elif val > 20: m_score['eficiencia_deportes'] = 3
            elif val > 33: m_score['eficiencia_deportes'] = 2
            else: m_score['eficiencia_deportes'] = max(1.0, 10 - (val/10))
        else:
            m_score['eficiencia_deportes'] = 0
            
        # 8. Conversión
        if deps > 0:
            val = (ggr_total / deps) * 100
            if val >= 15: m_score['eficiencia_conversion'] = 10
            elif val >= 10: m_score['eficiencia_conversion'] = 7.5
            elif val >= 7: m_score['eficiencia_conversion'] = 5
            elif val >= 5: m_score['eficiencia_conversion'] = 3
            else: m_score['eficiencia_conversion'] = max(0, val*2)
        else:
            m_score['eficiencia_conversion'] = 0
            
        # 9. Tendencia (Snapshot = Neutral 5.0)
        m_score['tendencia'] = 5.0
        
        # 10. Diversificación
        # Est. bets per type
        bets_c = ggr_c / 0.05
        bets_d = ggr_d / 0.05
        total_bets = bets_c + bets_d
        if total_bets > 0:
            hhi = (bets_c/total_bets)**2 + (bets_d/total_bets)**2
            m_score['diversificacion'] = (1 - hhi) * 10
        else:
            m_score['diversificacion'] = 0
            
        # 11. Calidad
        if players > 0:
            avg = total_bets / players
            if avg > 10000: m_score['calidad_jugadores'] = 8
            elif avg > 5000: m_score['calidad_jugadores'] = 6
            elif avg > 1000: m_score['calidad_jugadores'] = 4
            else: m_score['calidad_jugadores'] = 2
        else:
            m_score['calidad_jugadores'] = 0
            
        # Calculate Weighted Score
        total_s = 0
        for k, weight in PESOS_METRICAS.items():
            total_s += m_score.get(k, 0) * weight
            
        scores_mensuales.append(total_s)
        all_monthly_metrics.append(m_score)

    df_mensual['score_global'] = scores_mensuales
    
    # Store individual metric scores per month
    for metric_key in PESOS_METRICAS.keys():
        df_mensual[metric_key] = [m.get(metric_key, 0) for m in all_monthly_metrics]
    
    # Add Clase and Risk_Safe per month
    clases = []
    risk_flags = []
    for s in scores_mensuales:
        cat, _ = categorizar_agente(s)
        clases.append(cat)
        risk_flags.append(1 if 'A' in cat or 'B' in cat else 0)
    df_mensual['Clase'] = clases
    df_mensual['Risk_Safe'] = risk_flags
    
    return metricas, df_mensual

# ============================================================================
# LÓGICA DE SCORING Y CATEGORIZACIÓN
# ============================================================================

def calcular_score_total(metricas):
    return sum(v * PESOS_METRICAS.get(k, 0) for k, v in metricas.items())

def categorizar_agente(score):
    if score >= 8.5: return "A+++", "Excelencia excepcional"
    elif score >= 8.0: return "A++", "Excelencia alta"
    elif score >= 7.5: return "A+", "Excelencia"
    elif score >= 6.5: return "B+++", "Consolidado superior"
    elif score >= 5.5: return "B++", "Consolidado alto"
    elif score >= 4.5: return "B+", "Consolidado"
    elif score >= 3.5: return "C+++", "En desarrollo avanzado"
    elif score >= 2.5: return "C++", "En desarrollo medio"
    elif score >= 1.5: return "C+", "Principiante"
    else: return "C", "Crítico"

# ============================================================================
# LÓGICA DE ANÁLISIS PROFUNDO (De dashboard_agentes_limpio1.py)
# ============================================================================



# ============================================================================
# LÓGICA DE CRÉDITO Y PREDICCIÓN
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
