
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
        'num_depositos': 'sum',
        'num_retiros': 'sum',
        'total_depositos': 'sum',
        'total_retiros': 'sum',
        'apuestas_deportivas_ggr': 'sum',
        'casino_ggr': 'sum',
        'jugador_id': 'nunique'
    }).reset_index()

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
        else: metricas['eficiencia_casino'] = max(0, 10 - (val/10))

    # 7. Eficiencia Deportes
    if total_ggr_deportes > 0:
        val = (total_num_depositos / total_ggr_deportes) * 100
        if val < 5.5: metricas['eficiencia_deportes'] = 10
        elif val < 10: metricas['eficiencia_deportes'] = 7.5
        elif val < 14: metricas['eficiencia_deportes'] = 5
        elif val < 20: metricas['eficiencia_deportes'] = 3
        elif val < 33: metricas['eficiencia_deportes'] = 2
        else: metricas['eficiencia_deportes'] = max(0, 10 - (val/10))

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

    return metricas, df_mensual

# ============================================================================
# LÓGICA DE SCORING Y CATEGORIZACIÓN
# ============================================================================

def calcular_score_total(metricas):
    return sum(v * PESOS_METRICAS.get(k, 0) for k, v in metricas.items())

def categorizar_agente(score):
    if score >= 9.0: return "A+++", "Excelencia excepcional"
    elif score >= 8.5: return "A++", "Excelencia alta"
    elif score >= 8.0: return "A+", "Excelencia"
    elif score >= 7.0: return "B+++", "Consolidado superior"
    elif score >= 6.5: return "B++", "Consolidado alto"
    elif score >= 6.0: return "B+", "Consolidado"
    elif score >= 5.0: return "C+++", "En desarrollo avanzado"
    elif score >= 4.0: return "C++", "En desarrollo medio"
    elif score >= 3.0: return "C+", "Principiante"
    else: return "C", "Crítico"

# ============================================================================
# LÓGICA DE ANÁLISIS PROFUNDO (De dashboard_agentes_limpio1.py)
# ============================================================================

def analizar_retencion_cohortes(df_agente):
    """
    Analiza la retención de jugadores mes a mes.
    Retorna DataFrame con tasas de retención.
    """
    if 'creado' not in df_agente.columns or len(df_agente) == 0:
        return None
    
    df = df_agente.copy()
    df['creado'] = pd.to_datetime(df['creado'])
    df = df.dropna(subset=['creado'])
    df['periodo'] = df['creado'].dt.to_period('M')
    
    jugadores_por_mes = {}
    for mes in sorted(df['periodo'].unique()):
        jugadores_por_mes[mes] = set(df[df['periodo'] == mes]['jugador_id'].unique())
        
    meses_ordenados = sorted(jugadores_por_mes.keys())
    resultados = []
    
    for i in range(1, len(meses_ordenados)):
        mes_ant = meses_ordenados[i-1]
        mes_act = meses_ordenados[i]
        
        set_ant = jugadores_por_mes[mes_ant]
        set_act = jugadores_por_mes[mes_act]
        
        retenidos = set_ant.intersection(set_act)
        tasa = (len(retenidos) / len(set_ant)) * 100 if len(set_ant) > 0 else 0
        
        resultados.append({
            'mes': str(mes_act),
            'jugadores_anteriores': len(set_ant),
            'jugadores_actuales': len(set_act),
            'retenidos': len(retenidos),
            'tasa_retencion': round(tasa, 2)
        })
        
    return pd.DataFrame(resultados) if resultados else None

def analizar_crecimiento_organico(df_agente):
    """
    Analiza crecimiento orgánico vs retorno.
    """
    if 'creado' not in df_agente.columns or len(df_agente) == 0:
        return None
        
    df = df_agente.copy()
    df['creado'] = pd.to_datetime(df['creado'])
    df = df.dropna(subset=['creado'])
    df['periodo'] = df['creado'].dt.to_period('M')
    
    jugadores_por_mes = {}
    for mes in sorted(df['periodo'].unique()):
        jugadores_por_mes[mes] = set(df[df['periodo'] == mes]['jugador_id'].unique())
        
    meses_ordenados = sorted(jugadores_por_mes.keys())
    historico = set()
    resultados = []
    
    for i, mes in enumerate(meses_ordenados):
        actuales = jugadores_por_mes[mes]
        
        if i == 0:
            nuevos = actuales
            regresan = set()
            historico = actuales.copy()
        else:
            nuevos = actuales - historico
            # Regresan: estuvieron antes, no el mes pasado (simplificación: en este set, regresan son los que no son nuevos)
            # Definición 'limpio1': Regresan = (Actuales - MesAnterior) - Nuevos
            mes_ant = meses_ordenados[i-1]
            anteriores = jugadores_por_mes[mes_ant]
            regresan = (actuales - anteriores) - nuevos
            historico.update(actuales)
            
        resultados.append({
            'mes': str(mes),
            'total': len(actuales),
            'nuevos': len(nuevos),
            'regresan': len(regresan),
            'pct_nuevos': round(len(nuevos)/len(actuales)*100, 1) if len(actuales) > 0 else 0
        })
        
    return pd.DataFrame(resultados) if resultados else None

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
