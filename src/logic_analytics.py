"""
MÓDULO CENTRAL DE LÓGICA Y ANÁLISIS DE AGENTES
==============================================

Este módulo consolida la lógica de análisis (scoring, categorización, crédito, etc.)
alineado 100% con la versión en Julia (Métricas_agentes_Clasificación.jl).

Regla:
- El cálculo mensual se deriva SOLO filtrando datos hasta el mes de evaluación,
  mientras que las métricas puntuales filtran explícitamente solo por el mes.
"""

import numpy as np
import pandas as pd
from datetime import datetime

# ============================================================================
# CONSTANTES - PESOS DE LAS MÉTRICAS
# ============================================================================

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

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def calcular_percentil_25(historial):
    if len(historial) == 0:
        return 0.0
    return np.percentile(historial, 25)

def calcular_coeficiente_variacion(historial):
    if len(historial) < 2:
        return 0.0
    media = np.mean(historial)
    if media == 0:
        return 0.0
    desv_std = np.std(historial, ddof=1)
    return desv_std / abs(media)

def calcular_factor_volatilidad(cv_log):
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
    if len(historial) < 2:
        return 0.0
    
    n = len(historial)
    t = np.arange(1, n + 1)
    
    numerador = n * np.sum(t * historial) - np.sum(t) * np.sum(historial)
    denominador = n * np.sum(t**2) - (np.sum(t))**2
    
    if denominador == 0:
        return 0.0
    
    return numerador / denominador

def calcular_factor_tendencia(tendencia):
    if tendencia > 5000:
        return 1.15, "Crecimiento fuerte"
    elif tendencia > 0:
        return 1.05, "Crecimiento moderado"
    elif tendencia >= -5000:
        return 0.95, "Estancamiento"
    else:
        return 0.80, "Decrecimiento"

# ============================================================================
# CÁLCULO DE LAS 11 MÉTRICAS
# ============================================================================

def calcular_metricas_agente(df_agente: pd.DataFrame, total_jugadores_global: int = 1, mes_evaluacion=None):
    metricas = {k: 0.0 for k in PESOS_METRICAS.keys()}
    
    if df_agente is None or len(df_agente) == 0 or 'creado' not in df_agente.columns:
        return metricas, pd.DataFrame()
        
    df_agente = df_agente.copy()
    df_agente['creado'] = pd.to_datetime(df_agente['creado'], errors='coerce')
    df_agente = df_agente.dropna(subset=['creado'])
    
    if len(df_agente) == 0:
        return metricas, pd.DataFrame()
        
    df_agente['mes'] = df_agente['creado'].dt.to_period('M')
    
    # Agrupar por mes
    agg_dict = {
        'calculo_ngr': 'sum',
        'num_depositos': 'sum',
        'num_retiros': 'sum',
        'total_depositos': 'sum',
        'total_retiros': 'sum',
        'apuestas_deportivas_ggr': 'sum',
        'casino_ggr': 'sum',
        'jugador_id': 'nunique'
    }
    if 'tickets_deportes' in df_agente.columns:
        agg_dict['tickets_deportes'] = 'sum'
    if 'tickets_casino' in df_agente.columns:
        agg_dict['tickets_casino'] = 'sum'
        
    if 'total_apuesta_deportiva' in df_agente.columns:
        agg_dict['total_apuesta_deportiva'] = 'sum'
    if 'total_apuesta_casino' in df_agente.columns:
        agg_dict['total_apuesta_casino'] = 'sum'
        
    df_mensual = df_agente.groupby('mes').agg(agg_dict).reset_index()
    df_mensual = df_mensual.rename(columns={'jugador_id': 'jugador_id_unique'})
    
    for c in ['total_apuesta_deportiva', 'total_apuesta_casino']:
        if c in df_mensual.columns:
            df_mensual[c] = pd.to_numeric(df_mensual[c], errors='coerce').fillna(0.0).replace([np.inf, -np.inf], 0.0)
    
    df_mensual = df_mensual.sort_values('mes')
    
    if mes_evaluacion is None:
        mes_evaluacion = df_mensual['mes'].max()
        
    df_mensual = df_mensual[df_mensual['mes'] <= mes_evaluacion]
    
    if len(df_mensual) == 0:
        return metricas, pd.DataFrame()
        
    df_mes_eval = df_mensual[df_mensual['mes'] == mes_evaluacion]
    if len(df_mes_eval) == 0:
        return metricas, df_mensual
        
    # LIMITES MES DE EVALUACIÓN
    row = df_mes_eval.iloc[0]
    total_ngr = row['calculo_ngr']
    total_depositos = row['total_depositos']
    total_num_depositos = row['num_depositos']
    total_ggr_deportes = row['apuestas_deportivas_ggr']
    total_ggr_casino = row['casino_ggr']
    total_ggr = total_ggr_deportes + total_ggr_casino
    
    # TICKETS / APUESTAS DEL MES DE EVALUACIÓN
    margen_estimado = 0.05
    if 'tickets_deportes' in row and not pd.isna(row['tickets_deportes']):
        total_apuestas_deportes = row['tickets_deportes']
    else:
        total_apuestas_deportes = total_ggr_deportes / margen_estimado if total_ggr_deportes > 0 else 0.0

    if 'tickets_casino' in row and not pd.isna(row['tickets_casino']):
        total_apuestas_casino = row['tickets_casino']
    else:
        total_apuestas_casino = total_ggr_casino / margen_estimado if total_ggr_casino > 0 else 0.0
        
    total_apuestas = total_apuestas_deportes + total_apuestas_casino
    
    # 1. RENTABILIDAD DE COMISIÓN (12%)
    if total_depositos > 0:
        rentabilidad_pct = (total_ngr / total_depositos) * 100
        
        if rentabilidad_pct >= 8.0:
            score_pct = 7.0
        elif rentabilidad_pct >= 6.0:
            score_pct = 5.5
        elif rentabilidad_pct >= 4.0:
            score_pct = 4.0
        else:
            score_pct = max(0.0, min(7.0, rentabilidad_pct * 1.75))
            
        score_volumen = min(3.0, np.log10(total_ngr + 1) * 0.75) if total_ngr > 0 else 0.0
        metricas['rentabilidad'] = score_pct + score_volumen
    else:
        metricas['rentabilidad'] = 0.0
        
    # 2. VOLUMEN DE NEGOCIO (15%)
    total_transacciones = total_num_depositos + row['num_retiros']
    if total_transacciones > 0:
        volumen = np.log10(total_transacciones + 1) * 2.3
        metricas['volumen'] = min(10.0, max(0.0, volumen))
    else:
        metricas['volumen'] = 0.0
        
    # 3. FIDELIDAD DE JUGADORES (15%)
    df_agente_mes = df_agente[df_agente['mes'] == mes_evaluacion]
    jugadores_agente = df_agente_mes['jugador_id'].nunique()
    if total_jugadores_global > 0:
        proporcion = (jugadores_agente / total_jugadores_global) * 100
        metricas['fidelidad'] = min(10.0, proporcion * 2.5)
    else:
        metricas['fidelidad'] = 0.0
        
    # 4. ESTABILIDAD FINANCIERA (12%)
    comisiones_mensuales = df_mensual['calculo_ngr'].values
    total_meses = len(comisiones_mensuales)
    
    if total_meses == 0:
        metricas['estabilidad'] = 0.0
    elif total_meses == 1:
        metricas['estabilidad'] = 3.0
    else:
        min_comision = np.min(comisiones_mensuales)
        comisiones_log = np.log(comisiones_mensuales + abs(min_comision) + 1)
        cv_log = calcular_coeficiente_variacion(comisiones_log)
        ef = 1 - cv_log
        
        if ef >= 0.8:
            score_cv = 8.0 + ((ef - 0.8) / 0.2) * 2.0
        elif ef >= 0.6:
            score_cv = 6.0 + ((ef - 0.6) / 0.2) * 2.0
        elif ef >= 0.4:
            score_cv = 4.0 + ((ef - 0.4) / 0.2) * 2.0
        elif ef >= 0:
            score_cv = (ef / 0.4) * 4.0
        else:
            score_cv = 0.0
            
        n_meses_evaluar = min(3, total_meses)
        ultimos_meses = comisiones_mensuales[-n_meses_evaluar:]
        
        mes_1 = ultimos_meses[-1] > 0
        mes_2 = ultimos_meses[-2] > 0 if n_meses_evaluar >= 2 else True
        mes_3 = ultimos_meses[-3] > 0 if n_meses_evaluar >= 3 else True
        
        if n_meses_evaluar >= 3:
            if mes_1 and mes_2 and mes_3: score_reciente = 10.0
            elif mes_1 and mes_2 and not mes_3: score_reciente = 8.0
            elif mes_1 and not mes_2 and mes_3: score_reciente = 7.0
            elif not mes_1 and mes_2 and mes_3: score_reciente = 6.0
            elif mes_1 and not mes_2 and not mes_3: score_reciente = 5.0
            elif not mes_1 and not mes_2 and mes_3: score_reciente = 3.0
            else: score_reciente = 0.0
        else:
            if mes_1 and mes_2: score_reciente = 8.0
            elif mes_1 and not mes_2: score_reciente = 6.0
            elif not mes_1 and mes_2: score_reciente = 4.0
            else: score_reciente = 0.0
                
        metricas['estabilidad'] = score_cv * 0.4 + score_reciente * 0.6
        
    # 5. CRECIMIENTO DE DEPÓSITOS (10%)
    if total_meses >= 2:
        depositos_actual = df_mensual.iloc[-1]['num_depositos']
        depositos_anterior = df_mensual.iloc[-2]['num_depositos']
        
        if depositos_anterior > 0:
            crecimiento_pct = ((depositos_actual - depositos_anterior) / depositos_anterior) * 100
            
            if crecimiento_pct >= 20: metricas['crecimiento'] = 10.0
            elif crecimiento_pct >= 10: metricas['crecimiento'] = 8.0
            elif crecimiento_pct >= 5: metricas['crecimiento'] = 6.5
            elif crecimiento_pct >= 0: metricas['crecimiento'] = 5.0
            elif crecimiento_pct >= -10: metricas['crecimiento'] = 3.5
            elif crecimiento_pct >= -20: metricas['crecimiento'] = 2.0
            else: metricas['crecimiento'] = 1.0
        elif depositos_actual > 0:
            metricas['crecimiento'] = 10.0
        else:
            metricas['crecimiento'] = 0.0
    else:
        metricas['crecimiento'] = 5.0
        
    # 6. EFICIENCIA CASINO (8%)
    if total_ggr_casino > 0:
        efic_casino = (total_num_depositos / total_ggr_casino) * 100
        
        if efic_casino > 33: metricas['eficiencia_casino'] = 2.0
        elif efic_casino > 20: metricas['eficiencia_casino'] = 3.0
        elif efic_casino > 14: metricas['eficiencia_casino'] = 5.0
        elif efic_casino > 10: metricas['eficiencia_casino'] = 7.5
        elif efic_casino > 5.5: metricas['eficiencia_casino'] = 10.0
        else: metricas['eficiencia_casino'] = max(0.0, 10.0 - (efic_casino / 10.0))
    else:
        metricas['eficiencia_casino'] = 0.0
        
    # 7. EFICIENCIA DEPORTES (8%)
    if total_ggr_deportes > 0:
        efic_deportes = (total_num_depositos / total_ggr_deportes) * 100
        
        if efic_deportes > 33: metricas['eficiencia_deportes'] = 2.0
        elif efic_deportes > 20: metricas['eficiencia_deportes'] = 3.0
        elif efic_deportes > 14: metricas['eficiencia_deportes'] = 5.0
        elif efic_deportes > 10: metricas['eficiencia_deportes'] = 7.5
        elif efic_deportes > 5.5: metricas['eficiencia_deportes'] = 10.0
        else: metricas['eficiencia_deportes'] = max(0.0, 10.0 - (efic_deportes / 10.0))
    else:
        metricas['eficiencia_deportes'] = 0.0
        
    # 8. EFICIENCIA DE CONVERSIÓN (11%)
    if total_depositos > 0:
        conversion_pct = (total_ggr / total_depositos) * 100
        
        if conversion_pct >= 15: metricas['eficiencia_conversion'] = 10.0
        elif conversion_pct >= 10: metricas['eficiencia_conversion'] = 7.5
        elif conversion_pct >= 7: metricas['eficiencia_conversion'] = 5.0
        elif conversion_pct >= 5: metricas['eficiencia_conversion'] = 3.0
        else: metricas['eficiencia_conversion'] = max(0.0, conversion_pct * 2.0)
    else:
        metricas['eficiencia_conversion'] = 0.0
        
    # 9. TENDENCIA TÉCNICA (4%)
    if total_meses >= 3:
        tendencia = calcular_tendencia_lineal(comisiones_mensuales)
        if tendencia > 1000: metricas['tendencia'] = 8.0
        elif tendencia > 0: metricas['tendencia'] = 6.0
        elif tendencia > -1000: metricas['tendencia'] = 4.0
        else: metricas['tendencia'] = 2.0
    else:
        metricas['tendencia'] = 5.0
        
    # 10. DIVERSIFICACIÓN DE PRODUCTOS (3%)
    if total_apuestas > 0:
        p_casino = total_apuestas_casino / total_apuestas
        p_deportes = total_apuestas_deportes / total_apuestas
        hhi = p_casino**2 + p_deportes**2
        diversificacion = (1 - hhi) * 10.0
        metricas['diversificacion'] = diversificacion
    else:
        metricas['diversificacion'] = 0.0
        
    # 11. CALIDAD DE JUGADORES (2%)
    if jugadores_agente > 0:
        apuesta_promedio = total_apuestas / jugadores_agente
        if apuesta_promedio > 10000: metricas['calidad_jugadores'] = 8.0
        elif apuesta_promedio > 5000: metricas['calidad_jugadores'] = 6.0
        elif apuesta_promedio > 1000: metricas['calidad_jugadores'] = 4.0
        else: metricas['calidad_jugadores'] = 2.0
    else:
        metricas['calidad_jugadores'] = 0.0
        
    return metricas, df_mensual

# ============================================================================
# SCORE Y CATEGORIZACIÓN
# ============================================================================

def calcular_score_total(metricas: dict) -> float:
    score = 0.0
    for metrica, valor in metricas.items():
        if metrica in PESOS_METRICAS:
            score += valor * PESOS_METRICAS[metrica]
    return score

def categorizar_agente(score: float) -> tuple:
    if score >= 9.0:
        return "A+++", "Excelencia excepcional - Líderes absolutos"
    elif score >= 8.5:
        return "A++", "Excelencia alta - Top tier sobresaliente"
    elif score >= 8.0:
        return "A+", "Excelencia - Muy alto desempeño"
    elif score >= 7.5:
        return "B+++", "Consolidado superior - Buen track record"
    elif score >= 7.0:
        return "B++", "Consolidado alto - Desempeño sólido"
    elif score >= 6.5:
        return "B+", "Consolidado - Estable y confiable"
    elif score >= 5.5:
        return "C+++", "En desarrollo avanzado - Progreso visible"
    elif score >= 4.5:
        return "C++", "En desarrollo medio - Requiere mejoras"
    elif score >= 3.5:
        return "C+", "Principiante - Necesita atención"
    else:
        return "C", "Base - Punto de partida"

# ============================================================================
# PREDICCIÓN DE GGR - MÉTODOS AVANZADOS
# ============================================================================

def suavizado_exponencial_simple(serie, alpha=0.3):
    if len(serie) == 0: return 0.0
    s = serie[0]
    for x in serie[1:]:
        s = alpha * x + (1 - alpha) * s
    return s

def suavizado_exponencial_doble(serie, alpha=0.3, beta=0.1):
    if len(serie) < 2: return serie[-1] if len(serie)>0 else 0.0
    nivel = serie[0]
    tendencia = serie[1] - serie[0]
    for i in range(1, len(serie)):
        nivel_anterior = nivel
        nivel = alpha * serie[i] + (1 - alpha) * (nivel + tendencia)
        tendencia = beta * (nivel - nivel_anterior) + (1 - beta) * tendencia
    return nivel + tendencia

def suavizado_exponencial_triple(serie, periodo=12, alpha=0.3, beta=0.1, gamma=0.1):
    n = len(serie)
    if n < periodo * 2:
        return suavizado_exponencial_doble(serie, alpha, beta)
    
    nivel = np.mean(serie[:periodo])
    tendencia = (np.mean(serie[periodo:2*periodo]) - np.mean(serie[:periodo])) / periodo
    estacionalidad = np.zeros(periodo)
    for i in range(periodo):
        estacionalidad[i] = serie[i] / nivel if nivel > 0 else 1.0
        
    for i in range(periodo, n):
        nivel_anterior = nivel
        idx_est = i % periodo
        est = estacionalidad[idx_est] if estacionalidad[idx_est] > 0 else 1.0
        nivel = alpha * (serie[i] / est) + (1 - alpha) * (nivel + tendencia)
        tendencia = beta * (nivel - nivel_anterior) + (1 - beta) * tendencia
        estacionalidad[idx_est] = gamma * (serie[i] / nivel if nivel > 0 else 1.0) + (1 - gamma) * estacionalidad[idx_est]
        
    idx_est_pred = n % periodo
    prediccion = (nivel + tendencia) * estacionalidad[idx_est_pred]
    return prediccion

def prediccion_regresion_lineal(serie):
    n = len(serie)
    if n < 3: return serie[-1] if n>0 else 0.0
    x = np.arange(1, n + 1)
    y = np.array(serie)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerador = np.sum((x - x_mean) * (y - y_mean))
    denominador = np.sum((x - x_mean)**2)
    if denominador == 0: return y[-1]
    b1 = numerador / denominador
    b0 = y_mean - b1 * x_mean
    return max(0.0, b0 + b1 * (n + 1))

def promedio_movil_ponderado(serie):
    n = len(serie)
    if n >= 3:
        u3 = serie[-3:]
        return u3[0]*0.2 + u3[1]*0.3 + u3[2]*0.5
    elif n >= 2:
        return serie[-2]*0.4 + serie[-1]*0.6
    elif n == 1:
        return serie[-1]
    return 0.0

def validar_modelo(serie, metodo_prediccion):
    n = len(serie)
    if n < 5: return float('inf')
    n_validacion = min(3, n - 2)
    errores = []
    
    for i in range(1, n_validacion + 1):
        serie_entrenamiento = serie[:-i]
        valor_real = serie[-i]
        try:
            pred = metodo_prediccion(serie_entrenamiento)
            errores.append((valor_real - pred)**2)
        except:
            return float('inf')
            
    return np.sqrt(np.mean(errores))

def predecir_ggr_proximo_mes(df_mensual: pd.DataFrame, metodo: str = "auto") -> float:
    if len(df_mensual) == 0: return 0.0
    
    ggr_mensuales = []
    for _, row in df_mensual.iterrows():
        ggr = row.get('apuestas_deportivas_ggr', 0.0) + row.get('casino_ggr', 0.0)
        ggr_mensuales.append(ggr)
        
    # Filtrar válidos
    ggr_mensuales = [x for x in ggr_mensuales if 0 <= x < 1e9]
    if len(ggr_mensuales) == 0: return 0.0
    
    prediccion = 0.0
    n = len(ggr_mensuales)
    
    if metodo == "auto":
        if n >= 12:
            try:
                prediccion = suavizado_exponencial_triple(ggr_mensuales, 12)
                max_h = max(ggr_mensuales)
                if prediccion < 0 or prediccion > max_h * 2:
                    prediccion = suavizado_exponencial_doble(ggr_mensuales)
            except:
                prediccion = suavizado_exponencial_doble(ggr_mensuales)
        elif n >= 5:
            metodos = {
                "holt": suavizado_exponencial_doble,
                "regresion": prediccion_regresion_lineal,
                "promedio": promedio_movil_ponderado
            }
            mejor_metodo = "promedio"
            mejor_error = float('inf')
            for name, func in metodos.items():
                try:
                    err = validar_modelo(ggr_mensuales, func)
                    if err < mejor_error:
                        mejor_error = err
                        mejor_metodo = name
                except:
                    continue
            
            try:
                prediccion = metodos[mejor_metodo](ggr_mensuales)
            except:
                prediccion = promedio_movil_ponderado(ggr_mensuales)
        elif n >= 3:
            try:
                p_holt = suavizado_exponencial_doble(ggr_mensuales)
                p_prom = promedio_movil_ponderado(ggr_mensuales)
                prediccion = (p_holt + p_prom) / 2.0
            except:
                prediccion = promedio_movil_ponderado(ggr_mensuales)
        else:
            prediccion = promedio_movil_ponderado(ggr_mensuales)
    elif metodo == "holt_winters":
        prediccion = suavizado_exponencial_triple(ggr_mensuales, 12)
    elif metodo == "holt":
        prediccion = suavizado_exponencial_doble(ggr_mensuales)
    elif metodo == "regresion":
        prediccion = prediccion_regresion_lineal(ggr_mensuales)
    elif metodo == "promedio":
        prediccion = promedio_movil_ponderado(ggr_mensuales)
    else:
        prediccion = promedio_movil_ponderado(ggr_mensuales)
        
    prediccion = max(0.0, prediccion)
    max_historico = max(ggr_mensuales)
    if prediccion > max_historico * 3:
        prediccion = max_historico * 1.2
        
    return round(float(prediccion), 2)

# Alias para compatibilidad
def predecir_ggr(df_mensual):
    return predecir_ggr_proximo_mes(df_mensual)

# ============================================================================
# PREDICCIÓN CREDITICIA
# ============================================================================

def calcular_credito_sugerido(df_mensual: pd.DataFrame, score: float, metricas: dict) -> tuple:
    detalles_default = {
        "p25": 0.0,
        "cv": 0.0,
        "f_volatilidad": 0.0,
        "desc_volatilidad": "Sin datos",
        "tendencia": 0.0,
        "f_tendencia": 0.0,
        "desc_tendencia": "Sin datos",
        "f_score": 0.0,
        "meses_historial": 0,
        "mediana": 0.0
    }
    
    if len(df_mensual) == 0:
        return 0.0, detalles_default
        
    ngr_mensuales = df_mensual['calculo_ngr'].values
    ngr_validos = ngr_mensuales[ngr_mensuales > 0]
    
    if len(ngr_validos) == 0:
        return 0.0, detalles_default
        
    p25 = np.percentile(ngr_validos, 25)
    mediana = np.median(ngr_validos)
    
    base_credito = p25 * 0.6 + mediana * 0.4
    
    min_ngr = np.min(ngr_validos)
    ngr_log = np.log(ngr_validos + abs(min_ngr) + 1)
    
    cv_log = calcular_coeficiente_variacion(ngr_log)
    f_v, desc_volatilidad = calcular_factor_volatilidad(cv_log)
    
    tendencia = calcular_tendencia_lineal(ngr_validos)
    f_t, desc_tendencia = calcular_factor_tendencia(tendencia)
    
    S = score
    E = metricas.get("estabilidad", 0.0)
    f_s_final = 0.5 + 0.06 * ((S + E) / 2)
    
    comision_total = np.sum(ngr_validos)
    if comision_total >= 80000: f_volumen = 2.0
    elif comision_total >= 50000: f_volumen = 1.7
    elif comision_total >= 30000: f_volumen = 1.4
    elif comision_total >= 15000: f_volumen = 1.2
    elif comision_total >= 5000: f_volumen = 1.0
    else: f_volumen = 0.85
    
    credito = base_credito * f_s_final * f_v * f_t * f_volumen
    
    if p25 < 50:
        credito = 0.0
    else:
        limite_superior = 4 * mediana * f_volumen
        credito = min(credito, limite_superior)
        
    if len(ngr_validos) < 3:
        credito = credito * 0.5
        
    detalles = {
        "p25": p25,
        "cv": cv_log,
        "f_volatilidad": f_v,
        "desc_volatilidad": desc_volatilidad,
        "tendencia": tendencia,
        "f_tendencia": f_t,
        "desc_tendencia": desc_tendencia,
        "f_score": f_s_final,
        "meses_historial": len(ngr_validos),
        "mediana": mediana
    }
    
    return round(float(credito), 2), detalles

# ============================================================================
# COMPATIBILIDAD CON CÓDIGO EXISTENTE
# ============================================================================

def calcular_metricas_mensuales(df_agente: pd.DataFrame, total_jugadores_global: int = 1, mode: str = "snapshot") -> pd.DataFrame:
    """
    Construye un DataFrame mensual con las 11 métricas (0-10) y score_global.
    Esta función itera sobre cada mes disponible y llama a la lógica matriz 
    (calcular_metricas_agente), la cual internamente usa mes_evaluacion para 
    separar cálculos del mes y cálculos históricos.
    (El parámetro 'mode' se ignora intencionalmente porque el filtrado correcto 
    ya lo hace de forma nativa e intrínseca calcular_metricas_agente).
    """
    if df_agente is None or len(df_agente) == 0 or 'creado' not in df_agente.columns:
        return pd.DataFrame(columns=["mes", *PESOS_METRICAS.keys(), "score_global"])

    df = df_agente.copy()
    df['creado'] = pd.to_datetime(df['creado'], errors='coerce')
    df = df.dropna(subset=['creado'])
    if len(df) == 0:
        return pd.DataFrame(columns=["mes", *PESOS_METRICAS.keys(), "score_global"])

    df['mes'] = df['creado'].dt.to_period('M')
    meses_disponibles = sorted(df['mes'].unique())

    filas = []
    for mes in meses_disponibles:
        metricas_mes, _ = calcular_metricas_agente(df, total_jugadores_global, mes_evaluacion=mes)
        score_mes = calcular_score_total(metricas_mes)

        fila = {"mes": mes}
        for k in PESOS_METRICAS.keys():
            fila[k] = metricas_mes.get(k, 0.0)
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
    Wrapper para compatibilidad. Retorna métricas globales (último mes presente en df),
    el df agrupado original, y el df_mensual tabulado.
    """
    metricas_globales, df_mensual_original = calcular_metricas_agente(df_agente, total_jugadores_global)

    df_metricas_mensuales = calcular_metricas_mensuales(
        df_agente=df_agente,
        total_jugadores_global=total_jugadores_global,
        mode=monthly_mode
    )

    return metricas_globales, df_mensual_original, df_metricas_mensuales

def calcular_metricas_agente_con_mensual(df_agente, total_jugadores_global=1, monthly_mode="snapshot", debug_validate=False):
    return calcular_metricas_agente_refactor(df_agente, total_jugadores_global, monthly_mode, debug_validate)
