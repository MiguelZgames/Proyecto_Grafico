"""
SISTEMA DE ANÁLISIS DE AGENTES - VERSIÓN JULIA
===============================================

Sistema para calcular métricas de agentes, predicción crediticia y predicción de GGR.
Convertido desde Python a Julia.

Variables del dataset:
- nombre_usuario_agente: Nombre del agente
- nombre_usuario_jugador: Nombre del jugador
- jugador_id: ID del jugador
- creado: Fecha del movimiento
- num_depositos: Número de depósitos
- num_retiros: Número de retiros
- total_depositos: Total de depósitos
- total_retiros: Total de retiros
- apuestas_deportivas_ggr: GGR de apuestas deportivas
- casino_ggr: GGR de casino
- calculo_ngr: Cálculo NGR (comisión)

Autor: Conversión a Julia
Fecha: 2026-02-18
"""

using DataFrames
using CSV
using Dates
using Statistics
using LinearAlgebra

# ============================================================================
# CONSTANTES - PESOS DE LAS MÉTRICAS
# ============================================================================

const PESOS_METRICAS = Dict(
    "rentabilidad" => 0.12,
    "volumen" => 0.15,
    "fidelidad" => 0.15,
    "estabilidad" => 0.12,
    "crecimiento" => 0.10,
    "eficiencia_casino" => 0.08,
    "eficiencia_deportes" => 0.08,
    "eficiencia_conversion" => 0.11,
    "tendencia" => 0.04,
    "diversificacion" => 0.03,
    "calidad_jugadores" => 0.02
)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

"""
Calcula el percentil 25 del historial
"""
function calcular_percentil_25(historial::Vector{Float64})
    if length(historial) == 0
        return 0.0
    end
    return quantile(historial, 0.25)
end

"""
Calcula el coeficiente de variación (CV)
"""
function calcular_coeficiente_variacion(historial::Vector{Float64})
    if length(historial) < 2
        return 0.0
    end

    media = mean(historial)
    if media == 0
        return 0.0
    end

    desv_std = std(historial, corrected=true)
    return desv_std / abs(media)
end

"""
Calcula el factor de volatilidad basado en el CV logarítmico
"""
function calcular_factor_volatilidad(cv_log::Float64)
    if cv_log < 0.2
        return 1.00, "Baja volatilidad"
    elseif cv_log < 0.4
        return 0.85, "Moderada"
    elseif cv_log < 0.6
        return 0.70, "Alta"
    elseif cv_log < 0.8
        return 0.55, "Muy alta"
    else
        return 0.40, "Extrema"
    end
end

"""
Calcula la tendencia usando regresión lineal
"""
function calcular_tendencia_lineal(historial::Vector{Float64})
    if length(historial) < 2
        return 0.0
    end

    n = length(historial)
    t = collect(1:n)

    numerador = n * sum(t .* historial) - sum(t) * sum(historial)
    denominador = n * sum(t .^ 2) - sum(t)^2

    if denominador == 0
        return 0.0
    end

    return numerador / denominador
end

"""
Calcula el factor de tendencia
"""
function calcular_factor_tendencia(tendencia::Float64)
    if tendencia > 5000
        return 1.15, "Crecimiento fuerte"
    elseif tendencia > 0
        return 1.05, "Crecimiento moderado"
    elseif tendencia >= -5000
        return 0.95, "Estancamiento"
    else
        return 0.80, "Decrecimiento"
    end
end

# ============================================================================
# CARGAR Y PROCESAR DATOS
# ============================================================================

"""
Cargar datos del archivo CSV con mapeo de columnas
"""
function cargar_datos(archivo::String)
    println("🔄 Cargando datos del archivo: $archivo")

    # Cargar CSV
    df = CSV.read(archivo, DataFrame)
    println("✅ Datos cargados: $(nrow(df)) registros")

    # Mapear columnas del CSV real a nombres esperados
    rename!(df,
        "agente_username" => "nombre_usuario_agente",
        "player_username" => "nombre_usuario_jugador",
        "player_id" => "jugador_id",
        "date_evento" => "creado",
        "n_deposito" => "num_depositos",
        "n_retiro" => "num_retiros",
        "deposito" => "total_depositos",
        "retiro" => "total_retiros",
        "ggr_deportiva" => "apuestas_deportivas_ggr",
        "ggr_casino" => "casino_ggr",
        "comis_calculada" => "calculo_ngr",
        "deportiva_tickets" => "tickets_deportes",
        "casino_tickets" => "tickets_casino"
    )

    println("✅ Columnas mapeadas correctamente")

    # Convertir fechas (manejar si ya son Date o si son String)
    if eltype(df.creado) != Date
        df.creado = Date.(string.(df.creado), dateformat"yyyy-mm-dd")
    end

    # Convertir columnas numéricas
    columnas_numericas = [
        :num_depositos, :num_retiros, :total_depositos, :total_retiros,
        :apuestas_deportivas_ggr, :casino_ggr, :calculo_ngr,
        :tickets_deportes, :tickets_casino
    ]

    for col in columnas_numericas
        if col in names(df)
            df[!, col] = coalesce.(Float64.(df[!, col]), 0.0)
        end
    end

    println("✅ Datos procesados: $(nrow(df)) registros")

    return df
end

# ============================================================================
# CÁLCULO DE LAS 11 MÉTRICAS
# ============================================================================

"""
Calcula las 11 métricas para un agente en el mes de evaluación específico.
Métricas puntuales usan solo el mes de evaluación.
Métricas históricas (Estabilidad, Tendencia, Crecimiento) usan todo el historial hasta ese mes.
"""
function calcular_metricas_agente(df_agente::DataFrame, total_jugadores_global::Int, mes_evaluacion)
    metricas = Dict{String,Float64}()

    # Validar que exista la columna 'creado'
    if !("creado" in names(df_agente)) || nrow(df_agente) == 0
        return Dict(k => 0.0 for k in keys(PESOS_METRICAS)), DataFrame()
    end

    # Filtrar fechas válidas
    df_agente = df_agente[.!ismissing.(df_agente.creado), :]

    if nrow(df_agente) == 0
        return Dict(k => 0.0 for k in keys(PESOS_METRICAS)), DataFrame()
    end

    # Agregar columna de mes
    df_agente[!, :mes] = yearmonth.(df_agente.creado)

    # Agrupar por mes
    df_mensual = combine(groupby(df_agente, :mes),
        :calculo_ngr => sum => :calculo_ngr,
        :num_depositos => sum => :num_depositos,
        :num_retiros => sum => :num_retiros,
        :total_depositos => sum => :total_depositos,
        :total_retiros => sum => :total_retiros,
        :apuestas_deportivas_ggr => sum => :apuestas_deportivas_ggr,
        :casino_ggr => sum => :casino_ggr,
        :tickets_deportes => sum => :tickets_deportes,
        :tickets_casino => sum => :tickets_casino,
        :jugador_id => (x -> length(unique(x))) => :jugador_id_unique
    )

    # Ordenar por mes y filtrar hasta el mes de evaluación (historial)
    sort!(df_mensual, :mes)
    df_mensual = filter(row -> row.mes <= mes_evaluacion, df_mensual)

    if nrow(df_mensual) == 0
        return Dict(k => 0.0 for k in keys(PESOS_METRICAS)), DataFrame()
    end

    # Verificar que el agente tenga datos en el mes de evaluación
    df_mes_eval = filter(row -> row.mes == mes_evaluacion, df_mensual)
    if nrow(df_mes_eval) == 0
        return Dict(k => 0.0 for k in keys(PESOS_METRICAS)), df_mensual
    end

    # --- TOTALES DEL MES DE EVALUACIÓN (métricas mensuales, no acumuladas) ---
    total_ngr = df_mes_eval[1, :calculo_ngr]
    total_depositos = df_mes_eval[1, :total_depositos]
    total_num_depositos = df_mes_eval[1, :num_depositos]
    total_ggr_deportes = df_mes_eval[1, :apuestas_deportivas_ggr]
    total_ggr_casino = df_mes_eval[1, :casino_ggr]
    total_ggr = total_ggr_deportes + total_ggr_casino

    # Tickets reales del mes (apuestas)
    total_apuestas_deportes = df_mes_eval[1, :tickets_deportes]
    total_apuestas_casino = df_mes_eval[1, :tickets_casino]
    total_apuestas = total_apuestas_deportes + total_apuestas_casino

    # 1. RENTABILIDAD DE COMISIÓN (12%)
    if total_depositos > 0
        rentabilidad_pct = (total_ngr / total_depositos) * 100

        # Score base por porcentaje (0-7 puntos)
        if rentabilidad_pct >= 8.0
            score_pct = 7.0
        elseif rentabilidad_pct >= 6.0
            score_pct = 5.5
        elseif rentabilidad_pct >= 4.0
            score_pct = 4.0
        else
            score_pct = max(0, min(7, rentabilidad_pct * 1.75))
        end

        # Bonus por volumen absoluto (0-3 puntos)
        score_volumen = total_ngr > 0 ? min(3.0, log10(total_ngr + 1) * 0.75) : 0.0

        metricas["rentabilidad"] = score_pct + score_volumen
    else
        metricas["rentabilidad"] = 0.0
    end

    # 2. VOLUMEN DE NEGOCIO (15%) - Del mes de evaluación
    total_transacciones = total_num_depositos + df_mes_eval[1, :num_retiros]
    if total_transacciones > 0
        volumen = log10(total_transacciones + 1) * 2.3
        metricas["volumen"] = min(10, max(0, volumen))
    else
        metricas["volumen"] = 0.0
    end

    # 3. FIDELIDAD DE JUGADORES (15%) - Del mes de evaluación
    df_agente_mes = filter(row -> row.mes == mes_evaluacion, df_agente)
    jugadores_agente = length(unique(df_agente_mes.jugador_id))
    if total_jugadores_global > 0
        proporcion_jugadores = (jugadores_agente / total_jugadores_global) * 100
        metricas["fidelidad"] = min(10, proporcion_jugadores * 2.5)
    else
        metricas["fidelidad"] = 0.0
    end

    # 4. ESTABILIDAD FINANCIERA (12%)
    # Combina: CV logarítmico de todo el historial (40%) + patrón últimos 3 meses (60%)
    comisiones_mensuales = df_mensual.calculo_ngr
    total_meses = length(comisiones_mensuales)
    
    if total_meses == 0
        metricas["estabilidad"] = 0.0
    elseif total_meses == 1
        # Agente nuevo con solo 1 mes = 3.0 puntos por defecto
        metricas["estabilidad"] = 3.0
    else
        # --- COMPONENTE 1: CV logarítmico de todo el historial (40%) ---
        min_comision = minimum(comisiones_mensuales)
        comisiones_log = log.(comisiones_mensuales .+ abs(min_comision) .+ 1)
        cv_log = calcular_coeficiente_variacion(comisiones_log)
        ef = 1 - cv_log
        
        if ef >= 0.8
            score_cv = 8.0 + ((ef - 0.8) / 0.2) * 2.0
        elseif ef >= 0.6
            score_cv = 6.0 + ((ef - 0.6) / 0.2) * 2.0
        elseif ef >= 0.4
            score_cv = 4.0 + ((ef - 0.4) / 0.2) * 2.0
        elseif ef >= 0
            score_cv = (ef / 0.4) * 4.0
        else
            score_cv = 0.0
        end
        
        # --- COMPONENTE 2: Patrón últimos 3 meses (60%) ---
        n_meses_evaluar = min(3, total_meses)
        ultimos_meses = comisiones_mensuales[end-n_meses_evaluar+1:end]
        
        mes_1 = ultimos_meses[end] > 0      # Último mes (más reciente)
        mes_2 = n_meses_evaluar >= 2 ? ultimos_meses[end-1] > 0 : true
        mes_3 = n_meses_evaluar >= 3 ? ultimos_meses[end-2] > 0 : true
        
        if n_meses_evaluar >= 3
            if mes_1 && mes_2 && mes_3
                score_reciente = 10.0  # 3 meses seguidos con comisión
            elseif mes_1 && mes_2 && !mes_3
                score_reciente = 8.0   # Falló hace 2 meses, pero lleva 2 bien
            elseif mes_1 && !mes_2 && mes_3
                score_reciente = 7.0   # Falló el mes anterior, se recuperó
            elseif !mes_1 && mes_2 && mes_3
                score_reciente = 6.0   # Solo falló el último mes
            elseif mes_1 && !mes_2 && !mes_3
                score_reciente = 5.0   # Falló 2 meses, se recuperó en el último
            elseif !mes_1 && !mes_2 && mes_3
                score_reciente = 3.0   # Falló los últimos 2 meses
            else
                score_reciente = 0.0   # Sin comisiones en los 3 meses
            end
        else  # 2 meses disponibles
            if mes_1 && mes_2
                score_reciente = 8.0
            elseif mes_1 && !mes_2
                score_reciente = 6.0
            elseif !mes_1 && mes_2
                score_reciente = 4.0
            else
                score_reciente = 0.0
            end
        end
        
        # --- COMBINAR: 40% historial + 60% reciente ---
        metricas["estabilidad"] = score_cv * 0.4 + score_reciente * 0.6
    end

    # 5. CRECIMIENTO DE DEPÓSITOS (10%)
    if nrow(df_mensual) >= 2
        depositos_actual = df_mensual[end, :num_depositos]
        depositos_anterior = df_mensual[end-1, :num_depositos]

        if depositos_anterior > 0
            crecimiento_pct = ((depositos_actual - depositos_anterior) / depositos_anterior) * 100

            if crecimiento_pct >= 20
                metricas["crecimiento"] = 10.0
            elseif crecimiento_pct >= 10
                metricas["crecimiento"] = 8.0
            elseif crecimiento_pct >= 5
                metricas["crecimiento"] = 6.5
            elseif crecimiento_pct >= 0
                metricas["crecimiento"] = 5.0
            elseif crecimiento_pct >= -10
                metricas["crecimiento"] = 3.5
            elseif crecimiento_pct >= -20
                metricas["crecimiento"] = 2.0
            else
                metricas["crecimiento"] = 1.0
            end
        elseif depositos_actual > 0
            metricas["crecimiento"] = 10.0
        else
            metricas["crecimiento"] = 0.0
        end
    else
        metricas["crecimiento"] = 5.0
    end

    # 6. EFICIENCIA CASINO (8%)
    if total_ggr_casino > 0
        efic_casino = (total_num_depositos / total_ggr_casino) * 100

        if efic_casino > 33
            metricas["eficiencia_casino"] = 2.0
        elseif efic_casino > 20
            metricas["eficiencia_casino"] = 3.0
        elseif efic_casino > 14
            metricas["eficiencia_casino"] = 5.0
        elseif efic_casino > 10
            metricas["eficiencia_casino"] = 7.5
        elseif efic_casino > 5.5
            metricas["eficiencia_casino"] = 10.0
        else
            metricas["eficiencia_casino"] = max(0, 10 - (efic_casino / 10))
        end
    else
        metricas["eficiencia_casino"] = 0.0
    end

    # 7. EFICIENCIA DEPORTES (8%)
    if total_ggr_deportes > 0
        efic_deportes = (total_num_depositos / total_ggr_deportes) * 100

        if efic_deportes > 33
            metricas["eficiencia_deportes"] = 2.0
        elseif efic_deportes > 20
            metricas["eficiencia_deportes"] = 3.0
        elseif efic_deportes > 14
            metricas["eficiencia_deportes"] = 5.0
        elseif efic_deportes > 10
            metricas["eficiencia_deportes"] = 7.5
        elseif efic_deportes > 5.5
            metricas["eficiencia_deportes"] = 10.0
        else
            metricas["eficiencia_deportes"] = max(0, 10 - (efic_deportes / 10))
        end
    else
        metricas["eficiencia_deportes"] = 0.0
    end

    # 8. EFICIENCIA DE CONVERSIÓN (11%)
    if total_depositos > 0
        conversion_pct = (total_ggr / total_depositos) * 100

        if conversion_pct >= 15
            metricas["eficiencia_conversion"] = 10.0
        elseif conversion_pct >= 10
            metricas["eficiencia_conversion"] = 7.5
        elseif conversion_pct >= 7
            metricas["eficiencia_conversion"] = 5.0
        elseif conversion_pct >= 5
            metricas["eficiencia_conversion"] = 3.0
        else
            metricas["eficiencia_conversion"] = max(0, conversion_pct * 2)
        end
    else
        metricas["eficiencia_conversion"] = 0.0
    end

    # 9. TENDENCIA TÉCNICA (4%)
    if length(comisiones_mensuales) >= 3
        tendencia = calcular_tendencia_lineal(comisiones_mensuales)

        if tendencia > 1000
            metricas["tendencia"] = 8.0
        elseif tendencia > 0
            metricas["tendencia"] = 6.0
        elseif tendencia > -1000
            metricas["tendencia"] = 4.0
        else
            metricas["tendencia"] = 2.0
        end
    else
        metricas["tendencia"] = 5.0
    end

    # 10. DIVERSIFICACIÓN DE PRODUCTOS (3%)
    if total_apuestas > 0
        p_casino = total_apuestas_casino / total_apuestas
        p_deportes = total_apuestas_deportes / total_apuestas
        hhi = p_casino^2 + p_deportes^2
        diversificacion = (1 - hhi) * 10
        metricas["diversificacion"] = diversificacion
    else
        metricas["diversificacion"] = 0.0
    end

    # 11. CALIDAD DE JUGADORES (2%)
    if jugadores_agente > 0
        apuesta_promedio = total_apuestas / jugadores_agente

        if apuesta_promedio > 10000
            metricas["calidad_jugadores"] = 8.0
        elseif apuesta_promedio > 5000
            metricas["calidad_jugadores"] = 6.0
        elseif apuesta_promedio > 1000
            metricas["calidad_jugadores"] = 4.0
        else
            metricas["calidad_jugadores"] = 2.0
        end
    else
        metricas["calidad_jugadores"] = 0.0
    end

    return metricas, df_mensual
end

# ============================================================================
# SCORE Y CATEGORIZACIÓN
# ============================================================================

"""
Calcula el score total ponderado
"""
function calcular_score_total(metricas::Dict{String,Float64})
    score = 0.0
    for (metrica, valor) in metricas
        if haskey(PESOS_METRICAS, metrica)
            score += valor * PESOS_METRICAS[metrica]
        end
    end
    return score
end

"""
Categoriza al agente según su score total
"""
function categorizar_agente(score::Float64)
    if score >= 9.0
        return "A+++", "Excelencia excepcional - Líderes absolutos"
    elseif score >= 8.5
        return "A++", "Excelencia alta - Top tier sobresaliente"
    elseif score >= 8.0
        return "A+", "Excelencia - Muy alto desempeño"
    elseif score >= 7.5
        return "B+++", "Consolidado superior - Buen track record"
    elseif score >= 7.0
        return "B++", "Consolidado alto - Desempeño sólido"
    elseif score >= 6.5
        return "B+", "Consolidado - Estable y confiable"
    elseif score >= 5.5
        return "C+++", "En desarrollo avanzado - Progreso visible"
    elseif score >= 4.5
        return "C++", "En desarrollo medio - Requiere mejoras"
    elseif score >= 3.5
        return "C+", "Principiante - Necesita atención"
    else
        return "C", "Base - Punto de partida"
    end
end

# ============================================================================
# PREDICCIÓN DE GGR - MÉTODOS AVANZADOS
# ============================================================================

"""
Suavizado Exponencial Simple (SES) - Para series sin tendencia
"""
function suavizado_exponencial_simple(serie::Vector{Float64}, alpha::Float64=0.3)
    if length(serie) == 0
        return 0.0
    end

    # Inicializar con el primer valor
    s = serie[1]

    # Aplicar suavizado
    for i in 2:length(serie)
        s = alpha * serie[i] + (1 - alpha) * s
    end

    return s
end

"""
Suavizado Exponencial Doble (Holt) - Para series con tendencia
"""
function suavizado_exponencial_doble(serie::Vector{Float64}, alpha::Float64=0.3, beta::Float64=0.1)
    if length(serie) < 2
        return length(serie) > 0 ? serie[end] : 0.0
    end

    # Inicializar nivel y tendencia
    nivel = serie[1]
    tendencia = serie[2] - serie[1]

    # Aplicar suavizado
    for i in 2:length(serie)
        nivel_anterior = nivel
        nivel = alpha * serie[i] + (1 - alpha) * (nivel + tendencia)
        tendencia = beta * (nivel - nivel_anterior) + (1 - beta) * tendencia
    end

    # Predicción para el siguiente período
    return nivel + tendencia
end

"""
Suavizado Exponencial Triple (Holt-Winters) - Para series con tendencia y estacionalidad
"""
function suavizado_exponencial_triple(serie::Vector{Float64}, periodo::Int=12,
    alpha::Float64=0.3, beta::Float64=0.1, gamma::Float64=0.1)
    n = length(serie)

    if n < periodo * 2
        # No hay suficientes datos para estacionalidad, usar Holt
        return suavizado_exponencial_doble(serie, alpha, beta)
    end

    # Inicializar componentes
    nivel = mean(serie[1:periodo])
    tendencia = (mean(serie[periodo+1:2*periodo]) - mean(serie[1:periodo])) / periodo

    # Estacionalidad inicial
    estacionalidad = zeros(Float64, periodo)
    for i in 1:periodo
        estacionalidad[i] = serie[i] / nivel
    end

    # Aplicar suavizado
    for i in periodo+1:n
        nivel_anterior = nivel
        idx_est = ((i - 1) % periodo) + 1

        nivel = alpha * (serie[i] / estacionalidad[idx_est]) + (1 - alpha) * (nivel + tendencia)
        tendencia = beta * (nivel - nivel_anterior) + (1 - beta) * tendencia
        estacionalidad[idx_est] = gamma * (serie[i] / nivel) + (1 - gamma) * estacionalidad[idx_est]
    end

    # Predicción para el siguiente período
    idx_est_pred = (n % periodo) + 1
    prediccion = (nivel + tendencia) * estacionalidad[idx_est_pred]

    return prediccion
end

"""
Predicción usando Regresión Lineal simple
"""
function prediccion_regresion_lineal(serie::Vector{Float64})
    n = length(serie)
    if n < 3
        return length(serie) > 0 ? serie[end] : 0.0
    end

    # Variables independientes (tiempo)
    x = collect(1:n)
    y = serie

    # Calcular coeficientes de regresión
    x_mean = mean(x)
    y_mean = mean(y)

    numerador = sum((x .- x_mean) .* (y .- y_mean))
    denominador = sum((x .- x_mean) .^ 2)

    if denominador == 0
        return y[end]
    end

    # Pendiente y ordenada al origen
    b1 = numerador / denominador
    b0 = y_mean - b1 * x_mean

    # Predicción para el siguiente período (n+1)
    prediccion = b0 + b1 * (n + 1)

    return max(0, prediccion)  # No permitir predicciones negativas
end

"""
Promedio móvil ponderado (método original como fallback)
"""
function promedio_movil_ponderado(serie::Vector{Float64})
    if length(serie) >= 3
        ultimos_3 = serie[end-2:end]
        return ultimos_3[1] * 0.2 + ultimos_3[2] * 0.3 + ultimos_3[3] * 0.5
    elseif length(serie) >= 2
        return serie[end-1] * 0.4 + serie[end] * 0.6
    elseif length(serie) == 1
        return serie[end]
    else
        return 0.0
    end
end

"""
Calcular el error cuadrático medio (RMSE) para validación
"""
function calcular_rmse(real::Vector{Float64}, prediccion::Vector{Float64})
    if length(real) != length(prediccion) || length(real) == 0
        return Inf
    end
    return sqrt(mean((real .- prediccion) .^ 2))
end

"""
Validación cruzada simple: usar los últimos 3 meses para validar
"""
function validar_modelo(serie::Vector{Float64}, metodo_prediccion::Function)
    n = length(serie)
    if n < 5  # Necesitamos al menos 5 puntos para validar
        return Inf
    end

    # Usar los últimos 3 meses para validación
    n_validacion = min(3, n - 2)
    errores = Float64[]

    for i in 1:n_validacion
        serie_entrenamiento = serie[1:end-i]
        valor_real = serie[end-i+1]

        try
            prediccion = metodo_prediccion(serie_entrenamiento)
            push!(errores, (valor_real - prediccion)^2)
        catch
            return Inf
        end
    end

    return sqrt(mean(errores))
end

"""
Predice el GGR del próximo mes usando el mejor método disponible

Métodos disponibles (en orden de preferencia):
1. Suavizado Exponencial Triple (Holt-Winters) - Para series con estacionalidad
2. Suavizado Exponencial Doble (Holt) - Para series con tendencia
3. Regresión Lineal - Para series con tendencia clara
4. Promedio Móvil Ponderado - Fallback para series cortas

NOTA: Para usar XGBoost o AutoARIMA, instalar los paquetes correspondientes:
  - XGBoost.jl: Requiere más datos y características adicionales
  - StateSpaceModels.jl: Para AutoARIMA (requiere instalación)
"""
function predecir_ggr_proximo_mes(df_mensual::DataFrame, metodo::Symbol=:auto)
    if nrow(df_mensual) == 0
        return 0.0
    end

    # Calcular GGR total por mes
    ggr_mensuales = Float64[]
    for row in eachrow(df_mensual)
        ggr_total = get(row, :apuestas_deportivas_ggr, 0.0) + get(row, :casino_ggr, 0.0)
        push!(ggr_mensuales, ggr_total)
    end

    if length(ggr_mensuales) == 0
        return 0.0
    end

    # Filtrar valores negativos o extremadamente grandes (outliers)
    ggr_mensuales = filter(x -> x >= 0 && x < 1e9, ggr_mensuales)

    if length(ggr_mensuales) == 0
        return 0.0
    end

    prediccion = 0.0

    # Selección automática del mejor método
    if metodo == :auto
        n = length(ggr_mensuales)

        if n >= 12
            # Suficientes datos para Holt-Winters con estacionalidad
            try
                prediccion = suavizado_exponencial_triple(ggr_mensuales, 12)
                # Validar que la predicción sea razonable
                if prediccion < 0 || prediccion > maximum(ggr_mensuales) * 2
                    # Si es muy extrema, usar método alternativo
                    prediccion = suavizado_exponencial_doble(ggr_mensuales)
                end
            catch
                prediccion = suavizado_exponencial_doble(ggr_mensuales)
            end

        elseif n >= 5
            # Evaluar múltiples métodos y elegir el mejor
            metodos = Dict(
                :holt => x -> suavizado_exponencial_doble(x),
                :regresion => x -> prediccion_regresion_lineal(x),
                :promedio => x -> promedio_movil_ponderado(x)
            )

            mejor_metodo = :promedio
            mejor_error = Inf

            for (nombre, metodo_func) in metodos
                try
                    error = validar_modelo(ggr_mensuales, metodo_func)
                    if error < mejor_error
                        mejor_error = error
                        mejor_metodo = nombre
                    end
                catch
                    continue
                end
            end

            # Usar el mejor método encontrado
            try
                prediccion = metodos[mejor_metodo](ggr_mensuales)
            catch
                prediccion = promedio_movil_ponderado(ggr_mensuales)
            end

        elseif n >= 3
            # Usar Holt (doble) o promedio móvil
            try
                pred_holt = suavizado_exponencial_doble(ggr_mensuales)
                pred_promedio = promedio_movil_ponderado(ggr_mensuales)

                # Promediar ambas predicciones para mayor robustez
                prediccion = (pred_holt + pred_promedio) / 2
            catch
                prediccion = promedio_movil_ponderado(ggr_mensuales)
            end

        else
            # Muy pocos datos, usar promedio móvil simple
            prediccion = promedio_movil_ponderado(ggr_mensuales)
        end

        # Métodos específicos (si se solicita uno en particular)
    elseif metodo == :holt_winters
        prediccion = suavizado_exponencial_triple(ggr_mensuales, 12)
    elseif metodo == :holt
        prediccion = suavizado_exponencial_doble(ggr_mensuales)
    elseif metodo == :regresion
        prediccion = prediccion_regresion_lineal(ggr_mensuales)
    elseif metodo == :promedio
        prediccion = promedio_movil_ponderado(ggr_mensuales)
    else
        # Por defecto, usar promedio móvil
        prediccion = promedio_movil_ponderado(ggr_mensuales)
    end

    # Validaciones finales
    prediccion = max(0, prediccion)  # No permitir valores negativos

    # Limitar predicción extrema (no más de 3x el máximo histórico)
    max_historico = maximum(ggr_mensuales)
    if prediccion > max_historico * 3
        prediccion = max_historico * 1.2  # Crecer solo 20% sobre el máximo
    end

    return round(prediccion, digits=2)
end

# ============================================================================
# PREDICCIÓN CREDITICIA
# ============================================================================

"""
Calcula el crédito sugerido: C = (0.6×P25 + 0.4×Mediana) × f_s × f_v × f_t × f_volumen
"""
function calcular_credito_sugerido(df_mensual::DataFrame, score::Float64, metricas::Dict{String,Float64})
    ngr_mensuales = df_mensual.calculo_ngr

    detalles_default = Dict(
        "p25" => 0.0,
        "cv" => 0.0,
        "f_volatilidad" => 0.0,
        "desc_volatilidad" => "Sin datos",
        "tendencia" => 0.0,
        "f_tendencia" => 0.0,
        "desc_tendencia" => "Sin datos",
        "f_score" => 0.0,
        "meses_historial" => 0,
        "mediana" => 0.0
    )

    if length(ngr_mensuales) == 0
        return 0.0, detalles_default
    end

    # Filtrar valores válidos
    ngr_validos = filter(x -> x > 0, ngr_mensuales)

    if length(ngr_validos) == 0
        return 0.0, detalles_default
    end

    # Percentil 25 y mediana
    p25 = calcular_percentil_25(ngr_validos)
    mediana = median(ngr_validos)

    # Base de crédito: mezcla de P25 (seguridad) y mediana (recompensa)
    # 60% P25 + 40% mediana → más generoso con agentes de alta comisión
    base_credito = p25 * 0.6 + mediana * 0.4

    # Factor de volatilidad con transformación logarítmica
    min_ngr = minimum(ngr_validos)
    ngr_desplazados = ngr_validos .+ abs(min_ngr) .+ 1
    ngr_log = log.(ngr_desplazados)

    cv_log = calcular_coeficiente_variacion(ngr_log)
    f_v, desc_volatilidad = calcular_factor_volatilidad(cv_log)

    # Factor de tendencia
    tendencia = calcular_tendencia_lineal(ngr_validos)
    f_t, desc_tendencia = calcular_factor_tendencia(tendencia)

    # Factor de score: f_s = 0.5 + 0.06 * ((S + E) / 2)
    S = score
    E = metricas["estabilidad"]
    f_s_final = 0.5 + 0.06 * ((S + E) / 2)

    # Factor de volumen - más generoso con comisiones altas
    comision_total = sum(ngr_validos)
    if comision_total >= 80000
        f_volumen = 2.0
    elseif comision_total >= 50000
        f_volumen = 1.7
    elseif comision_total >= 30000
        f_volumen = 1.4
    elseif comision_total >= 15000
        f_volumen = 1.2
    elseif comision_total >= 5000
        f_volumen = 1.0
    else
        f_volumen = 0.85
    end

    # Crédito base
    credito = base_credito * f_s_final * f_v * f_t * f_volumen

    # Límites de seguridad
    if p25 < 50
        credito = 0.0
    else
        limite_superior = 4 * mediana * f_volumen
        credito = min(credito, limite_superior)
    end

    if length(ngr_validos) < 3
        credito = credito * 0.5
    end

    detalles = Dict(
        "p25" => p25,
        "cv" => cv_log,
        "f_volatilidad" => f_v,
        "desc_volatilidad" => desc_volatilidad,
        "tendencia" => tendencia,
        "f_tendencia" => f_t,
        "desc_tendencia" => desc_tendencia,
        "f_score" => f_s_final,
        "meses_historial" => length(ngr_validos),
        "mediana" => mediana
    )

    return credito, detalles
end

# ============================================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================================

"""
Procesa todos los agentes y genera resultados
"""
function procesar_agentes(archivo_csv::String)
    println("="^60)
    println("PROCESAMIENTO DE AGENTES")
    println("="^60)

    # Cargar datos
    df = cargar_datos(archivo_csv)

    # Determinar mes de evaluación (ultimo mes con datos)
    meses_disponibles = unique(yearmonth.(df.creado))
    mes_evaluacion = maximum(meses_disponibles)
    fecha_eval = Date(mes_evaluacion[1], mes_evaluacion[2], 1)
    println("\n📅 Mes de evaluación: $(Dates.monthname(fecha_eval)) $(mes_evaluacion[1])")

    # Total de jugadores únicos del mes de evaluación
    mask_mes = [yearmonth(d) == mes_evaluacion for d in df.creado]
    df_mes = df[mask_mes, :]
    total_jugadores_global = length(unique(df_mes.jugador_id))
    println("Total jugadores únicos en el mes: $total_jugadores_global")

    # Obtener agentes con actividad en el mes de evaluación
    agentes = unique(df_mes.nombre_usuario_agente)
    total_agentes = length(agentes)

    println("\n🔄 Procesando $total_agentes agentes con actividad en $(Dates.monthname(fecha_eval))...")

    resultados = []

    for (idx, agente) in enumerate(agentes)
        if idx % 100 == 0
            println("Procesando $idx/$total_agentes...")
        end

        # Filtrar TODOS los datos del agente (historial completo)
        df_agente = df[df.nombre_usuario_agente.==agente, :]

        try
            # Calcular métricas para el mes de evaluación
            metricas, df_mensual = calcular_metricas_agente(df_agente, total_jugadores_global, mes_evaluacion)

            # Calcular score
            score = calcular_score_total(metricas)

            # Categorizar
            categoria, descripcion = categorizar_agente(score)

            # Calcular crédito
            credito, detalles = calcular_credito_sugerido(df_mensual, score, metricas)

            # Predecir GGR
            ggr_prediccion = predecir_ggr_proximo_mes(df_mensual)

            # Fechas
            fecha_min = minimum(df_agente.creado)
            fecha_max = maximum(df_agente.creado)

            resultado = Dict(
                "agente" => agente,
                "score" => round(score, digits=2),
                "categoria" => categoria,
                "descripcion_categoria" => descripcion,
                "credito_sugerido" => round(credito, digits=2),
                "ggr_prediccion" => ggr_prediccion,
                "fecha_inicio" => string(fecha_min),
                "fecha_fin" => string(fecha_max),
                "meses_historial" => detalles["meses_historial"],
                "p25" => round(detalles["p25"], digits=2),
                "mediana" => round(detalles["mediana"], digits=2),
                "metricas" => metricas
            )

            push!(resultados, resultado)

        catch e
            println("Error en $agente: $e")
            continue
        end
    end

    println("\n✓ Procesados $(length(resultados)) agentes")

    return resultados
end

"""
Muestra un resumen de los resultados
"""
function mostrar_resumen(resultados::Vector)
    println("\n" * "="^80)
    println("RESUMEN DE RESULTADOS")
    println("="^80)

    # Ordenar por score
    sort!(resultados, by=x -> x["score"], rev=true)

    # Top 10
    println("\n🏆 TOP 10 AGENTES POR SCORE:")
    println("-"^80)
    println(rpad("#", 5) * rpad("Agente", 30) * rpad("Score", 10) * rpad("Categoría", 12) * "Crédito")
    println("-"^80)

    for (i, res) in enumerate(resultados[1:min(10, length(resultados))])
        println(
            rpad(string(i), 5) *
            rpad(res["agente"], 30) *
            rpad(string(res["score"]), 10) *
            rpad(res["categoria"], 12) *
            "\$" * string(round(Int, res["credito_sugerido"]))
        )
    end

    # Detalle de métricas por agente (Top 10)
    println("\n" * "="^80)
    println("DETALLE DE MÉTRICAS POR AGENTE (TOP 10)")
    println("="^80)
    
    for (i, res) in enumerate(resultados[1:min(10, length(resultados))])
        metricas = res["metricas"]
        println("\n📋 #$i - $(res["agente"]) | Score: $(res["score"]) | Categoría: $(res["categoria"]) | Crédito: \$$(round(Int, res["credito_sugerido"]))")
        println("-"^80)
        println("  Rentabilidad (12%):    $(round(metricas["rentabilidad"], digits=2))")
        println("  Volumen (15%):         $(round(metricas["volumen"], digits=2))")
        println("  Fidelidad (15%):       $(round(metricas["fidelidad"], digits=2))")
        println("  Estabilidad (12%):     $(round(metricas["estabilidad"], digits=2))")
        println("  Crecimiento (10%):     $(round(metricas["crecimiento"], digits=2))")
        println("  Efic. Casino (8%):     $(round(metricas["eficiencia_casino"], digits=2))")
        println("  Efic. Deportes (8%):   $(round(metricas["eficiencia_deportes"], digits=2))")
        println("  Conversión (11%):      $(round(metricas["eficiencia_conversion"], digits=2))")
        println("  Tendencia (4%):        $(round(metricas["tendencia"], digits=2))")
        println("  Diversificación (3%):  $(round(metricas["diversificacion"], digits=2))")
        println("  Calidad (2%):          $(round(metricas["calidad_jugadores"], digits=2))")
    end
    
    # Análisis de similitud con el líder del mes
    if length(resultados) >= 2
        lider = resultados[1]
        lider_metricas = lider["metricas"]
        lider_score = lider["score"]

        println("\n" * "="^80)
        println("ANÁLISIS DE SIMILITUD CON EL LÍDER: $(lider["agente"]) (Score: $lider_score)")
        println("="^80)

        nombres_metricas = [
            ("rentabilidad", "Rentabilidad"),
            ("volumen", "Volumen"),
            ("fidelidad", "Fidelidad"),
            ("estabilidad", "Estabilidad"),
            ("crecimiento", "Crecimiento"),
            ("eficiencia_casino", "Efic. Casino"),
            ("eficiencia_deportes", "Efic. Deportes"),
            ("eficiencia_conversion", "Conversión"),
            ("tendencia", "Tendencia"),
            ("diversificacion", "Diversificación"),
            ("calidad_jugadores", "Calidad")
        ]

        for (i, res) in enumerate(resultados[2:min(10, length(resultados))])
            metricas = res["metricas"]
            sim_score = lider_score > 0 ? round(res["score"] / lider_score * 100, digits=1) : 0.0

            println("\n📊 #$(i+1) - $(res["agente"]) | Score: $(res["score"]) | Similitud global: $(sim_score)%")
            println("-"^80)

            for (key, nombre) in nombres_metricas
                val_agente = round(metricas[key], digits=2)
                val_lider = round(lider_metricas[key], digits=2)
                if val_lider > 0
                    sim = round(min(100.0, val_agente / val_lider * 100), digits=1)
                elseif val_agente == 0
                    sim = 100.0
                else
                    sim = 0.0
                end
                # Barra visual de similitud
                barra_len = round(Int, sim / 5)
                barra = "█"^barra_len * "░"^(20 - barra_len)
                println("  $(rpad(nombre, 18)) $(rpad(string(val_agente), 6))/ $(rpad(string(val_lider), 6)) $(barra) $(sim)%")
            end
        end
    end

    # Estadísticas generales
    println("\n📊 ESTADÍSTICAS GENERALES:")
    println("-"^80)
    println("Total de agentes: $(length(resultados))")
    println("Score promedio: $(round(mean([r["score"] for r in resultados]), digits=2))")
    println("Crédito total: \$$(round(Int, sum([r["credito_sugerido"] for r in resultados])))")

    # Conteo por categoría
    categorias_count = Dict{String,Int}()
    for res in resultados
        cat = res["categoria"]
        categorias_count[cat] = get(categorias_count, cat, 0) + 1
    end

    println("\n📈 DISTRIBUCIÓN POR CATEGORÍA:")
    for (cat, count) in sort(collect(categorias_count), by=x -> x[2], rev=true)
        println("  $cat: $count agentes")
    end

    println("="^80)
end

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

"""
Función principal para ejecutar el análisis
"""
function main()
    archivo = "reporte_detallado_jugadores_final.csv"

    if !isfile(archivo)
        println("❌ Error: No se encontró el archivo '$archivo'")
        println("Por favor, asegúrate de que el archivo esté en el directorio actual.")
        return
    end

    # Procesar agentes
    resultados = procesar_agentes(archivo)

    # Mostrar resumen
    mostrar_resumen(resultados)

    println("\n✅ Análisis completado exitosamente")
end

# Ejecutar si es el script principal
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
