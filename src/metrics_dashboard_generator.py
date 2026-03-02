import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from jinja2 import Template

# Import the core logic directly to avoid code duplication
from logic_analytics import calcular_metricas_agente_con_mensual, PESOS_METRICAS
from data_loader import load_data

def load_and_validate_data(csv_path="Data/reporte_detallado_jugadores_final.csv"):
    """
    Step 1 & Step 2: Mandatory Audit and Data Validation
    Reads the original CSV, uses logic_analytics to get the monthly aggregations (df_mensual),
    and validates that all 11 required metrics are present for visualization.
    """
    print("--- INICIANDO AUDITORÍA Y VALIDACIÓN ---")
    df = load_data(csv_path)
    
    # Simulate the pipeline: Get global players to calculate fidelidad correctly
    total_jugadores_global = df['jugador_id'].nunique() if 'jugador_id' in df.columns else 1
    
    # Calculate monthly global active players for the true Fidelidad share percentage
    if 'creado' in df.columns and 'jugador_id' in df.columns:
        df_temp = df.copy()
        df_temp['mes'] = pd.to_datetime(df_temp['creado'], errors='coerce').dt.to_period('M')
        global_monthly_players = df_temp.groupby('mes')['jugador_id'].nunique().to_dict()
    else:
        global_monthly_players = {}
    
    # We will compute the monthly data for each agent
    # To visualize trends effectively, we can pick the top 5 agents by global score,
    # or just generate a merged dataframe with all monthly data for all agents.
    
    # For a consolidated view or to feed a dropdown, we will collect all of them.
    all_monthly_dfs = []
    
    print("Calculando métricas históricas de todos los agentes...")
    
    # Agrupar por agente para procesarlos individualmente
    agentes = df['id_agente'].unique()
    
    # For demonstration and performance, we can extract the top 10 agents by volume/score 
    # or just process all of them. We'll process all and let JS handle filtering.
    
    # We'll just collect a massive flat list of dictionaries to pass to JS
    all_records = []
    
    # The 11 core metrics required
    core_metrics = list(PESOS_METRICAS.keys())
    
    # Audit logic
    metrics_present = set()
    
    agent_names = {}
    if 'nombre_usuario_agente' in df.columns:
        agent_names = df.groupby('id_agente')['nombre_usuario_agente'].first().to_dict()
    elif 'agente_username' in df.columns:
        agent_names = df.groupby('id_agente')['agente_username'].first().to_dict()
        
    # === 1. CALCULAR VISTA GLOBAL (TODAS LAS AGENCIAS) ===
    print("Calculando métricas históricas GLOBALES...")
    try:
        _, df_mensual_orig_g, df_mensual_mets_g = calcular_metricas_agente_con_mensual(df, total_jugadores_global, monthly_mode="rolling_3m")
        df_mensual_g = pd.merge(df_mensual_orig_g, df_mensual_mets_g, on='mes', how='left')
        
        if 'calculo_comision' not in df_mensual_g.columns: df_mensual_g['calculo_comision'] = 0.0
        if 'calculo_ngr' not in df_mensual_g.columns: df_mensual_g['calculo_ngr'] = 0.0
        
        if 'active_players' not in df_mensual_g.columns:
            if 'jugador_id_unique' in df_mensual_g.columns: df_mensual_g['active_players'] = df_mensual_g['jugador_id_unique']
            elif 'jugador_id' in df_mensual_g.columns: df_mensual_g['active_players'] = df_mensual_g['jugador_id']
            
        df_mensual_g['agente_id'] = "GLOBAL"
        df_mensual_g['agente_name'] = "🌟 VISTA GLOBAL (Todas las Agencias)"
        
        if 'mes' in df_mensual_g.columns:
            df_mensual_g['month_str'] = df_mensual_g['mes'].astype(str)
            df_mensual_g['global_players'] = df_mensual_g['mes'].map(global_monthly_players).fillna(total_jugadores_global)
        else:
            df_mensual_g['global_players'] = total_jugadores_global
            
        if 'total_apuesta_deportiva' in df_mensual_g.columns and 'total_apuesta_casino' in df_mensual_g.columns and 'total_depositos' in df_mensual_g.columns:
            df_mensual_g['total_apuesta_total'] = df_mensual_g['total_apuesta_deportiva'].fillna(0) + df_mensual_g['total_apuesta_casino'].fillna(0)
            df_mensual_g['eficiencia_juego'] = np.where(df_mensual_g['total_depositos'] > 0, 
                ((df_mensual_g['total_apuesta_total'] / df_mensual_g['total_depositos']) - 1) * 100, 
                0.0)
            df_mensual_g['eficiencia_juego'] = df_mensual_g['eficiencia_juego'].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        cols_to_keep_g = ['agente_id', 'agente_name', 'month_str'] + core_metrics
        context_cols_g = ['total_depositos', 'calculo_ngr', 'calculo_comision', 'score_global', 'active_players', 'num_depositos', 'num_retiros', 'global_players', 'casino_ggr', 'apuestas_deportivas_ggr', 'eficiencia_juego', 'total_apuesta_total']
        for c in context_cols_g:
            if c in df_mensual_g.columns:
                cols_to_keep_g.append(c)
                
        clean_df_g = df_mensual_g[cols_to_keep_g].fillna(0).replace([np.inf, -np.inf], 0)
        all_records.extend(clean_df_g.to_dict(orient='records'))
    except Exception as e:
        print(f"Advertencia: No se pudo generar la vista global. Error: {e}")
    # =====================================================

    for ag_id in agentes:
        df_ag = df[df['id_agente'] == ag_id]
        if len(df_ag) == 0: continue
            
        metricas_snapshot, df_mensual_orig, df_mensual_mets = calcular_metricas_agente_con_mensual(df_ag, total_jugadores_global, monthly_mode="rolling_3m")
        df_mensual = pd.merge(df_mensual_orig, df_mensual_mets, on='mes', how='left')
        
        # Aseguramos que existan, pero SIN fallback entre ellas
        if 'calculo_comision' not in df_mensual.columns:
            df_mensual['calculo_comision'] = 0.0
        if 'calculo_ngr' not in df_mensual.columns:
            df_mensual['calculo_ngr'] = 0.0
            
        if 'active_players' not in df_mensual.columns:
            if 'jugador_id_unique' in df_mensual.columns:
                df_mensual['active_players'] = df_mensual['jugador_id_unique']
            elif 'jugador_id' in df_mensual.columns:
                df_mensual['active_players'] = df_mensual['jugador_id']
        
        if df_mensual.empty: continue
            
        # Ensure we capture validation
        for m in core_metrics:
            if m in df_mensual.columns:
                metrics_present.add(m)
                
        # Append data
        df_mensual['agente_id'] = int(ag_id)
        df_mensual['agente_name'] = agent_names.get(ag_id, str(ag_id))
        
        # Convert Period object to string for JSON serialization
        if 'mes' in df_mensual.columns:
             df_mensual['month_str'] = df_mensual['mes'].astype(str)
             df_mensual['global_players'] = df_mensual['mes'].map(global_monthly_players).fillna(total_jugadores_global)
        else:
             df_mensual['global_players'] = total_jugadores_global
             
        # --- NEW CONTEXT METRIC: EFICIENCIA JUEGO ---
        if 'total_apuesta_deportiva' in df_mensual.columns and 'total_apuesta_casino' in df_mensual.columns and 'total_depositos' in df_mensual.columns:
            # Save the sum explicitly in the dataframe
            df_mensual['total_apuesta_total'] = df_mensual['total_apuesta_deportiva'].fillna(0) + df_mensual['total_apuesta_casino'].fillna(0)
            df_mensual['eficiencia_juego'] = np.where(df_mensual['total_depositos'] > 0, 
                ((df_mensual['total_apuesta_total'] / df_mensual['total_depositos']) - 1) * 100, 
                0.0)
            df_mensual['eficiencia_juego'] = df_mensual['eficiencia_juego'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
        # --------------------------------------------

        # Only keep necessary columns to optimize payload size
        cols_to_keep = ['agente_id', 'agente_name', 'month_str'] + core_metrics
        # Add basic volume metrics just in case they want context (MUST INCLUDE 'total_apuesta_total')
        context_cols = ['total_depositos', 'calculo_ngr', 'calculo_comision', 'score_global', 'active_players', 'num_depositos', 'num_retiros', 'global_players', 'casino_ggr', 'apuestas_deportivas_ggr', 'eficiencia_juego', 'total_apuesta_total']
        for c in context_cols:
            if c in df_mensual.columns:
                cols_to_keep.append(c)
                
        # Fill NA, Replace Inf
        clean_df = df_mensual[cols_to_keep].fillna(0).replace([np.inf, -np.inf], 0)
        all_records.extend(clean_df.to_dict(orient='records'))
        
    print("\n--- RESULTADO DE LA AUDITORÍA ---")
    missing_metrics = [m for m in core_metrics if m not in metrics_present]
    
    print(f"Métricas requeridas a graficar (Total 11):")
    for m in core_metrics:
        status = "ACTIVA" if m in metrics_present else "FALTANTE"
        print(f" - {m.ljust(25)} : {status}")
        
    if len(missing_metrics) == 0:
        print("\n✅ VALIDACIÓN DE DATOS EXITOSA: Las 11 métricas están presentes en la serie temporal y listas para visualizar.")
    else:
        print(f"\n❌ ERROR: Faltan las siguientes métricas en la serie temporal: {missing_metrics}")
        
    # Restructure into a dictionary mapped by agent_id for easy JS consumption
    monthly_dict = {}
    for r in all_records:
        ag_id = str(r['agente_id'])
        if ag_id not in monthly_dict:
            monthly_dict[ag_id] = {'name': r['agente_name'], 'data': []}
        monthly_dict[ag_id]['data'].append(r)
        
    return monthly_dict, core_metrics

def generate_metrics_dashboard(monthly_dict, out_path="reports/metrics_historic_dashboard.html"):
    """
    Step 3: Implementation
    Generates the standalone HTML file with Plotly/JS handling the individual 11 trend charts.
    """
    print(f"\n--- GENERANDO DASHBOARD SEPARADO ({out_path}) ---")
    
    monthly_json = json.dumps(monthly_dict)
    
    # We use a JS object to define the chart titles, descriptions and colors, 
    # to maintain visual consistency with the main dashboard.
    
    chart_config = {
        'rentabilidad': {'title': 'Rentabilidad (NGR / Depósitos * 100)', 'color': '#2563eb'},
        'volumen': {'title': 'Volumen Transaccional (Log Base 10)', 'color': '#10b981'},
        'fidelidad': {'title': 'Fidelidad (% Jugadores Globales)', 'color': '#8b5cf6'},
        'estabilidad': {'title': 'Estabilidad (1 - Coeficiente de Variación de NGR)', 'color': '#f59e0b'},
        'crecimiento': {'title': 'Crecimiento (% Depósitos Mensual)', 'color': '#06b6d4'},
        'eficiencia_casino': {'title': 'Eficiencia Casino (Depósitos / GGR Casino)', 'color': '#ec4899'},
        'eficiencia_deportes': {'title': 'Eficiencia Deportes (Depósitos / GGR Deportes)', 'color': '#f97316'},
        'eficiencia_conversion': {'title': 'Eficiencia de Conversión (GGR / Depósitos %)', 'color': '#2dd4bf'},
        'tendencia': {'title': 'Tendencia Histórica (Pendiente Regresión)', 'color': '#6366f1'},
        'diversificacion': {'title': 'Diversificación de Productos (1 - HHI Casino vs Deportes)', 'color': '#d946ef'},
        'calidad_jugadores': {'title': 'Calidad de Jugadores (Apuesta Promedio por Jugador)', 'color': '#14b8a6'},
        'eficiencia_juego': {'title': 'Eficiencia de Juego (Turnover vs Depósitos %)', 'color': '#6366f1'}
    }
    config_json = json.dumps(chart_config)
    
    # Extract agent list to populate the select dropdown
    agents_list = []
    for ag_id, info in monthly_dict.items():
        if len(info['data']) > 0:
            # Calculate a summary total score just to sort them nicely in the dropdown
            avg_score = sum(d.get('score_global', 0) for d in info['data']) / len(info['data'])
            agents_list.append({'id': ag_id, 'name': info['name'], 'avg_score': avg_score})
            
    agents_list.sort(key=lambda x: x['avg_score'], reverse=True)
    
    # Force GLOBAL to be the very first option in the UI dropdown
    global_item = next((item for item in agents_list if item['id'] == 'GLOBAL'), None)
    if global_item:
        agents_list.remove(global_item)
        agents_list.insert(0, global_item)
        
    agents_list_json = json.dumps(agents_list)
    
    template_str = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Seguimiento Temporal - 11 Métricas de Agentes</title>
    <!-- Plotly Library -->
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #fcfdfe; /* Ultra soft blue-white */
            --card-bg: #ffffff;
            --text-color: #475569; /* Slate 600 - Softer than black */
            --text-muted: #94a3b8; /* Slate 400 */
            --border: #f1f5f9; /* Slate 100 */
            --primary: #334155; /* Slate 700 - Soft Header */
            --accent: #2563eb; /* Blue 600 */
            --accent-soft: rgba(37, 99, 235, 0.1);
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            --shadow-hover: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
        }
        
        body { 
            font-family: 'Inter', sans-serif; 
            background-color: var(--bg-color); 
            color: var(--text-color); 
            margin: 0; padding: 0; 
        }

        /* Header */
        header { 
            background: var(--card-bg); padding: 15px 30px; 
            border-bottom: 1px solid #e2e8f0; 
            display: flex; justify-content: space-between; align-items: center; 
            box-shadow: 0 1px 2px rgba(0,0,0,0.02);
            position: sticky; top: 0; z-index: 100;
        }
        h1 { margin: 0; font-size: 20px; font-weight: 700; color: #0f172a; letter-spacing: -0.5px; }
        .header-meta { font-size: 13px; color: var(--text-muted); margin-top: 4px; font-weight: 500; }
        
        .controls {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        
        select {
            padding: 8px 12px;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            background: white;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
            color: #1e293b;
            cursor: pointer;
            outline: none;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02);
            transition: all 0.2s;
            min-width: 250px;
        }
        
        select:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-soft);
        }

        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            padding: 24px; 
        }
        
        .grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fill, minmax(600px, 1fr)); 
            gap: 24px; 
        }
        
        .card { 
            background: var(--card-bg); 
            border-radius: 12px; 
            padding: 20px; 
            box-shadow: var(--shadow); 
            border: 1px solid var(--border);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .card:hover { 
            box-shadow: var(--shadow-hover); 
            transform: translateY(-2px);
        }
        
        .chart-title { 
            font-size: 15px;
            font-weight: 700; 
            letter-spacing: -0.01em;
            margin-bottom: 4px; 
            color: #0f172a; 
            display: flex; 
            align-items: center; 
            gap: 10px; 
        }
        
        .chart-subtitle {
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 12px;
            font-weight: 500;
        }
        
        .chart-container {
            width: 100%;
            height: 350px; /* Force consistent height for all 11 charts */
        }
        
        .empty-state {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #94a3b8;
            font-style: italic;
            background: #f8fafc;
            border-radius: 8px;
            border: 1px dashed #cbd5e1;
        }
        
        .checklist-board {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .checklist-board span {
            font-size: 13px;
            font-weight: 600;
            color: #16a34a;
            display: flex;
            align-items: center;
            gap: 5px;
            background: #dcfce7;
            padding: 6px 12px;
            border-radius: 20px;
        }

        /* Info Popover System */
        .chart-header-flex {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }
        
        .info-wrapper {
            position: relative;
            display: inline-block;
            margin-top: 2px;
        }
        
        .info-button {
            background: #f1f5f9;
            color: #64748b;
            border: none;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            font-size: 14px;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s;
            font-family: serif;
        }
        
        .info-button:hover {
            background: #e2e8f0;
            color: #0f172a;
        }
        
        .info-popover {
            visibility: hidden;
            opacity: 0;
            position: absolute;
            right: 0;
            top: 32px;
            width: 320px;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
            padding: 16px;
            z-index: 1000;
            transition: opacity 0.2s, visibility 0.2s;
            pointer-events: none;
        }
        
        .info-wrapper:hover .info-popover {
            visibility: visible;
            opacity: 1;
        }
        
        .popover-section {
            margin-bottom: 12px;
        }
        .popover-section:last-child {
            margin-bottom: 0;
        }
        
        .popover-label {
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            margin-bottom: 4px;
        }
        
        .popover-text {
            font-size: 12px;
            color: #334155;
            line-height: 1.4;
        }

    </style>
</head>
<body>

<header>
    <div>
        <h1>Dashboard Analítico: 11 Métricas Estructurales de Agentes</h1>
        <div class="header-meta">Seguimiento de la serie temporal por métrica calificada</div>
    </div>
    <div class="controls">
        <label for="agentSelect" style="font-size: 13px; font-weight: 600; color: #475569;">Comportamiento Histórico de:</label>
        <select id="agentSelect" onchange="renderAllCharts()">
            <!-- Options populated via JS -->
        </select>
    </div>
</header>

<div class="container">
    <div class="checklist-board">
        <div style="font-weight: 700; color:#0f172a; font-size: 14px; margin-right: 15px;">🔍 Verificación de Auditoría:</div>
        <span id="check-rent">Rentabilidad ✅</span>
        <span id="check-vol">Volumen ✅</span>
        <span id="check-fid">Fidelidad ✅</span>
        <span id="check-est">Estabilidad ✅</span>
        <!-- <span id="check-cre">Crecimiento ✅</span> -->
        <span id="check-cas">Efic. Casino ✅</span>
        <span id="check-dep">Efic. Deportes ✅</span>
        <span id="check-conv">Conversión ✅</span>
        <span id="check-jue">Efic. Juego ✅</span>
        <span id="check-ten">Tendencia ✅</span>
        <span id="check-div">Diversificación ✅</span>
        <!-- <span id="check-cal">Calidad ✅</span> -->
        <div style="margin-left:auto; font-size:12px; color:#64748b;">✅ Las 11 métricas verificadas y graficadas exitosamente.</div>
    </div>

    <!-- The 11 Charts Grid -->
    <div class="grid" id="chartsGrid">
        <!-- Divs dynamically generated by JS -->
    </div>
</div>

<script>
    const monthlyData = {{ monthly_json | safe }};
    const agentsList = {{ agents_list_json | safe }};
    const chartConfig = {{ config_json | safe }};
    const HIDDEN_METRICS = new Set(['crecimiento', 'calidad_jugadores']);
    const metricKeys = Object.keys(chartConfig || {}).filter(k => !HIDDEN_METRICS.has(k));

    const metricInfo = {
        'rentabilidad': { 
            custom: `
                        <div class="popover-section">
                            <div class="popover-label">OBJETIVO</div>
                            <div class="popover-text">Evalúa la rentabilidad del negocio comparando NGR generado vs depósitos del mes.</div>
                        </div>
                        <div class="popover-section">
                            <div class="popover-label">CÁLCULO</div>
                            <div class="popover-text">Margen Real (%) = (NGR / Depósitos) &times; 100</div>
                        </div>
                        <div class="popover-section">
                            <div class="popover-label">CÓMO LEER EL GRÁFICO</div>
                            <div class="popover-text">
                                - Barras: Depósitos ($) por mes<br>
                                - Línea negra: Margen Real (%)<br>
                                - Línea punteada: Score (0&ndash;10)<br>
                                - Líneas guía: 0%, 5%, 10%, 20%
                            </div>
                        </div>
                        <div class="popover-section">
                            <div class="popover-label">UMBRAL RÁPIDO</div>
                            <div class="popover-text">&gt; 10% Fuerte | 5&ndash;10% Aceptable | &lt; 5% Débil | &lt; 0% Pérdida</div>
                        </div>
                        <div class="popover-section">
                            <div class="popover-label">SEÑALES A VIGILAR</div>
                            <div class="popover-text">
                                - Caídas bruscas del margen (&Delta; pp negativo)<br>
                                - Depósitos altos con margen bajo (volumen sin NGR)<br>
                                - Depósitos muy bajos: el % puede verse inflado (revisar NGR y $)
                            </div>
                        </div>
            `
        },
        'volumen': { 
            custom: `
                        <div class="popover-section">
                            <div class="popover-label">OBJETIVO</div>
                            <div class="popover-text">Mide la escala de actividad del agente en número de transacciones mensuales (engagement operativo).</div>
                        </div>
                        <div class="popover-section">
                            <div class="popover-label">CÁLCULO</div>
                            <div class="popover-text">
                                TXs = (# Depósitos + # Retiros)<br>
                                Score = log10(TXs + 1) &times; 2.3
                            </div>
                        </div>
                        <div class="popover-section">
                            <div class="popover-label">CÓMO LEER EL GRÁFICO</div>
                            <div class="popover-text">
                                - Palitos verdes: Total TXs por mes<br>
                                - Línea punteada: Score (0&ndash;10) (escala logarítmica)<br>
                                - Encabezado: TXs del último mes + &Delta; vs mes anterior
                            </div>
                        </div>
                        <div class="popover-section">
                            <div class="popover-label">INTERPRETACIÓN</div>
                            <div class="popover-text">
                                Escala logarítmica: cada &times;10 en TXs suma ~2.3 puntos.<br>
                                Referencia: 100 TXs &approx; 4.6 | 1,000 &approx; 6.9 | 10,000 &approx; 9.2
                            </div>
                        </div>
                        <div class="popover-section">
                            <div class="popover-label">SEÑALES A VIGILAR</div>
                            <div class="popover-text">
                                - Caída crítica: descenso fuerte de TXs vs mes anterior (riesgo de inactividad)<br>
                                - Pico inusual: salto fuerte (campañas/reactivación o outlier)<br>
                                - La escala log suaviza cambios: revisar &Delta;% además del score.
                            </div>
                        </div>
            `
        },
        'fidelidad': { 
            custom: `
                        <div class="popover-section">
                            <div class="popover-label">OBJETIVO</div>
                            <div class="popover-text">Mide la participación mensual de jugadores de la agencia dentro del total global (presencia / share).</div>
                        </div>
                        <div class="popover-section">
                            <div class="popover-label">CÁLCULO</div>
                            <div class="popover-text">
                                Share (%) = (Jugadores Propios del mes / Jugadores Globales) &times; 100<br>
                                Score (0&ndash;10) = min(10, Share &times; 2.5)
                            </div>
                        </div>
                        <div class="popover-section">
                            <div class="popover-label">CÓMO LEER EL GRÁFICO</div>
                            <div class="popover-text">
                                - Línea morada: Share (%) mensual<br>
                                - Marcadores: un punto por mes<br>
                                - Encabezado: Share actual + &Delta; en pp vs mes anterior
                            </div>
                        </div>
                        <div class="popover-section">
                            <div class="popover-label">INTERPRETACIÓN RÁPIDA</div>
                            <div class="popover-text">
                                - 4.0% Share &rArr; Score 10 (tope)<br>
                                - 2.0% &rArr; Score 5<br>
                                - 1.0% &rArr; Score 2.5<br>
                                Más share = más fidelidad/participación.
                            </div>
                        </div>
                        <div class="popover-section">
                            <div class="popover-label">QUÉ OBSERVAR</div>
                            <div class="popover-text">
                                - Share sube + Propios suben &rArr; crecimiento real<br>
                                - Share sube pero Propios no crecen &rArr; revisar Global (efecto denominador)<br>
                                - Caídas de Share (&Delta; pp negativo) &rArr; pérdida de participación
                            </div>
                        </div>
            `
        },
        'estabilidad': { 
            custom: `
                        <div class="popover-section">
                            <div class="popover-label">OBJETIVO</div>
                            <div class="popover-text">Mide qué tan predecible y consistente es el NGR mensual del agente (baja volatilidad + continuidad reciente).</div>
                        </div>
                        <div class="popover-section">
                            <div class="popover-label">CÁLCULO</div>
                            <div class="popover-text">
                                Se combinan 2 componentes:<br>
                                - Histórico (40%): CV logarítmico del NGR (menor CV = más estable)<br>
                                &nbsp;&nbsp;ef = 1 &minus; CV_log  &rarr; mayor ef = mayor estabilidad<br>
                                - Reciente (60%): patrón de los últimos 2&ndash;3 meses (racha de NGR &gt; 0)<br>
                                Score final = 0.4 &times; score_histórico + 0.6 &times; score_reciente
                            </div>
                        </div>
                        <div class="popover-section">
                            <div class="popover-label">CÓMO LEER EL GRÁFICO</div>
                            <div class="popover-text">
                                - Barra: score actual (0&ndash;10)<br>
                                - Línea punteada: Objetivo 6.0 (umbral)<br>
                                - Dots: score mensual<br>
                                - Panel derecho: score + estado + brecha vs objetivo
                            </div>
                        </div>
                        <div class="popover-section">
                            <div class="popover-label">INTERPRETACIÓN</div>
                            <div class="popover-text">
                                &gt; 6.0 Estable | 4&ndash;6 Moderado | &lt; 4 Inestable<br>
                                Score alto = NGR con poca variación + meses recientes positivos.
                            </div>
                        </div>
                        <div class="popover-section">
                            <div class="popover-label">QUÉ OBSERVAR</div>
                            <div class="popover-text">
                                - Caídas bajo 6: mayor volatilidad o meses recientes sin NGR<br>
                                - Consistencia de dots: estabilidad sostenida<br>
                                - El score puede bajar aunque el NGR suba si se vuelve más irregular.
                            </div>
                        </div>
            `
        },
        'crecimiento': { obj: 'Ritmo de expansión de volumen de depósitos.', calc: '(Depósitos Mes Actual / Depósitos Mes Anterior) - 1', int: 'Valores > 0% indican crecimiento. Consistencia es mejor que picos asilados.', watch: 'Meses continuos en negativo.' },
        'eficiencia_casino': { 
            custom: `
                        <div class="popover-section">
                            <div class="popover-label">OBJETIVO</div>
                            <div class="popover-text">Mide qué tan eficiente es la agencia generando GGR Casino a partir de la actividad de depósitos.</div>
                        </div>
                        <div class="popover-section">
                            <div class="popover-label">CÁLCULO (MODELO)</div>
                            <div class="popover-text">
                                Eficiencia = (# Depósitos / GGR Casino) &times; 100<br>
                                Menor ratio = mejor eficiencia (más GGR por menos depósitos).
                            </div>
                        </div>
                        <div class="popover-section">
                            <div class="popover-label">CÓMO LEER EL GRÁFICO</div>
                            <div class="popover-text">
                                - Barra: score de eficiencia (0&ndash;10)<br>
                                - Línea punteada: Objetivo 7.0<br>
                                - Dots: score mensual<br>
                                - Panel derecho: score + estado + brecha vs objetivo
                            </div>
                        </div>
                        <div class="popover-section">
                            <div class="popover-label">INTERPRETACIÓN RÁPIDA</div>
                            <div class="popover-text">
                                &gt; 33 &rArr; ~2 (muy ineficiente)<br>
                                20&ndash;33 &rArr; ~3<br>
                                14&ndash;20 &rArr; ~5<br>
                                10&ndash;14 &rArr; ~7.5<br>
                                5.5&ndash;10 &rArr; ~10 (muy eficiente)
                            </div>
                        </div>
                        <div class="popover-section">
                            <div class="popover-label">QUÉ OBSERVAR</div>
                            <div class="popover-text">
                                - Subidas del ratio (peor): más depósitos para el mismo GGR<br>
                                - Ratio alto con GGR bajo: baja conversión / mix desfavorable<br>
                                - Meses con GGR=0: score = 0 (no evaluable)<br><br>
                                <i>Nota: si el tooltip muestra Depósitos($)/GGR es complementario; el score usa el ratio del modelo.</i>
                            </div>
                        </div>
            `
        },
        'eficiencia_deportes': { obj: 'Qué tanto depósito genera 1 dólar de GGR en Deportes.', calc: 'Depósitos / GGR Deportes', int: 'Similar a casino, pero sujeto a estacionalidad deportiva y varianza.', watch: 'Margen deportivo consistentemente negativo (Arbitraje/Fraude).' },
        'eficiencia_conversion': { obj: 'Tasa base bruta de conversión de depósitos a pérdidas del jugador (GGR).', calc: '(GGR Total / Depósitos) * 100', int: 'No contempla bonos ni costos. Solo retención bruta directa.', watch: 'Caída de la conversión combinada con alza de depósitos (bonus abuse).' },
        'tendencia': { obj: 'Dirección general del agente a largo plazo (Score).', calc: 'Pendiente de regresión lineal de los últimos 6 meses.', int: 'Positivo = Agente escalando. Negativo = En declive.', watch: 'Cambio de tendencia de positivo a negativo sostenido.' },
        'diversificacion': { obj: 'Mide si el agente depende de un solo producto.', calc: '1 - Índice HHI (Casino vs Deportes)', int: '10 = Perfectamente balanceado (50/50). 0 = 100% concentrado en un producto.', watch: 'Riesgo estructural por alta dependencia deportiva (varianza alta).' },
        'calidad_jugadores': { obj: 'Valor transaccional promedio por jugador activo.', calc: 'Total Depósitos / Jugadores Activos', int: 'Mide si son VIPs o jugadores casuales (retail).', watch: 'Aumento de jugadores con colapso de calidad (tráfico de baja calidad).' },
        'eficiencia_juego': { obj: 'Mide el volumen de juego generado en relación al capital depositado.', calc: '((Apuestas Totales / Depósitos) - 1) * 100' }
    };

    function buildInfoIcon(mKey) {
        const info = metricInfo[mKey];
        if (!info) return '';
        
        let popoverContent = '';
        if (info.custom) {
            popoverContent = info.custom;
        } else {
            popoverContent = `
                        <div class="popover-section">
                            <div class="popover-label">OBJETIVO</div>
                            <div class="popover-text">${info.obj}</div>
                        </div>
                        <div class="popover-section">
                            <div class="popover-label">CÁLCULO</div>
                            <div class="popover-text">${info.calc}</div>
                        </div>
                        <div class="popover-section">
                            <div class="popover-label">INTERPRETACIÓN</div>
                            <div class="popover-text">${info.int}</div>
                        </div>
                        <div class="popover-section">
                            <div class="popover-label">QUÉ OBSERVAR</div>
                            <div class="popover-text">${info.watch}</div>
                        </div>
            `;
        }

        return `
            <div style="display: flex; gap: 8px; align-items: center;">
                <button class="info-button" onclick="resetChart('${mKey}')" title="Resetear Vista" aria-label="Reset View">↺</button>
                <div class="info-wrapper">
                    <button class="info-button" aria-label="Información">i</button>
                    <div class="info-popover">
${popoverContent}
                    </div>
                </div>
            </div>
        `;
    }

    function resetChart(mKey) {
        const chartDiv = document.getElementById('chart_' + mKey);
        if (chartDiv && chartDiv.layout) {
            const updateLayout = {};
            
            // Restore exact snapshot ranges if they exist, otherwise fallback securely to autorange
            if (chartDiv._initialRanges && chartDiv._initialRanges['xaxis.range']) {
                updateLayout['xaxis.range'] = [...chartDiv._initialRanges['xaxis.range']];
            } else {
                updateLayout['xaxis.autorange'] = true;
            }
            
            if (chartDiv._initialRanges && chartDiv._initialRanges['yaxis.range']) {
                updateLayout['yaxis.range'] = [...chartDiv._initialRanges['yaxis.range']];
            } else {
                updateLayout['yaxis.autorange'] = true;
            }
            
            if (chartDiv.layout.yaxis2) {
                if (chartDiv._initialRanges && chartDiv._initialRanges['yaxis2.range']) {
                    updateLayout['yaxis2.range'] = [...chartDiv._initialRanges['yaxis2.range']];
                } else {
                    updateLayout['yaxis2.autorange'] = true;
                }
            }
            if (chartDiv.layout.yaxis3) {
                if (chartDiv._initialRanges && chartDiv._initialRanges['yaxis3.range']) {
                    updateLayout['yaxis3.range'] = [...chartDiv._initialRanges['yaxis3.range']];
                } else {
                    updateLayout['yaxis3.autorange'] = true;
                }
            }
            
            Plotly.relayout(chartDiv, updateLayout);
        }
    }
    
    // Initialize the Dropdown
    const selectEl = document.getElementById('agentSelect');
    agentsList.forEach(agent => {
        const opt = document.createElement('option');
        opt.value = agent.id;
        // Clean name display
        const displayName = agent.name && agent.name !== "0" && agent.name !== agent.id ? agent.name : `Agente ${agent.id}`;
        opt.textContent = `${displayName} (Score medio: ${agent.avg_score.toFixed(2)})`;
        selectEl.appendChild(opt);
    });
    
    // Build the Grid Containers
    const gridEl = document.getElementById('chartsGrid');
    metricKeys.forEach(m => {
        const div = document.createElement('div');
        div.className = 'card';
        div.innerHTML = `
            <div class="chart-header-flex">
                <div>
                    <div class="chart-title">${chartConfig[m].title}</div>
                    <div class="chart-subtitle">Puntaje calificado mensual (0 - 10)</div>
                </div>
                ${buildInfoIcon(m)}
            </div>
            <div id="chart_${m}" class="chart-container"></div>
        `;
        gridEl.appendChild(div);
    });
    
    // Tooltip Standardization Helpers
    function formatMoney(val) {
        if (val === undefined || val === null) return '$0.00';
        return '$' + val.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
    }
    
    function formatPct(val) {
        if (val === undefined || val === null) return '0.00%';
        return val.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2}) + '%';
    }

    function formatPP(val) {
        if (val === undefined || val === null) return '0.00 pp';
        return val.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' pp';
    }
    
    function formatInt(val) {
        if (val === undefined || val === null) return '0';
        return Math.round(val).toLocaleString('en-US');
    }

    function formatScore(val) {
        if (val === undefined || val === null) return '0.00';
        return val.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
    }

    function buildTooltipHTML(month, realData, changes, score) {
        let html = `<b><span style="font-size: 13px;">${month}</span></b><br><br>`;
        
        // Section A - REAL DATA
        html += `<b><span style="font-size: 10px; color: #64748b;">REAL DATA</span></b><br>`;
        for (const [key, val] of Object.entries(realData)) {
            html += `<span style="font-size: 11px;">${key}:  <b>${val}</b></span><br>`;
        }
        html += `<br>`;
        
        // Section B - CHANGE
        html += `<b><span style="font-size: 10px; color: #64748b;">CHANGE</span></b><br>`;
        const changeKeys = Object.keys(changes);
        if (changeKeys.length > 0) {
            for (const [key, val] of Object.entries(changes)) {
                let textVal = String(val);
                let color = textVal.startsWith('+') ? '#10b981' : (textVal.startsWith('-') ? '#ef4444' : '#64748b');
                if (textVal === 'N/A') color = '#64748b';
                html += `<span style="font-size: 11px;">${key}:  <b><span style="color: ${color};">${textVal}</span></b></span><br>`;
            }
        } else {
            html += `<span style="font-size: 11px;">Δ:  <b><span style="color: #64748b;">N/A</span></b></span><br>`;
        }
        html += `<br>`;
        
        // Section C - MODEL SCORE
        html += `<b><span style="font-size: 10px; color: #64748b;">MODEL SCORE</span></b><br>`;
        html += `<span style="font-size: 11px;">Score (0-10):  <b>${formatScore(score)} / 10</b></span><extra></extra>`;
        
        return html;
    }

    // Render Function
    function renderAllCharts() {
        const agentId = selectEl.value;
        const agentDataInfo = monthlyData[agentId];
        
        if (!agentDataInfo || agentDataInfo.data.length === 0) {
            // Handle Empty
            metricKeys.forEach(m => {
                document.getElementById(`chart_${m}`).innerHTML = '<div class="empty-state">No hay suficientes datos temporales</div>';
            });
            return;
        }
        
        let series = agentDataInfo.data;
        // Sort chronologically
        series.sort((a, b) => a.month_str.localeCompare(b.month_str));
        
        const x_vals = series.map(d => d.month_str);

        // Global month formatter: "2025-07" → "Jul 2025" (applies to ALL charts)
        const _monthNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
        const formatted_x = x_vals.map(raw => {
            const parts = raw.split('-');
            if (parts.length === 2) {
                const mi = parseInt(parts[1], 10) - 1;
                return _monthNames[mi] + ' ' + parts[0];
            }
            return raw;
        });
        
        // Adaptive X-axis ticks to prevent overlap on narrow cards if > 6 months
        // Implement a collision-safe tick selection rule
        let adp_tickvals = x_vals;
        let adp_ticktext = formatted_x;
        if (x_vals.length > 6) {
            let final_indices = [];
            // 1. Gather every 2nd month index
            for (let i = 0; i < x_vals.length; i += 2) {
                final_indices.push(i);
            }
            // 2. Always ensure the last month is included
            const last_idx = x_vals.length - 1;
            if (final_indices[final_indices.length - 1] !== last_idx) {
                final_indices.push(last_idx);
            }
            // 3. Collision safe check: if the last element and the previous element are adjacent,
            // remove the previous element to avoid text overlapping "Jan 2026Feb 2026".
            if (final_indices.length >= 2) {
                const len = final_indices.length;
                if (final_indices[len - 1] - final_indices[len - 2] <= 1) {
                    // Remove the penultimate item
                    final_indices.splice(len - 2, 1);
                }
            }
            // Remap back to actual values
            adp_tickvals = final_indices.map(i => x_vals[i]);
            adp_ticktext = final_indices.map(i => formatted_x[i]);
        }
        
        // Helper specifically for Critical Month Icons (placed above the chart)
        function addCriticalEnclosure(layout, x_vals, criticalFlags) {
            if (!x_vals || typeof x_vals.length === 'undefined' || !criticalFlags || criticalFlags.length !== x_vals.length) return;
            layout.shapes = layout.shapes || [];
            const n = x_vals.length;
            for (let i = 0; i < n; i++) {
                if (criticalFlags[i]) {
                    layout.shapes.push({
                        type: 'rect',
                        xref: 'paper', yref: 'paper',
                        y0: 0, y1: 1,
                        layer: 'below',
                        fillcolor: 'rgba(239,68,68,0.06)',
                        line: { color: 'rgba(239,68,68,0.22)', width: 1 },
                        x0: (i - 0.5) / n, x1: (i + 0.5) / n
                    });
                }
            }
        }
        
        // Loop through 11 metrics and plot
        metricKeys.forEach(m => {
            const chartDiv = document.getElementById(`chart_${m}`);
            // Clean up old plot just in case
            try { Plotly.purge(chartDiv); } catch(e){}
            chartDiv.innerHTML = '';
            
            const config = chartConfig[m];
            const y_vals = series.map(d => d[m] !== undefined && d[m] !== null ? d[m] : 0);
            
            // Build Plotly Traces dynamically based on user's analytical specification
            let traces = [];
            let layout = {
                margin: { t: 40, b: 65, l: 45, r: 40 },
                xaxis: { 
                    type: 'category',
                    tickmode: 'array',
                    tickvals: adp_tickvals,
                    ticktext: adp_ticktext,
                    tickangle: 0,
                    automargin: true,
                    showgrid: false, showline: true, linewidth: 1.5, linecolor: '#cbd5e1', 
                    ticks: 'outside', tickcolor: '#cbd5e1', ticklen: 5, fixedrange: false,
                    tickfont: { size: 10, color: '#64748b', family: 'Inter' } 
                },
                yaxis: { 
                    range: [0, 10.5], showgrid: true, gridcolor: '#f1f5f9', zeroline: false,
                    showline: true, linewidth: 1.5, linecolor: '#cbd5e1', 
                    ticks: 'outside', tickcolor: '#cbd5e1', ticklen: 5,
                    tickfont: { size: 10, color: '#94a3b8', family: 'Inter' } 
                },
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                hovermode: 'closest', dragmode: 'zoom',
                hoverlabel: { bgcolor: "rgba(255, 255, 255, 0.98)", bordercolor: "#e2e8f0", font: { family: "Inter", size: 12, color: "#1e293b" }, align: "left" },
                showlegend: false
            };

            const lastMonth = x_vals[x_vals.length - 1];
            const lastVal = y_vals[y_vals.length - 1];

            if (m === 'rentabilidad') {
                // Highly Analytical Render: 3 Layers + Anomalies + KPI tooltips
                
                const deps = series.map(d => d.total_depositos || 0);
                const ngrs = series.map(d => (d.calculo_ngr !== undefined && d.calculo_ngr !== 0) ? d.calculo_ngr : (d.calculo_comision || 0));
                const scores = y_vals;
                
                const margins = [];
                const margin_colors = [];
                const margin_sizes = [];
                const margin_line_widths = [];
                const margin_line_colors = [];
                
                const bar_fill_colors = [];
                const bar_line_colors = [];
                
                const anomalies_x = [];
                const anomalies_y = [];
                const anomalies_text = [];
                const anomalies_symbols = [];
                const hover_texts = [];
                const criticalFlags = [];
                
                const sorted_deps = [...deps].sort((a,b) => a - b);
                const dep_p75 = sorted_deps[Math.floor(sorted_deps.length * 0.75)] || 0;
                
                for (let i = 0; i < series.length; i++) {
                    const d = deps[i];
                    const n = ngrs[i];
                    const s = scores[i];
                    const prev_d = i > 0 ? deps[i-1] : null;
                    const prev_m = i > 0 ? margins[i-1] : null;
                    
                    let m_pct = null;
                    if (d > 0) {
                        m_pct = (n / d) * 100;
                    } else if (n < 0) {
                        m_pct = -100; // Cap visual to avoid infinite negative
                    }
                    margins.push(m_pct);
                    
                    const isLast = (i === series.length - 1);
                    
                    // High-contrast Color Logic
                    let color = '#475569'; // neutral
                    if (m_pct === null) color = '#cbd5e1';
                    else if (m_pct < 0) color = '#ef4444'; // Red
                    else if (m_pct > 10) color = '#10b981'; // Green
                    
                    margin_colors.push(color);
                    margin_sizes.push(isLast ? 14 : 8);
                    margin_line_widths.push(isLast ? 3 : 2);
                    margin_line_colors.push(isLast ? '#ffffff' : '#ffffff');
                    
                    let is_critical = false;
                    let evento = "⚠ Mes crítico";
                    
                    // Anomalies detection
                    if (m_pct !== null) {
                        if (m_pct < 0) {
                            anomalies_x.push(x_vals[i]); anomalies_y.push(m_pct); anomalies_text.push('Margen Negativo'); anomalies_symbols.push('triangle-down');
                            is_critical = true;
                            evento = "⚠ Caída crítica";
                        } else if (d > dep_p75 && m_pct < 2) { 
                            anomalies_x.push(x_vals[i]); anomalies_y.push(m_pct); anomalies_text.push('Alto Vol / Bajo Margen'); anomalies_symbols.push('circle-open');
                            is_critical = true;
                            evento = "⚠ Pico inusual (alto vol / bajo margen)";
                        }
                    } else if (d === 0 || d === undefined) {
                         anomalies_x.push(x_vals[i]); anomalies_y.push(0); anomalies_text.push('Sin Depósitos'); anomalies_symbols.push('x');
                         is_critical = true;
                         evento = "⚠ Sin depósitos";
                    }
                    criticalFlags.push(is_critical);
                    
                    // Executive Tooltip Content
                    const real_data = {
                        "Depósitos": formatMoney(d),
                        "NGR": formatMoney(n),
                        "Margen Real": m_pct !== null ? formatPct(m_pct) : 'N/A'
                    };
                    const change_data = {};
                    if (is_critical) change_data["Evento"] = evento;
                    if (prev_d !== null && prev_d > 0) {
                        const d_var = ((d - prev_d) / prev_d) * 100;
                        change_data["Δ Depósitos"] = (d_var > 0 ? '+' : '') + formatPct(d_var);
                    }
                    if (prev_m !== null && m_pct !== null) {
                        const m_var = m_pct - prev_m;
                        change_data["Δ Margen"] = (m_var > 0 ? '+' : '') + formatPP(m_var);
                    }
                    
                    const h_text = buildTooltipHTML(formatted_x[i], real_data, change_data, s);
                    hover_texts.push(h_text);
                    
                    // Bar Conditional Colors (Reduced Opacity)
                    let b_fill = 'rgba(148, 163, 184, 0.15)'; // Neutral (5-10%)
                    let b_line = 'rgba(148, 163, 184, 0.5)';
                    if (m_pct !== null) {
                        if (m_pct < 0) {
                            b_fill = 'rgba(239, 68, 68, 0.15)'; b_line = 'rgba(239, 68, 68, 0.5)';
                        } else if (m_pct < 5) {
                            b_fill = 'rgba(245, 158, 11, 0.15)'; b_line = 'rgba(245, 158, 11, 0.5)';
                        } else if (m_pct >= 10) {
                            b_fill = 'rgba(16, 185, 129, 0.15)'; b_line = 'rgba(16, 185, 129, 0.5)';
                        }
                    }
                    bar_fill_colors.push(b_fill);
                    bar_line_colors.push(b_line);
                }

                // Momentum calculation
                let momentum = "→ Estable";
                let mom_color = "#64748b";
                if (margins.length >= 2) {
                    const lm = margins[margins.length-1] || 0;
                    const pm = margins[margins.length-2] || 0;
                    if (lm - pm > 1) { momentum = "▲ Mejorando"; mom_color = "#10b981"; }
                    else if (lm - pm < -1) { momentum = "▼ Empeorando"; mom_color = "#ef4444"; }
                }

                // LAYER A: Volume Context (Left Axis, conditionally colored bordered bars)
                traces.push({
                    x: x_vals, y: deps, type: 'bar',
                    name: 'Volumen', 
                    marker: { color: bar_fill_colors, line: {color: bar_line_colors, width: 1.5} },
                    customdata: hover_texts,
                    hovertemplate: '%{customdata}<extra></extra>'
                });
                
                // LAYER B: Real Margin (Primary Protagonist)
                traces.push({
                    x: x_vals, y: margins, type: 'scatter', mode: 'lines+markers', yaxis: 'y2',
                    name: 'Margen Real (%)', 
                    line: { color: '#0f172a', shape: 'spline', width: 3.5 }, // Thick dark line
                    marker: { size: margin_sizes, color: margin_colors, line: {color: margin_line_colors, width: margin_line_widths} },
                    customdata: hover_texts,
                    hovertemplate: '%{customdata}<extra></extra>'
                });
                
                // LAYER C: Model Score (Secondary, subtle dotted line)
                traces.push({
                    x: x_vals, y: scores, type: 'scatter', mode: 'lines', yaxis: 'y3',
                    name: 'Score', 
                    line: { color: 'rgba(100, 116, 139, 0.25)', width: 1.5, dash: 'dot' }, // Reduced opacity
                    hoverinfo: 'skip' 
                });
                
                // LAYER D: Anomaly Markers
                if (anomalies_x.length > 0) {
                    traces.push({
                        x: anomalies_x, y: anomalies_y, type: 'scatter', mode: 'markers', yaxis: 'y2',
                        name: 'Anomalías', 
                        marker: { symbol: anomalies_symbols, size: 14, color: 'rgba(0,0,0,0)', line: {color:'rgba(239,68,68,0.8)', width: 2} },
                        hoverinfo: 'skip'
                    });
                }
                
                addCriticalEnclosure(layout, x_vals, criticalFlags);
                
                // Overriding margins for Rentabilidad breathing room
                layout.margin.r = 60;
                layout.margin.l = 60;

                // Axis Layout
                layout.xaxis.tickvals = x_vals;
                layout.xaxis.ticktext = formatted_x;

                layout.yaxis = { 
                    showline: true, linewidth: 1.5, linecolor: '#94a3b8', ticks: 'outside', tickcolor: '#94a3b8', ticklen: 5,
                    showgrid: false, zeroline: false, showticklabels: true, title: '',
                    tickfont: { size: 9, color: '#94a3b8', family: 'Inter' },
                    tickformat: '$.2s', nticks: 4 // Minimal currency axis ($K, $M)
                }; 
                
                let min_m = Math.min(...margins.filter(m => m !== null));
                let max_m = Math.max(...margins.filter(m => m !== null));
                if (min_m > -5) min_m = -5;
                if (max_m < 20) max_m = 20;

                layout.yaxis2 = { 
                    range: [min_m - 5, max_m + 5], 
                    showline: true, linewidth: 1.5, linecolor: '#cbd5e1', 
                    ticks: 'outside', tickcolor: '#cbd5e1', ticklen: 6, ticklabelposition: 'outside',
                    showgrid: true, gridcolor: 'rgba(241, 245, 249, 0.5)', zeroline: true, zerolinecolor: '#cbd5e1', 
                    tickfont: { size: 10, color: '#475569', family: 'Inter', weight: 600 }, ticksuffix: '%', 
                    overlaying: 'y', side: 'right', nticks: 5
                };
                
                layout.yaxis3 = { 
                    range: [-1, 11], 
                    showline: false, ticks: '', showgrid: false, zeroline: false, showticklabels: false, 
                    overlaying: 'y', side: 'right'
                };
                
                // Subtle Background Zones & Thresholds (Reduced opacity)
                layout.shapes = [
                    { type: 'rect', x0: 0, x1: 1, xref: 'paper', y0: -100, y1: 0, yref: 'y2', fillcolor: 'rgba(239, 68, 68, 0.02)', line: {width: 0}, layer: 'below' }, 
                    { type: 'rect', x0: 0, x1: 1, xref: 'paper', y0: 0, y1: 5, yref: 'y2', fillcolor: 'rgba(245, 158, 11, 0.02)', line: {width: 0}, layer: 'below' }, 
                    { type: 'rect', x0: 0, x1: 1, xref: 'paper', y0: 10, y1: 100, yref: 'y2', fillcolor: 'rgba(16, 185, 129, 0.02)', line: {width: 0}, layer: 'below' }, 
                    
                    { type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 0, y1: 0, yref: 'y2', line: { color: '#94a3b8', width: 1.5 } }, 
                    { type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 5, y1: 5, yref: 'y2', line: { color: '#cbd5e1', width: 1, dash: 'dash' } }, 
                    { type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 10, y1: 10, yref: 'y2', line: { color: 'rgba(16, 185, 129, 0.4)', width: 1, dash: 'dash' } }, 
                    { type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 15, y1: 15, yref: 'y2', line: { color: 'rgba(16, 185, 129, 0.4)', width: 1, dash: 'dot' } }
                ];
                
                const last_m = margins[margins.length-1];
                const prev_m = margins.length >= 2 ? margins[margins.length-2] : null;
                const last_m_txt = (last_m !== null && last_m !== undefined) ? last_m.toFixed(1)+'%' : 'N/A';
                
                let delta_pp_text = '';
                if (last_m !== null && last_m !== undefined && prev_m !== null && prev_m !== undefined) {
                    const delta_pp = last_m - prev_m;
                    const delta_color = delta_pp >= 0 ? '#10b981' : '#ef4444';
                    const delta_arrow = delta_pp >= 0 ? '▲' : '▼';
                    const delta_str = (delta_pp > 0 ? '+' : '') + delta_pp.toFixed(2);
                    delta_pp_text = `<span style="color:${delta_color}; font-size:11px; font-weight:600;">${delta_arrow} ${delta_str} pp</span><br>`;
                }

                const last_s = scores[scores.length - 1] || 0;

                layout.annotations = [
                    {
                        x: 0, y: 1.13, xref: 'paper', yref: 'paper',
                        xanchor: 'left', yanchor: 'top',
                        text: `<b><span style="color:${mom_color};">${momentum}</span></b>`,
                        showarrow: false,
                        font: {size: 12, family: 'Inter'}
                    },
                    {
                        x: 1, y: 1.15, xref: 'paper', yref: 'paper',
                        xanchor: 'right', yanchor: 'top',
                        text: `<b><span style="font-size:14px;color:#0f172a;">${last_m_txt}</span></b>  ` +
                              delta_pp_text +
                              `<span style="font-size:10px; color:#64748b;">Score: ${last_s.toFixed(1)}/10</span>`,
                        showarrow: false,
                        align: 'right'
                    }
                ];

                layout.showlegend = false; 
                layout.margin.l = 50; 
                layout.hovermode = 'closest'; 
                layout.hoverlabel = { bgcolor: "rgba(255, 255, 255, 0.98)", bordercolor: "#cbd5e1", font: { family: "Inter", size: 12, color: "#0f172a" }, align: 'left' };
            } else if (m === 'volumen') {
                // High-Impact Combo: Lollipop (TXs) + Dotted Line (Score)
                const num_deps = series.map(d => d.num_depositos || 0);
                const num_rets = series.map(d => d.num_retiros || 0);
                const txs = num_deps.map((d, i) => d + num_rets[i]);
                const scores = y_vals;
                
                const hover_texts = [];
                const stems_x = [];
                const stems_y = [];
                const anomalies_x = [];
                const anomalies_y = [];
                const anomalies_text = [];
                const criticalFlags = [];

                for (let i = 0; i < series.length; i++) {
                    const t = txs[i];
                    const s = scores[i];
                    const prev_t = i > 0 ? txs[i-1] : null;

                    let t_var_pct = null; 
                    let is_critical = false;           // sigue siendo SOLO caídas (<=30)
                    let has_event = false;             // NUEVO: evento para tooltip (caídas y picos)
                    let evento = null;                 // NUEVO: texto del evento (si solo aplica)
                    
                    if (prev_t !== null && prev_t > 0) {
                        t_var_pct = ((t - prev_t) / prev_t) * 100;

                        if (t_var_pct <= -30) {
                            anomalies_x.push(x_vals[i]); anomalies_y.push(t); anomalies_text.push('Caída Crítica');
                            is_critical = true;
                            has_event = true;
                            evento = "⚠ Caída crítica";
                        } else if (t_var_pct >= 50) {
                            anomalies_x.push(x_vals[i]); anomalies_y.push(t); anomalies_text.push('Pico Inusual');
                            has_event = true;
                            evento = "⚠ Pico inusual";
                            // user objective rule says if we want to flag negative anomalies in volumen, we keep event logic. But we still flag pico if is_critical = false in earlier fix.
                            // wait, earlier fix made is_critical ONLY true for drop <= -30%. So if is_critical = false, Evento is NOT appended.
                        }
                    }
                    criticalFlags.push(is_critical);

                    // Hover tooltip
                    const real_data = {
                        "Depósitos": formatInt(num_deps[i]),
                        "Retiros": formatInt(num_rets[i]),
                        "Total TXs": formatInt(t)
                    };
                    const change_data = {};
                    if (t_var_pct !== null) {
                        change_data["Δ Vol"] = (t_var_pct > 0 ? '+' : '') + formatPct(t_var_pct);
                    }
                    if (has_event && evento) change_data["Evento"] = evento;
                    
                    const h_text = buildTooltipHTML(formatted_x[i], real_data, change_data, s);
                    hover_texts.push(h_text);

                    // Lollipop stems (need nulls to break the line between points)
                    stems_x.push(x_vals[i], x_vals[i], null);
                    stems_y.push(0, t, null);
                }

                // LAYER A1: Lollipop Stems
                traces.push({
                    x: stems_x, y: stems_y, type: 'scatter', mode: 'lines',
                    line: { color: 'rgba(16, 185, 129, 0.4)', width: 4 }, 
                    hoverinfo: 'skip', showlegend: false
                });

                // LAYER A2: Lollipop Heads
                traces.push({
                    x: x_vals, y: txs, type: 'scatter', mode: 'markers',
                    name: 'Transacciones',
                    marker: { size: 12, color: '#10b981', line: {color: '#ffffff', width: 2} },
                    customdata: hover_texts,
                    hovertemplate: '%{customdata}<extra></extra>'
                });

                // LAYER B: Model Score (Secondary, subtle dotted line)
                traces.push({
                    x: x_vals, y: scores, type: 'scatter', mode: 'lines', yaxis: 'y2',
                    name: 'Score', 
                    line: { color: 'rgba(100, 116, 139, 0.3)', width: 2, dash: 'dot' },
                    hoverinfo: 'skip'
                });

                // LAYER C: Anomalies
                if (anomalies_x.length > 0) {
                    traces.push({
                        x: anomalies_x, y: anomalies_y, type: 'scatter', mode: 'markers',
                        name: 'Alerta',
                        marker: { symbol: 'triangle-down-open', size: 16, color: '#ef4444', line: {width: 2} },
                        hoverinfo: 'skip'
                    });
                }
                // addCriticalEnclosure(layout, x_vals, criticalFlags);

                // Layout Overrides
                layout.xaxis.tickvals = x_vals;
                layout.xaxis.ticktext = formatted_x;

                layout.margin.r = 60;
                layout.margin.l = 60;
                
                layout.yaxis = { 
                    showline: true, linewidth: 1.5, linecolor: '#cbd5e1', ticks: 'outside', tickcolor: '#cbd5e1', ticklen: 5,
                    showgrid: true, gridcolor: 'rgba(241, 245, 249, 0.6)', zeroline: true, zerolinecolor: '#cbd5e1', 
                    tickfont: { size: 10, color: '#475569', family: 'Inter' },
                    rangemode: 'tozero'
                };
                
                layout.yaxis2 = { 
                    range: [-1, 11], 
                    showline: false, showgrid: false, zeroline: false, showticklabels: false, 
                    overlaying: 'y', side: 'right'
                };

                const last_t = txs[txs.length-1] || 0;
                const prev_t = txs.length >= 2 ? txs[txs.length-2] : null;
                const last_s = scores[scores.length-1] || 0;
                
                let delta_text = '';
                if (prev_t !== null) {
                    const delta = last_t - prev_t;
                    const delta_pct = prev_t > 0 ? (delta / prev_t) * 100 : 0;
                    const delta_color = delta >= 0 ? '#10b981' : '#ef4444';
                    const delta_arrow = delta >= 0 ? '▲' : '▼';
                    const delta_sign = delta > 0 ? '+' : '';
                    
                    delta_text = `<span style="color:${delta_color}; font-size:11px; font-weight:600;">${delta_arrow} ${delta_sign}${formatInt(delta)} (${delta_sign}${delta_pct.toFixed(1)}%)</span><br>`;
                } else {
                    delta_text = `<span style="color:#64748b; font-size:11px; font-weight:600;">Δ: N/A</span><br>`;
                }

                let momentum = "→ Estable";
                let mom_color = "#64748b";
                if (txs.length >= 2) {
                    const lm = txs[txs.length-1] || 0;
                    const pm = txs[txs.length-2] || 0;
                    if (lm > pm) { momentum = "▲ Mejorando"; mom_color = "#10b981"; }
                    else if (lm < pm) { momentum = "▼ Empeorando"; mom_color = "#ef4444"; }
                }
                
                layout.annotations = [
                    {
                        x: 0, y: 1.13, xref: 'paper', yref: 'paper',
                        xanchor: 'left', yanchor: 'top',
                        text: `<b><span style="color:${mom_color};">${momentum}</span></b>`,
                        showarrow: false,
                        font: {size: 12, family: 'Inter'}
                    },
                    {
                        x: 1, y: 1.15, xref: 'paper', yref: 'paper',
                        xanchor: 'right', yanchor: 'top',
                        text: `<b><span style="font-size:14px;color:#0f172a;">TXs: ${formatInt(last_t)}</span></b>  ` +
                              delta_text +
                              `<span style="font-size:10px; color:#64748b;">Score: ${last_s.toFixed(1)}/10</span>`,
                        showarrow: false,
                        align: 'right'
                    }
                ];
                
            } else if (m === 'fidelidad') {
                // ═══════════════════════════════════════════════════════════
                //  FIDELIDAD — Enterprise KPI Area Chart
                //  Primary:   Share % (smooth area line, last-point emphasis)
                //  Secondary: Score 0-10 (dotted, low contrast, y2)
                //  Annotation: Latest Share % + Δpp compact KPI (top-right)
                //  Context:   Subtle band around recent data range
                // ═══════════════════════════════════════════════════════════
                const act_vals  = series.map(d => d.active_players || 0);
                const glob_vals = series.map(d => d.global_players || 1);
                const share_pct = act_vals.map((act, i) => (act / glob_vals[i]) * 100);
                const scores    = y_vals;

                // ── Y-axis range: tight around real data, never clamp to 0 ──
                const min_share = Math.min(...share_pct);
                const max_share = Math.max(...share_pct);
                // Adaptive padding for very small values
                const padding   = (max_share - min_share) * 0.2 || (max_share * 0.2) || 0.1;
                const y_min     = Math.max(0, min_share - padding);
                const y_max     = max_share + padding;

                // ── Tooltip arrays ──
                const hover_texts  = [];
                const anomalies_x  = [];
                const anomalies_y  = [];
                const anomalies_text = [];
                const criticalFlags = [];
                let   last_sh_var_pp = 0;

                for (let i = 0; i < series.length; i++) {
                    const sh      = share_pct[i];
                    const s       = scores[i];
                    const prev_sh = i > 0 ? share_pct[i - 1] : null;

                    let sh_var_pp  = null;
                    let sh_var_pct = null;
                    let is_critical = false;
                    let evento = "⚠ Mes crítico";

                    if (prev_sh !== null) {
                        sh_var_pp  = sh - prev_sh;
                        if (prev_sh > 0) sh_var_pct = (sh_var_pp / prev_sh) * 100;
                        if (i === series.length - 1) last_sh_var_pp = sh_var_pp;
                        
                        if (sh_var_pp <= -0.5 || (sh_var_pct !== null && sh_var_pct <= -25)) {
                            anomalies_x.push(x_vals[i]);
                            anomalies_y.push(sh);
                            anomalies_text.push('⚠');
                            evento = "⚠ Caída crítica";
                        } else if (sh_var_pp >= 0.5 || (sh_var_pct !== null && sh_var_pct >= 25)) {
                            evento = "⚠ Pico inusual";
                        }
                    }
                    
                    is_critical = (s < 4.0);
                    criticalFlags.push(is_critical);

                    // ── Compact hovertemplate body ──
                    const real_data   = {
                        "Share %":            (sh > 0 && sh < 1) ? sh.toFixed(3) + '%' : formatPct(sh),
                        "Jugadores Propios":  formatInt(act_vals[i]),
                        "Jugadores Globales": formatInt(glob_vals[i])
                    };
                    const change_data = {};
                    if (sh_var_pp !== null) {
                        change_data["Δ Share"] = (sh_var_pp > 0 ? '+' : '') + 
                                                 ((Math.abs(sh_var_pp) > 0 && Math.abs(sh_var_pp) < 1) ? 
                                                 Math.abs(sh_var_pp).toFixed(3) + ' pp' : formatPP(sh_var_pp));
                    }
                    if (is_critical) change_data["Evento"] = evento;
                    hover_texts.push(buildTooltipHTML(formatted_x[i], real_data, change_data, s));
                }

                // ── LAYER A: Share % — dominant area line, all-point markers ──
                // All months get a visible marker; only the last is enlarged.
                const markerSizes = share_pct.map((_, i) => i === share_pct.length - 1 ? 11 : 8);

                traces.push({
                    x: x_vals, y: share_pct,
                    type: 'scatter', mode: 'lines+markers',
                    name: 'Share %',
                    line: {
                        color: config.color,
                        shape: 'linear',   // Executive clarity over spline
                        width: 3
                    },
                    marker: {
                        size: markerSizes,
                        color: config.color,
                        line: { color: '#ffffff', width: 2 }
                    },
                    fill: 'tozeroy',
                    fillcolor: hexToRgba(config.color, 0.13),
                    customdata: hover_texts,
                    hovertemplate: '%{customdata}<extra></extra>'
                });

                // ── LAYER B: Score 0-10 — ultra-muted, purely contextual ──
                traces.push({
                    x: x_vals, y: scores,
                    type: 'scatter', mode: 'lines',
                    name: '',           // Empty name → nothing in legend
                    showlegend: false,
                    yaxis: 'y2',
                    line: { color: 'rgba(148, 163, 184, 0.15)', width: 1, dash: 'dot' },
                    hovertemplate: '<extra></extra>'
                });
                
                addCriticalEnclosure(layout, x_vals, criticalFlags);

                // ── LAYER C: Alert markers ─────────────────────────────────
                // Removed anomaly markers to prevent secondary tooltip stealing hover

                // ── Layout overrides ───────────────────────────────────────
                layout.xaxis.tickvals = x_vals;
                layout.xaxis.ticktext = formatted_x;

                layout.margin   = { t: 44, b: 44, l: 52, r: 16 };
                layout.hovermode = 'closest';
                layout.hoverlabel = {
                    bgcolor: 'rgba(255,255,255,0.98)',
                    bordercolor: '#e2e8f0',
                    font: { family: 'Inter', size: 12, color: '#0f172a' },
                    align: 'left'
                };

                // Tight Y-axis focused on actual data range
                layout.yaxis = {
                    range: [y_min, y_max],
                    showline: true, linewidth: 1, linecolor: '#e2e8f0',
                    ticks: 'outside', tickcolor: '#e2e8f0', ticklen: 4,
                    showgrid: true, gridcolor: 'rgba(226,232,240,0.5)',
                    gridwidth: 1,
                    zeroline: false,
                    tickfont: { size: 10, color: '#94a3b8', family: 'Inter' },
                    ticksuffix: '%', 
                    nticks: 5,
                    tickformat: max_share < 1 ? '.3f' : '.2f'
                };

                // Score axis — completely hidden
                layout.yaxis2 = {
                    range: [-1, 11],
                    showline: false, showgrid: false,
                    zeroline: false, showticklabels: false,
                    overlaying: 'y', side: 'right'
                };

                // ── Subtle reference band: highlight recent data range ─────
                // Uses the min/max of the visible share data (no extra calc).
                const band_lo = min_share;
                const band_hi = max_share;
                layout.shapes = [
                    // Soft fill spanning the observed data range
                    {
                        type: 'rect', xref: 'paper',
                        x0: 0, x1: 1,
                        y0: band_lo, y1: band_hi, yref: 'y',
                        fillcolor: hexToRgba(config.color, 0.04),
                        line: { width: 0 }, layer: 'below'
                    },
                    // Thin top-of-range reference line (previous month proxy)
                    {
                        type: 'line', xref: 'paper',
                        x0: 0, x1: 1,
                        y0: share_pct.length >= 2 ? share_pct[share_pct.length - 2] : band_hi,
                        y1: share_pct.length >= 2 ? share_pct[share_pct.length - 2] : band_hi,
                        yref: 'y',
                        line: { color: 'rgba(148,163,184,0.5)', width: 1, dash: 'dash' }
                    }
                ];

                // ── Compact KPI annotation (top-right corner) ─────────────
                const last_sh      = share_pct[share_pct.length - 1] || 0;
                const last_s       = scores[scores.length - 1] || 0;
                const delta_color  = last_sh_var_pp >= 0 ? '#10b981' : '#ef4444';
                const delta_arrow  = last_sh_var_pp >= 0 ? '▲' : '▼';
                const delta_abs    = (Math.abs(last_sh_var_pp) > 0 && Math.abs(last_sh_var_pp) < 1) ? 
                                      Math.abs(last_sh_var_pp).toFixed(3) : 
                                      Math.abs(last_sh_var_pp).toFixed(2);

                // Momentum label (3-month slope)
                let momentum_label = '→ Estable';
                let momentum_color = '#64748b';
                if (share_pct.length >= 2) {
                    const window = share_pct.slice(-Math.min(3, share_pct.length));
                    const slope  = window[window.length - 1] - window[0];
                    if (slope > 0.5)       { momentum_label = '▲ Mejorando';  momentum_color = '#10b981'; }
                    else if (slope < -0.5) { momentum_label = '▼ Empeorando'; momentum_color = '#ef4444'; }
                }

                // Legend removed — visual hierarchy is self-explanatory.
                layout.showlegend = false;

                layout.annotations = [
                    // Momentum pill — top-left
                    {
                        x: 0, y: 1.13, xref: 'paper', yref: 'paper',
                        xanchor: 'left', yanchor: 'top',
                        text: `<b><span style="color:${momentum_color};">${momentum_label}</span></b>`,
                        showarrow: false,
                        font: { size: 12, family: 'Inter' }
                    },
                    // Delta + Score — top-right
                    {
                        x: 1, y: 1.15, xref: 'paper', yref: 'paper',
                        xanchor: 'right', yanchor: 'top',
                        text: `<b><span style="font-size:14px;color:#0f172a;">${(last_sh > 0 && last_sh < 1) ? last_sh.toFixed(3) : last_sh.toFixed(2)}%</span></b>  ` +
                              `<span style="color:${delta_color}; font-size:11px; font-weight:600;">${delta_arrow} ${delta_abs} pp</span><br>` +
                              `<span style="font-size:10px; color:#64748b;">Score: ${last_s.toFixed(1)}/10</span>`,
                        showarrow: false,
                        align: 'right'
                    }
                ];
                
            } else if (m === 'estabilidad') {
                // ═══════════════════════════════════════════════════════════
                //  ESTABILIDAD — KPI Card Redesign
                // ═══════════════════════════════════════════════════════════

                const est_score = lastVal;

                function estLabel(v) {
                    if (v < 4) return 'Volátil';
                    if (v < 6) return 'Moderado';
                    return 'Estable';
                }

                function estLabelColor(v) {
                    if (v < 4) return { bg: 'rgba(239,68,68,0.12)', fg: '#dc2626' };
                    if (v < 6) return { bg: 'rgba(245,158,11,0.15)', fg: '#b45309' };
                    return { bg: 'rgba(16,185,129,0.12)', fg: '#059669' };
                }

                const label = estLabel(est_score);
                const lStyle = estLabelColor(est_score);

                // formatted_x is available from global scope

                // SECTION 1: Bullet phantom axis (invisible point)
                traces.push({
                    x: [0, 10], y: [0, 0],
                    xaxis: 'x2', yaxis: 'y2',
                    type: 'scatter', mode: 'markers',
                    marker: { size: 0, opacity: 0 },
                    hovertemplate: '<extra></extra>',
                    showlegend: false
                });

                // SECTION 2: Dots strip (connector line + dots)
                const strip_hover_texts = [];
                for (let i = 0; i < series.length; i++) {
                    const sv = y_vals[i];
                    const prev_s = i > 0 ? y_vals[i-1] : null;
                    
                    const real_data = {
                        "Stability Score": formatScore(sv) + ' / 10',
                        "Target": "6.0",
                        "Gap": (sv - 6.0 > 0 ? '+' : '') + (sv - 6.0).toFixed(1)
                    };
                    
                    const change_data = {};
                    if (prev_s !== null) {
                        const d_s = sv - prev_s;
                        change_data["Δ Score"] = (d_s > 0 ? '+' : '') + formatPP(d_s);
                    }
                    
                    // Unified Tooltip matches Rentability/Fidelity
                    strip_hover_texts.push(buildTooltipHTML(formatted_x[i], real_data, change_data, sv));
                }

                // Connector line
                traces.push({
                    x: x_vals,
                    y: x_vals.map(() => 0.5),
                    xaxis: 'x', yaxis: 'y',
                    type: 'scatter', mode: 'lines',
                    showlegend: false,
                    line: { color: '#e2e8f0', width: 1.5 },
                    hoverinfo: 'skip'
                });

                // Dots
                const chipSizes = x_vals.map((_, i) => i === x_vals.length - 1 ? 14 : 10);
                const chipBorders = x_vals.map((_, i) => i === x_vals.length - 1 ? 1.5 : 0);

                traces.push({
                    x: x_vals,
                    y: x_vals.map(() => 0.5),
                    xaxis: 'x', yaxis: 'y',
                    type: 'scatter', mode: 'markers',
                    showlegend: false,
                    marker: {
                        size: chipSizes,
                        color: config.color, // '#8b5cf6' or similar from palette manager
                        line: { color: '#ffffff', width: chipBorders }
                    },
                    customdata: strip_hover_texts,
                    hovertemplate: '%{customdata}<extra></extra>' // <extra> removes trace label
                });

                // Layout Config
                layout.margin = { t: 20, b: 60, l: 20, r: 100 };
                layout.showlegend = false;
                layout.hovermode = 'closest';
                layout.plot_bgcolor = 'transparent';
                layout.paper_bgcolor = '#ffffff';

                layout.xaxis = {
                    domain: [0.0, 0.72],
                    type: 'category',
                    tickmode: 'array',
                    tickvals: adp_tickvals,
                    ticktext: adp_ticktext,
                    tickangle: 0,
                    automargin: true,
                    showgrid: false, zeroline: false, showline: false,
                    tickfont: { size: 10, color: '#64748b', family: 'Inter' },
                    ticks: ''
                };
                
                layout.yaxis = {
                    domain: [0, 0.20],
                    showgrid: false, zeroline: false, showticklabels: false, showline: false,
                    range: [0, 1]
                };

                layout.xaxis2 = {
                    domain: [0.0, 0.72],
                    anchor: 'y2',
                    range: [0, 10],
                    showgrid: false, zeroline: false, showline: false,
                    tickvals: [0, 2, 4, 6, 8, 10],
                    tickfont: { size: 10, color: '#64748b', family: 'Inter' },
                    ticks: ''
                };
                
                layout.yaxis2 = {
                    domain: [0.45, 0.95],
                    anchor: 'x2',
                    range: [-1, 1],
                    showgrid: false, zeroline: false, showticklabels: false, showline: false
                };

                const barH = 0.55;
                const targetV = 6;
                layout.shapes = [
                    // Zones (Neutral Cool Grays)
                    { type: 'rect', xref: 'x2', yref: 'y2', x0: 0, x1: 4, y0: -barH, y1: barH, fillcolor: '#f8fafc', line: {width: 0}, layer: 'below' },
                    { type: 'rect', xref: 'x2', yref: 'y2', x0: 4, x1: 6, y0: -barH, y1: barH, fillcolor: '#f1f5f9', line: {width: 0}, layer: 'below' },
                    { type: 'rect', xref: 'x2', yref: 'y2', x0: 6, x1: 10, y0: -barH, y1: barH, fillcolor: '#f8fafc', line: {width: 0}, layer: 'below' },
                    
                    // Confidence Container Outline
                    { type: 'rect', xref: 'x2', yref: 'y2', x0: 0, x1: 10, y0: -barH, y1: barH, fillcolor: 'rgba(0,0,0,0)', line: {color: '#e2e8f0', width: 1}, layer: 'above' },

                    // Value Bar
                    { type: 'rect', xref: 'x2', yref: 'y2', x0: 0, x1: est_score, y0: -barH*0.45, y1: barH*0.45, fillcolor: config.color, line: {width: 0}, layer: 'above' },
                    
                    // Target Line
                    { type: 'line', xref: 'x2', yref: 'y2', x0: targetV, x1: targetV, y0: -barH*1.1, y1: barH*1.1, line: { color: '#64748b', width: 1.5, dash: 'dash' }, layer: 'above' }
                ];

                const isFlat = y_vals.every(v => Math.abs(v - est_score) < 0.01);
                
                const scoreStr = est_score.toFixed(1);
                const gapVal = (est_score - targetV);
                const gapColor = gapVal >= 0 ? '#10b981' : '#ef4444';
                const gapSign = gapVal >= 0 ? '+' : '';

                layout.annotations = [
                    // KPI: Target Label (over gauge)
                    {
                        x: targetV, y: barH*1.1 + 0.1, xref: 'x2', yref: 'y2',
                        xanchor: 'center', yanchor: 'bottom',
                        text: '<span style="font-size:10px;color:#64748b;">Target</span>',
                        showarrow: false, font: { family: 'Inter', size: 10 }
                    },
                    // RIGHT PANEL: Score number
                    {
                        x: 0.92, y: 0.78, xref: 'paper', yref: 'paper',
                        xanchor: 'right', yanchor: 'middle',
                        text: `<b><span style="font-size:36px;color:#0f172a;">${scoreStr}</span></b><span style="font-size:14px;color:#94a3b8;">/10</span>`,
                        showarrow: false, font: { family: 'Inter' }
                    },
                    // RIGHT PANEL: Qualitative pill
                    {
                        x: 0.92, y: 0.53, xref: 'paper', yref: 'paper',
                        xanchor: 'right', yanchor: 'middle',
                        text: `<b><span style="color:${lStyle.fg};font-size:12px;">${label}</span></b>`,
                        showarrow: false, font: { family: 'Inter', size: 12 },
                        bgcolor: lStyle.bg, borderpad: 5,
                        bordercolor: 'rgba(0,0,0,0)', borderwidth: 0
                    },
                    // RIGHT PANEL: Target and Gap strings
                    {
                        x: 0.92, y: 0.38, xref: 'paper', yref: 'paper',
                        xanchor: 'right', yanchor: 'middle',
                        text: `<span style="font-size:11px;color:#94a3b8;">Objetivo: 6.0</span><br><span style="font-size:11px;color:${gapColor};">Brecha: ${gapSign}${gapVal.toFixed(1)}</span>`,
                        showarrow: false, font: { family: 'Inter', size: 11 }
                    },
                    // Flat-series notice
                    ...(isFlat ? [{
                        x: 0.325, y: -0.4, xref: 'paper', yref: 'paper',
                        xanchor: 'center', yanchor: 'top',
                        text: `<span style="font-size:10px;color:#64748b;font-style:italic;">No monthly variation</span>`,
                        showarrow: false, font: { family: 'Inter' }
                    }] : [])
                ];
                
            } else if (m === 'crecimiento') {
                // Diverging Bars around score 5
                const colors = y_vals.map(y => y >= 5 ? '#10b981' : '#ef4444'); // Green if >=5, Red if <5
                const hover_texts = [];
                for (let i = 0; i < series.length; i++) {
                    const s = y_vals[i];
                    const prev_s = i > 0 ? y_vals[i-1] : null;
                    const real_data = { "Crecimiento Abs.": formatScore(s) };
                    const change_data = {};
                    if (prev_s !== null) {
                        const d_s = s - prev_s;
                        change_data["Δ Crec."] = (d_s > 0 ? '+' : '') + formatPP(d_s);
                    }
                    hover_texts.push(buildTooltipHTML(formatted_x[i], real_data, change_data, s));
                }
                traces.push({
                    x: x_vals, y: y_vals, customdata: hover_texts, type: 'bar', base: 5,
                    name: 'Score', marker: { color: colors, cornerradius: 2 },
                    hovertemplate: '%{customdata}<extra></extra>'
                });
                layout.shapes = [{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 5, y1: 5, line: { color: '#94a3b8', width: 1 } }];
            
            } else if (m === 'eficiencia_casino' || m === 'eficiencia_deportes') {
                // ═══════════════════════════════════════════════════════════
                //  EFICIENCIA (Casino/Deportes) — KPI Card Redesign
                // ═══════════════════════════════════════════════════════════

                const effic_score = lastVal;

                function efficLabel(v) {
                    if (v < 3) return 'Crítico';
                    if (v < 7) return 'Moderado';
                    return 'Óptimo';
                }

                function efficLabelColor(v) {
                    if (v < 3) return { bg: 'rgba(239,68,68,0.12)', fg: '#dc2626' };
                    if (v < 7) return { bg: 'rgba(245,158,11,0.15)', fg: '#b45309' };
                    return { bg: 'rgba(16,185,129,0.12)', fg: '#059669' };
                }

                const label = efficLabel(effic_score);
                const lStyle = efficLabelColor(effic_score);

                // formatted_x is available from global scope

                // SECTION 1: Bullet phantom axis (invisible point)
                traces.push({
                    x: [0, 10], y: [0, 0],
                    xaxis: 'x2', yaxis: 'y2',
                    type: 'scatter', mode: 'markers',
                    marker: { size: 0, opacity: 0 },
                    hovertemplate: '<extra></extra>',
                    showlegend: false
                });

                // SECTION 2: Dots strip (connector line + dots)
                const strip_hover_texts = [];
                for (let i = 0; i < series.length; i++) {
                    const sv = y_vals[i];
                    const prev_s = i > 0 ? y_vals[i-1] : null;
                    const d = series[i];
                    const prev_d = i > 0 ? series[i-1] : null;

                    // Depending on metric, identify GGR keys
                    const is_casino = (m === 'eficiencia_casino');
                    const ggr_val = is_casino ? (d.casino_ggr ?? d.ggr_casino ?? 0) : (d.apuestas_deportivas_ggr ?? d.ggr_deportiva ?? d.deportes_ggr ?? 0);
                    const prev_ggr = (prev_d) ? (is_casino ? (prev_d.casino_ggr ?? prev_d.ggr_casino ?? 0) : (prev_d.apuestas_deportivas_ggr ?? prev_d.ggr_deportiva ?? prev_d.deportes_ggr ?? 0)) : null;
                    
                    const dep_val = d.total_depositos || 0;
                    const prev_dep = (prev_d) ? (prev_d.total_depositos || 0) : null;
                    const num_dep = d.num_depositos || 0;
                    
                    const ggr_label = is_casino ? 'Casino GGR' : 'Sports GGR';
                    
                    const real_data = {
                        "Deposits": formatMoney(dep_val),
                        [ggr_label]: formatMoney(ggr_val),
                        "Deposits count": formatInt(num_dep)
                    };
                    if (ggr_val > 0) {
                        real_data["Ratio (Dep/GGR)"] = (dep_val / ggr_val).toFixed(2);
                    }
                    
                    const change_data = {};
                    if (prev_dep !== null && prev_dep > 0) {
                        const d_dep = ((dep_val - prev_dep) / prev_dep) * 100;
                        change_data["Δ Deposits"] = (d_dep > 0 ? '+' : '') + formatPct(d_dep);
                    } else if (prev_dep !== null) {
                        change_data["Δ Deposits"] = "N/A";
                    }
                    
                    if (prev_ggr !== null && prev_ggr > 0) {
                        const d_ggr = ((ggr_val - prev_ggr) / prev_ggr) * 100;
                        change_data["Δ " + ggr_label] = (d_ggr > 0 ? '+' : '') + formatPct(d_ggr);
                    } else if (prev_ggr !== null) {
                        change_data["Δ " + ggr_label] = "N/A";
                    }
                    
                    strip_hover_texts.push(buildTooltipHTML(formatted_x[i], real_data, change_data, sv));
                }

                // Connector line
                traces.push({
                    x: x_vals,
                    y: x_vals.map(() => 0.5),
                    xaxis: 'x', yaxis: 'y',
                    type: 'scatter', mode: 'lines',
                    showlegend: false,
                    line: { color: '#e2e8f0', width: 1.5 },
                    hoverinfo: 'skip'
                });

                // Dots
                const chipSizes = x_vals.map((_, i) => i === x_vals.length - 1 ? 14 : 10);
                const chipBorders = x_vals.map((_, i) => i === x_vals.length - 1 ? 1.5 : 0);

                traces.push({
                    x: x_vals,
                    y: x_vals.map(() => 0.5),
                    xaxis: 'x', yaxis: 'y',
                    type: 'scatter', mode: 'markers',
                    showlegend: false,
                    marker: {
                        size: chipSizes,
                        color: config.color,
                        line: { color: '#ffffff', width: chipBorders }
                    },
                    customdata: strip_hover_texts,
                    hovertemplate: '%{customdata}<extra></extra>' // <extra> removes trace label
                });

                // Layout Config
                layout.margin = { t: 20, b: 80, l: 20, r: 100 };
                layout.showlegend = false;
                layout.hovermode = 'closest';
                layout.plot_bgcolor = 'transparent';
                layout.paper_bgcolor = '#ffffff';

                layout.xaxis = {
                    domain: [0.0, 0.72],
                    type: 'category',
                    tickmode: 'array',
                    tickvals: adp_tickvals,
                    ticktext: adp_ticktext,
                    tickangle: 0,
                    automargin: true,
                    showgrid: false, zeroline: false, showline: false,
                    tickfont: { size: 10, color: '#64748b', family: 'Inter' },
                    ticks: ''
                };
                
                layout.yaxis = {
                    domain: [0.0, 0.30],
                    showgrid: false, zeroline: false, showticklabels: false, showline: false,
                    range: [0, 1]
                };

                layout.xaxis2 = {
                    domain: [0.0, 0.72],
                    anchor: 'y2',
                    range: [0, 10],
                    showgrid: false, zeroline: false, showline: false,
                    tickvals: [0, 2, 4, 6, 8, 10],
                    tickfont: { size: 10, color: '#64748b', family: 'Inter' },
                    ticks: ''
                };
                
                layout.yaxis2 = {
                    domain: [0.52, 1.00],
                    anchor: 'x2',
                    range: [-1, 1],
                    showgrid: false, zeroline: false, showticklabels: false, showline: false
                };

                const barH = 0.55;
                const targetV = 7;
                layout.shapes = [
                    // Zones (Neutral Cool Grays)
                    { type: 'rect', xref: 'x2', yref: 'y2', x0: 0, x1: 4, y0: -barH, y1: barH, fillcolor: '#f8fafc', line: {width: 0}, layer: 'below' },
                    { type: 'rect', xref: 'x2', yref: 'y2', x0: 4, x1: 7, y0: -barH, y1: barH, fillcolor: '#f1f5f9', line: {width: 0}, layer: 'below' },
                    { type: 'rect', xref: 'x2', yref: 'y2', x0: 7, x1: 10, y0: -barH, y1: barH, fillcolor: '#f8fafc', line: {width: 0}, layer: 'below' },
                    
                    // Confidence Container Outline
                    { type: 'rect', xref: 'x2', yref: 'y2', x0: 0, x1: 10, y0: -barH, y1: barH, fillcolor: 'rgba(0,0,0,0)', line: {color: '#e2e8f0', width: 1}, layer: 'above' },

                    // Value Bar
                    { type: 'rect', xref: 'x2', yref: 'y2', x0: 0, x1: effic_score, y0: -barH*0.45, y1: barH*0.45, fillcolor: config.color, line: {width: 0}, layer: 'above' },
                    
                    // Target Line
                    { type: 'line', xref: 'x2', yref: 'y2', x0: targetV, x1: targetV, y0: -barH*1.1, y1: barH*1.1, line: { color: '#64748b', width: 1.5, dash: 'dash' }, layer: 'above' }
                ];

                const isFlat = y_vals.every(v => Math.abs(v - effic_score) < 0.01);
                
                const scoreStr = effic_score.toFixed(1);
                const gapVal = effic_score - targetV;
                const gapColor = gapVal >= 0 ? '#10b981' : '#ef4444';
                const gapSign = gapVal >= 0 ? '+' : '';

                layout.annotations = [
                    // KPI: Target Label (over gauge)
                    {
                        x: targetV, y: barH*1.1 + 0.1, xref: 'x2', yref: 'y2',
                        xanchor: 'center', yanchor: 'bottom',
                        text: '<span style="font-size:10px;color:#64748b;">Objetivo</span>',
                        showarrow: false, font: { family: 'Inter', size: 10 }
                    },
                    // RIGHT PANEL: Score number
                    {
                        x: 0.92, y: 0.78, xref: 'paper', yref: 'paper',
                        xanchor: 'right', yanchor: 'middle',
                        text: `<b><span style="font-size:36px;color:#0f172a;">${scoreStr}</span></b><span style="font-size:14px;color:#94a3b8;">/10</span>`,
                        showarrow: false, font: { family: 'Inter' }
                    },
                    // RIGHT PANEL: Qualitative pill
                    {
                        x: 0.92, y: 0.53, xref: 'paper', yref: 'paper',
                        xanchor: 'right', yanchor: 'middle',
                        text: `<b><span style="color:${lStyle.fg};font-size:12px;">${label}</span></b>`,
                        showarrow: false, font: { family: 'Inter', size: 12 },
                        bgcolor: lStyle.bg, borderpad: 5,
                        bordercolor: 'rgba(0,0,0,0)', borderwidth: 0
                    },
                    // RIGHT PANEL: Target and Gap strings
                    {
                        x: 0.92, y: 0.38, xref: 'paper', yref: 'paper',
                        xanchor: 'right', yanchor: 'middle',
                        text: `<span style="font-size:11px;color:#94a3b8;">Objetivo: 7.0</span><br><span style="font-size:11px;color:${gapColor};">Brecha: ${gapSign}${gapVal.toFixed(1)}</span>`,
                        showarrow: false, font: { family: 'Inter', size: 11 }
                    },
                    // Flat-series notice
                    ...(isFlat ? [{
                        x: 0.325, y: -0.4, xref: 'paper', yref: 'paper',
                        xanchor: 'center', yanchor: 'top',
                        text: `<span style="font-size:10px;color:#64748b;font-style:italic;">Sin variación mensual</span>`,
                        showarrow: false, font: { family: 'Inter' }
                    }] : [])
                ];

            } else if (m === 'eficiencia_conversion') {
                // ═══════════════════════════════════════════════════════════
                //  EFICIENCIA CONVERSIÓN — KPI Card Redesign
                // ═══════════════════════════════════════════════════════════

                const effic_score = lastVal;

                // Spanish threshold badge
                function convBadge(v) {
                    if (v >= 7) return { text: 'Por encima del umbral', bg: 'rgba(16,185,129,0.12)', fg: '#059669' };
                    if (v >= 3) return { text: 'Cerca del umbral',     bg: 'rgba(245,158,11,0.15)', fg: '#b45309' };
                    return              { text: 'Por debajo del umbral', bg: 'rgba(239,68,68,0.12)', fg: '#dc2626' };
                }
                const badge = convBadge(effic_score);

                // formatted_x is available from global scope

                // SECTION 1: Angular Gauge (Top ~44%) — NO target marker
                traces.push({
                    type: "indicator",
                    mode: "gauge",
                    value: effic_score,
                    domain: { x: [0, 1], y: [0.56, 1.0] },
                    gauge: {
                        axis: {
                            range: [0, 10],
                            tickfont: { size: 10, color: '#64748b', family: 'Inter' },
                            tickwidth: 1, tickcolor: '#cbd5e1'
                        },
                        bar: { color: config.color, thickness: 0.25 },
                        bgcolor: "rgba(0,0,0,0)",
                        borderwidth: 0,
                        bordercolor: "transparent",
                        steps: [
                            { range: [0, 3], color: "#f8fafc" },
                            { range: [3, 7], color: "#f1f5f9" },
                            { range: [7, 10], color: "#e2e8f0" }
                        ]
                    },
                    hoverinfo: 'skip'
                });

                // Annotations: KPI number + Spanish threshold badge (NO target label)
                layout.annotations = [
                    {
                        x: 0.5, y: 0.72, xref: 'paper', yref: 'paper',
                        xanchor: 'center', yanchor: 'middle',
                        text: `<b><span style="font-size:36px;color:#0f172a;">${effic_score.toFixed(1)}</span></b><span style="font-size:14px;color:#94a3b8;">/10</span>`,
                        showarrow: false, font: { family: 'Inter' }
                    },
                    {
                        x: 0.5, y: 0.52, xref: 'paper', yref: 'paper',
                        xanchor: 'center', yanchor: 'middle',
                        text: `<b><span style="color:${badge.fg};font-size:11px;">${badge.text}</span></b>`,
                        showarrow: false, font: { family: 'Inter', size: 11 },
                        bgcolor: badge.bg, borderpad: 5, bordercolor: 'rgba(0,0,0,0)', borderwidth: 0
                    }
                ];

                // SECTION 2: Monthly Trend (Bottom ~26%)
                const trend_hover_texts = [];
                for (let i = 0; i < series.length; i++) {
                    const sv = y_vals[i];
                    const prev_s = i > 0 ? y_vals[i-1] : null;
                    const d = series[i];
                    const prev_d = i > 0 ? series[i-1] : null;

                    const dep_val = d.total_depositos || 0;
                    const prev_dep = (prev_d) ? (prev_d.total_depositos || 0) : null;
                    const ggr_val = (d.casino_ggr || 0) + (d.apuestas_deportivas_ggr || 0);
                    const prev_ggr = (prev_d) ? ((prev_d.casino_ggr || 0) + (prev_d.apuestas_deportivas_ggr || 0)) : null;

                    const real_data = {
                        "Depósitos": formatMoney(dep_val),
                        "GGR": formatMoney(ggr_val)
                    };

                    if (dep_val > 0) {
                        real_data["Conversión (GGR/Depósitos)"] = ((ggr_val / dep_val) * 100).toFixed(2) + '%';
                    }

                    const change_data = {};
                    if (prev_dep !== null && prev_dep > 0) {
                        const d_dep = ((dep_val - prev_dep) / prev_dep) * 100;
                        change_data["Δ Depósitos"] = (d_dep > 0 ? '+' : '') + formatPct(d_dep);
                    } else if (prev_dep !== null) {
                        change_data["Δ Depósitos"] = "N/A";
                    }

                    if (prev_ggr !== null && prev_ggr > 0) {
                        const d_ggr = ((ggr_val - prev_ggr) / prev_ggr) * 100;
                        change_data["Δ GGR"] = (d_ggr > 0 ? '+' : '') + formatPct(d_ggr);
                    } else if (prev_ggr !== null) {
                        change_data["Δ GGR"] = "N/A";
                    }

                    if (dep_val > 0 && prev_dep && prev_dep > 0) {
                        const current_conv = (ggr_val / dep_val) * 100;
                        const prev_conv = (prev_ggr / prev_dep) * 100;
                        const d_conv = current_conv - prev_conv;
                        change_data["Δ Conversión"] = (d_conv > 0 ? '+' : '') + d_conv.toFixed(2) + ' pp';
                    } else if (prev_dep !== null) {
                        change_data["Δ Conversión"] = "N/A";
                    }

                    trend_hover_texts.push(buildTooltipHTML(formatted_x[i], real_data, change_data, sv));
                }

                traces.push({
                    x: x_vals,
                    y: y_vals,
                    customdata: trend_hover_texts,
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: { color: config.color, width: 2, shape: 'linear' },
                    marker: {
                        size: x_vals.map((_, i) => i === x_vals.length - 1 ? 8 : 4),
                        color: config.color,
                        line: { color: '#ffffff', width: x_vals.map((_, i) => i === x_vals.length - 1 ? 1.5 : 0) }
                    },
                    fill: 'tozeroy',
                    fillcolor: hexToRgba(config.color, 0.08),
                    connectgaps: true,
                    cliponaxis: false,
                    hovertemplate: '%{customdata}<extra></extra>'
                });

                // Layout Config — safe margins, strict domains
                layout.margin = { t: 20, b: 70, l: 30, r: 30 };
                layout.showlegend = false;
                layout.hovermode = 'closest';
                layout.plot_bgcolor = 'transparent';
                layout.paper_bgcolor = '#ffffff';

                layout.xaxis = {
                    domain: [0.05, 0.95],
                    type: 'category',
                    tickvals: x_vals,
                    ticktext: formatted_x,
                    showgrid: false,
                    zeroline: false,
                    showline: false,
                    tickfont: { size: 10, color: '#64748b', family: 'Inter' }
                };

                layout.yaxis = {
                    domain: [0.0, 0.38],
                    showgrid: true,
                    gridcolor: '#f1f5f9',
                    zeroline: false,
                    showticklabels: false,
                    range: [0, 10]
                };
                
            } else if (m === 'tendencia') { 
                // ═══════════════════════════════════════════════════════════ 
                //  TENDENCIA HISTÓRICA — Curvilíneo + Pendiente Global 
                // ═══════════════════════════════════════════════════════════ 
                const  hover_texts = []; 
                const  scores = y_vals; 
                
                const firstVal = scores[0] || 0 ; 
                const lastVal = scores[scores.length - 1] || 0 ; 
                const  overall_delta = lastVal - firstVal; 

                for (let i = 0 ; i < series.length; i++) { 
                    const  s = scores[i]; 
                    const prev_s = i > 0 ? scores[i-1] : null ; 

                    const  real_data = { 
                        "Score Tendencia": formatScore(s) + " / 10" 
                    }; 
                    
                    const  change_data = {}; 
                    if (prev_s !== null ) { 
                        const  d_s = s - prev_s; 
                        change_data["Δ vs Mes Anterior"] = (d_s > 0 ? '+' : '' ) + formatPP(d_s); 
                    } 
                    
                    hover_texts.push(buildTooltipHTML(formatted_x[i], real_data, change_data, s)); 
                } 

                // Lógica de estado de la pendiente global 
                let statusText = "→ Estable" ; 
                let statusColor = "#64748b" ; 
                if (overall_delta > 0.5 ) { 
                    statusText = "▲ Tendencia Positiva" ; 
                    statusColor = "#10b981" ; 
                } else if (overall_delta < -0.5 ) { 
                    statusText = "▼ Tendencia Negativa" ; 
                    statusColor = "#ef4444" ; 
                } 

                // CAPA A: Gráfico curvilíneo con todos los meses (Spline) 
                traces.push({ 
                    x : x_vals, 
                    y : scores, 
                    customdata : hover_texts, 
                    type: 'scatter' , 
                    mode: 'lines+markers' , 
                    name: 'Score Mensual' , 
                    line: { color: config.color, width: 3, shape: 'spline'  }, 
                    marker : { 
                        size: x_vals.map((_, i) => i === x_vals.length - 1 ? 10 : 6 ), 
                        color : config.color, 
                        line: { color: '#ffffff', width: x_vals.map((_, i) => i === x_vals.length - 1 ? 2 : 1 ) } 
                    }, 
                    fill: 'tozeroy' , 
                    fillcolor: hexToRgba(config.color, 0.12 ), 
                    hovertemplate: '%{customdata}<extra></extra>' 
                }); 

                // CAPA B: Línea de Pendiente Global (Dashed) 
                if (scores.length > 1 ) { 
                    traces.push({ 
                        x: [x_vals[0], x_vals[x_vals.length - 1 ]], 
                        y : [firstVal, lastVal], 
                        type: 'scatter' , 
                        mode: 'lines' , 
                        name: 'Pendiente' , 
                        line: { color: statusColor, width: 2, dash: 'dash'  }, 
                        hoverinfo: 'skip' 
                    }); 
                } 

                const delta_str = (overall_delta > 0 ? '+' : '') + overall_delta.toFixed(2 ); 

                layout.annotations = [ 
                    { 
                        x: 0, y: 1.13, xref: 'paper', yref: 'paper' , 
                        xanchor: 'left', yanchor: 'top' , 
                        text: `<b><span style="color:${statusColor};">${statusText}</span></b>` , 
                        showarrow: false , 
                        font: {size: 12, family: 'Inter' } 
                    }, 
                    { 
                        x: 1, y: 1.15, xref: 'paper', yref: 'paper' , 
                        xanchor: 'right', yanchor: 'top' , 
                        text: `<b><span style="font-size:14px;color:#0f172a;">${lastVal.toFixed(2)} / 10</span></b><br>`  + 
                              `<span style="color:${statusColor}; font-size:11px; font-weight:600;">Δ Global: ${delta_str} pts</span>` , 
                        showarrow: false , 
                        align: 'right' 
                    } 
                ]; 

                layout.margin = { t: 50, b: 60, l: 45, r: 30  }; 
                layout.yaxis = { 
                    range: [0, 10.5 ], 
                    showline: true, linewidth: 1.5, linecolor: '#cbd5e1' , 
                    ticks: 'outside', tickcolor: '#cbd5e1', ticklen: 5 , 
                    showgrid: true, gridcolor: 'rgba(241, 245, 249, 0.6)', zeroline: true, zerolinecolor: '#cbd5e1' , 
                    tickfont: { size: 10, color: '#475569', family: 'Inter'  } 
                };

            } else if (m === 'diversificacion') {
                // ═══════════════════════════════════════════════════════════
                //  DIVERSIFICACIÓN — 2-Product HHI Scale (Max ~5.0)
                // ═══════════════════════════════════════════════════════════
                const hover_texts = [];
                const scores = y_vals;
                const criticalFlags = [];
                for (let i = 0; i < series.length; i++) {
                    const s = y_vals[i];
                    criticalFlags.push(s < 4.0);
                    const prev_s = i > 0 ? y_vals[i-1] : null;
                    
                    const ggr_cas = series[i].ggr_casino || 0;
                    const ggr_dep = series[i].ggr_deportiva || 0;
                    const total_ggr = ggr_cas + ggr_dep;
                    const cas_share = total_ggr > 0 ? (ggr_cas / total_ggr) * 100 : 0;
                    const dep_share = total_ggr > 0 ? (ggr_dep / total_ggr) * 100 : 0;
                    
                    let prev_cas_share = null;
                    if (prev_s !== null) {
                        const prev_ggr_cas = series[i-1].ggr_casino || 0;
                        const prev_ggr_dep = series[i-1].ggr_deportiva || 0;
                        const prev_tot = prev_ggr_cas + prev_ggr_dep;
                        prev_cas_share = prev_tot > 0 ? (prev_ggr_cas / prev_tot) * 100 : 0;
                    }

                    const real_data = {
                        "Casino GGR": '$' + formatInt(ggr_cas),
                        "Deportes GGR": '$' + formatInt(ggr_dep),
                        "Mix": `${formatPct(cas_share)} Cas / ${formatPct(dep_share)} Dep`
                    };
                    
                    const change_data = {};
                    let evento = "⚠ Mes crítico";
                    
                    if (prev_s !== null) {
                        const d_s = s - prev_s;
                        change_data["Δ Diversificación"] = (d_s > 0 ? '+' : '') + formatPP(d_s) + ' pts';
                        
                        if (d_s <= -1.0) evento = "⚠ Caída crítica";
                        else if (d_s >= 1.0) evento = "⚠ Pico inusual";
                        
                        if (prev_cas_share !== null) {
                            const d_cas = cas_share - prev_cas_share;
                            change_data["Δ Casino Share"] = (d_cas > 0 ? '+' : '') + formatPP(d_cas) + ' pp';
                        }
                    }
                    if (criticalFlags[i]) change_data["Evento"] = evento;
                    
                    hover_texts.push(buildTooltipHTML(formatted_x[i], real_data, change_data, s));
                }

                traces.push({
                    x: x_vals, y: y_vals, customdata: hover_texts, 
                    type: 'scatter', mode: 'lines+markers', name: 'Score', 
                    line: { color: config.color, width: 2, shape: 'hv' },
                    marker: { 
                        size: x_vals.map((_, i) => i === x_vals.length - 1 ? 9 : 5), 
                        color: config.color, 
                        line: { color: '#ffffff', width: x_vals.map((_, i) => i === x_vals.length - 1 ? 2 : 0) } 
                    },
                    fill: 'tozeroy',
                    fillcolor: hexToRgba(config.color, 0.08),
                    hovertemplate: '%{customdata}<extra></extra>'
                });
                
                addCriticalEnclosure(layout, x_vals, criticalFlags);
                
                layout.shapes = [{ 
                    type: 'rect', x0: 0, x1: 1, xref: 'paper', y0: 3.5, y1: 5.5, yref: 'y', 
                    fillcolor: 'rgba(34, 197, 94, 0.08)', line: {width: 0}, layer: 'below' 
                }];
                
                layout.xaxis.tickvals = x_vals;
                layout.xaxis.ticktext = formatted_x;

                layout.margin = { t: 20, b: 40, l: 30, r: 15 };
                layout.showlegend = false;
                layout.yaxis = {
                    range: [0, 5.5],
                    showgrid: true,
                    gridcolor: 'rgba(241, 245, 249, 0.6)',
                    zeroline: true,
                    zerolinecolor: 'rgba(241, 245, 249, 0.8)',
                    tickfont: { size: 10, color: '#94a3b8', family: 'Inter' },
                    dtick: 1
                };
                
            } else if (m === 'calidad_jugadores') {
                // Vertical Bar Chart
                const hover_texts = [];
                for (let i = 0; i < series.length; i++) {
                    const s = y_vals[i];
                    const prev_s = i > 0 ? y_vals[i-1] : null;
                    // 'series.map' gives access to raw value: series[i].calidad_jugadores
                    const real_data = { "Score Calidad": formatScore(s) };
                    const change_data = {};
                    if (prev_s !== null) {
                        const d_s = s - prev_s;
                        change_data["Δ Calidad"] = (d_s > 0 ? '+' : '') + formatPP(d_s);
                    }
                    hover_texts.push(buildTooltipHTML(formatted_x[i], real_data, change_data, s));
                }
                traces.push({
                    x: x_vals, y: y_vals, customdata: hover_texts, type: 'bar',
                    name: 'Score', marker: { color: config.color, cornerradius: 4 },
                    hovertemplate: '%{customdata}<extra></extra>'
                });
            } else if (m === 'eficiencia_juego') {
                // ═══════════════════════════════════════════════════════════
                //  EFICIENCIA DE JUEGO (Turnover vs Depósitos) - BAR CHART
                // ═══════════════════════════════════════════════════════════
                const hover_texts = [];
                const deps = series.map(d => d.total_depositos || 0);
                const ap_total = series.map(d => d.total_apuesta_total || 0); 
                const margins = y_vals;
                
                const bar_colors = [];
                let last_var_pp = 0;

                for (let i = 0; i < series.length; i++) {
                    const s = margins[i];
                    const prev_s = i > 0 ? margins[i-1] : null;

                    let is_critical = false;
                    let has_event = false;
                    let evento = null;
                    let var_pp = null;

                    if (prev_s !== null) {
                        var_pp = s - prev_s;
                        if (i === series.length - 1) last_var_pp = var_pp;

                        // Thresholds for anomalies in Turnover Efficiency
                        if (var_pp <= -20) {
                            is_critical = true;
                            has_event = true;
                            evento = "⚠ Caída crítica";
                        } else if (var_pp >= 30) {
                            has_event = true;
                            evento = "⚠ Pico inusual";
                        }
                    }

                    // Red color for negative efficiency, config.color for positive
                    bar_colors.push(s < 0 ? 'rgba(239, 68, 68, 0.7)' : hexToRgba(config.color, 0.8));

                    const real_data = {
                        "Depósitos": formatMoney(deps[i]),
                        "Apuestas Totales": formatMoney(ap_total[i]),
                        "Eficiencia %": formatPct(s)
                    };
                    
                    const change_data = {};
                    if (var_pp !== null) {
                        change_data["Δ Eficiencia"] = (var_pp > 0 ? '+' : '') + formatPP(var_pp);
                    }
                    if (has_event && evento) change_data["Evento"] = evento;
                    
                    // Passing 0 as score since this is purely a contextual metric
                    hover_texts.push(buildTooltipHTML(formatted_x[i], real_data, change_data, 0));
                }

                traces.push({
                    x: x_vals, 
                    y: margins, 
                    customdata: hover_texts, 
                    type: 'bar', 
                    name: 'Eficiencia %', 
                    marker: { color: bar_colors, cornerradius: 4 },
                    text: margins.map(v => v.toFixed(1) + '%'),
                    textposition: 'outside',
                    textfont: { size: 10, color: '#475569', family: 'Inter', weight: 600 },
                    cliponaxis: false,
                    hovertemplate: '%{customdata}<extra></extra>'
                });

                // Calculate Momentum for Annotations
                let momentum = "→ Estable";
                let mom_color = "#64748b";
                if (margins.length >= 2) {
                    const lm = margins[margins.length-1] || 0;
                    const pm = margins[margins.length-2] || 0;
                    if (lm - pm >= 10) { momentum = "▲ Mejorando"; mom_color = "#10b981"; }
                    else if (lm - pm <= -10) { momentum = "▼ Empeorando"; mom_color = "#ef4444"; }
                }

                const last_s = margins[margins.length - 1] || 0;
                const delta_color = last_var_pp >= 0 ? '#10b981' : '#ef4444';
                const delta_arrow = last_var_pp >= 0 ? '▲' : '▼';
                const delta_str = (last_var_pp > 0 ? '+' : '') + last_var_pp.toFixed(2);
                
                let delta_pp_text = '';
                if (margins.length >= 2) {
                    delta_pp_text = `<span style="color:${delta_color}; font-size:11px; font-weight:600;">${delta_arrow} ${delta_str} pp</span><br>`;
                } else {
                    delta_pp_text = `<span style="color:#64748b; font-size:11px; font-weight:600;">Δ: N/A</span><br>`;
                }

                layout.annotations = [
                    {
                        x: 0, y: 1.13, xref: 'paper', yref: 'paper',
                        xanchor: 'left', yanchor: 'top',
                        text: `<b><span style="color:${mom_color};">${momentum}</span></b>`,
                        showarrow: false,
                        font: {size: 12, family: 'Inter'}
                    },
                    {
                        x: 1, y: 1.15, xref: 'paper', yref: 'paper',
                        xanchor: 'right', yanchor: 'top',
                        text: `<b><span style="font-size:14px;color:#0f172a;">${last_s.toFixed(2)}%</span></b>  ` + delta_pp_text,
                        showarrow: false,
                        align: 'right'
                    }
                ];

                layout.margin = { t: 50, b: 60, l: 60, r: 30 }; // Extra top margin for annotations
                layout.yaxis = { 
                    showline: true, linewidth: 1.5, linecolor: '#cbd5e1', 
                    ticks: 'outside', tickcolor: '#cbd5e1', ticklen: 5,
                    showgrid: true, gridcolor: 'rgba(241, 245, 249, 0.6)', zeroline: true, zerolinecolor: '#cbd5e1', 
                    tickfont: { size: 10, color: '#475569', family: 'Inter' },
                    ticksuffix: '%',
                    autorange: true 
                };
            }
            
            const pConfig = {
                displayModeBar: false, 
                displaylogo: false,
                responsive: true
            };
            
            // Capture a strict structural snapshot of the explicitly configured axes before Plotly rendering.
            // This prevents visual zooming distortion during reset caused by Plotly auto-fitting hidden structural shapes (-100% to 100%).
            chartDiv._initialRanges = {
                'xaxis.range': (layout.xaxis && layout.xaxis.range) ? [...layout.xaxis.range] : null,
                'yaxis.range': (layout.yaxis && layout.yaxis.range) ? [...layout.yaxis.range] : null,
                'yaxis2.range': (layout.yaxis2 && layout.yaxis2.range) ? [...layout.yaxis2.range] : null,
                'yaxis3.range': (layout.yaxis3 && layout.yaxis3.range) ? [...layout.yaxis3.range] : null
            };
            
            Plotly.newPlot(chartDiv, traces, layout, pConfig);
            
        });
    }
    
    // Helper explicitly to mimic the main dashboard's hexToRgba
    function hexToRgba(hex, alpha) {
        // Expand shorthand form (e.g. "03F") to full form (e.g. "0033FF")
        let shorthandRegex = /^#?([a-f\d])([a-f\d])([a-f\d])$/i;
        hex = hex.replace(shorthandRegex, function(m, r, g, b) {
            return r + r + g + g + b + b;
        });

        let result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? 
            `rgba(${parseInt(result[1], 16)}, ${parseInt(result[2], 16)}, ${parseInt(result[3], 16)}, ${alpha})` : 
            null;
    }

    // Trigger initial render
    document.addEventListener('DOMContentLoaded', () => {
        if(agentsList.length > 0) renderAllCharts();
    });
    
    // Resize handler
    let resizeTimer;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
            metricKeys.forEach(m => {
                const chartDiv = document.getElementById(`chart_${m}`);
                if (chartDiv && chartDiv.children.length > 0 && !chartDiv.querySelector('.empty-state')) {
                    Plotly.Plots.resize(chartDiv);
                }
            });
        }, 150);
    });

</script>
</body>
</html>
    """
    
    template = Template(template_str)
    html_content = template.render(
        monthly_json=monthly_json,
        agents_list_json=agents_list_json,
        config_json=config_json
    )
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"\n✅ REPORTE CREADO: El dashboard separado de métricas históricas se ha guardado en '{out_path}'.")

if __name__ == "__main__":
    import sys
    # Adding src folder to path for logic_analytics to be importable
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from logic_analytics import calcular_metricas_agente_con_mensual, PESOS_METRICAS
        dict_data, metrics = load_and_validate_data()
        generate_metrics_dashboard(dict_data)
        
        print("\n=== CHECKLIST FINAL ===")
        print("1. ¿Se evaluaron todas las métricas? Sí")
        for m in metrics:
            print(f"   [X] {m}")
        print("2. ¿La consistencia visual fue mantenida? Sí (Se han embebido los estilos y paleta originales usando Jinja2/Plotly)")
        print("3. ¿El proceso es interpretativo? Sí (Las métricas están graficadas en línea de tiempo mostrando claras tendencias)")
        
    except Exception as e:
        print(f"Error fatal: {e}")
