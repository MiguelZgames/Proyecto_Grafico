import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json
from jinja2 import Template
from src.logic_analytics import PESOS_METRICAS

def calculate_similarity(row, centroids, class_order, metrics):
    current_class = row['Clase']
    if current_class not in class_order:
        return None
        
    try:
        current_idx = class_order.index(current_class)
        # Find next better class (lower index) that actually has a centroid
        target_class = None
        target_vals = None
        
        # Search upwards (towards index 0)
        for i in range(current_idx - 1, -1, -1):
            candidate = class_order[i]
            if candidate in centroids:
                target_class = candidate
                target_vals = centroids[candidate]
                break
                
        if not target_class:
            return {"target": "Top", "dist": 0, "gaps": []}
            
        # Calculate Cosine Distance
        # 1 - Cosine Similarity
        # Similarity = (A . B) / (||A|| * ||B||)
        
        vec_a = [] # Current
        vec_b = [] # Target
        
        for m in metrics:
            val_a = float(row.get(m, 0))
            val_b = float(target_vals.get(m, 0))
            vec_a.append(val_a)
            vec_b.append(val_b)
            
        a = np.array(vec_a)
        b = np.array(vec_b)
        
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            dist = 1.0 # Max distance if one vector is zero
        else:
            sim = dot / (norm_a * norm_b)
            # Clip to [-1, 1] to avoid float errors
            sim = max(-1.0, min(1.0, sim))
            dist = 1.0 - sim
            
        # Identify Gaps (still based on magnitude difference for actionable advice)
        gaps = []
        for i, m in enumerate(metrics):
            diff = vec_b[i] - vec_a[i]
            if diff > 0.5: # Improvement needed (threshold to avoid noise)
                gaps.append({
                    "metric": m, 
                    "diff": diff, 
                    "target": vec_b[i], 
                    "current": vec_a[i],
                    "impact": diff * PESOS_METRICAS.get(m, 0.1) # Prioritize by weighted impact
                })
        
        # Sort gaps by weighted impact
        gaps.sort(key=lambda x: x['impact'], reverse=True)
        
        return {
            "target": target_class,
            "dist": round(dist, 4), # 4 decimals for cosine
            "gaps": gaps[:3] # Top 3 improvements
        }
    except Exception as e:
        print(f"Error in similarity: {e}")
        return None


def generate_html_report(df_agents, df_monthly=None, out_path="reports/dashboard.html"):
    """
    Genera un dashboard HTML autocontenido con los resultados de la clasificación.
    """
    print("Generating HTML Report...")
    
    # 1. Preprocesamiento de datos para visualización
    df = df_agents.copy()
    
    # KPIs Generales
    total_agents = len(df)
    
    # Detailed Class Counts
    # Dynamically find classes and sort them logically
    unique_classes = df['Clase'].unique().tolist()
    
    def sort_key_clase(c):
        # A < B < C
        # +++ < ++ < + < nothing
        # Map letter to number: A=100, B=200, C=300
        if not isinstance(c, str): return 999
        letter = c[0]
        rank = 0
        if letter == 'A': rank = 100
        elif letter == 'B': rank = 200
        elif letter == 'C': rank = 300
        else: rank = 400
        
        # Suffix: +++ = -3, ++ = -2, + = -1, "" = 0
        suffix = c.count('+')
        return rank - suffix

    unique_classes.sort(key=sort_key_clase)
    class_order = unique_classes
    
    class_counts = {cls: len(df[df['Clase'] == cls]) for cls in class_order}
    
    pct_risky = (len(df[df['Risk_Safe'] == 0]) / total_agents * 100) if total_agents > 0 else 0
    
    # Calculate Centroids for Similarity Analysis
    metrics_for_sim = [
        'rentabilidad', 'volumen', 'fidelidad', 'estabilidad', 
        'crecimiento', 'eficiencia_casino', 'eficiencia_deportes', 
        'eficiencia_conversion', 'tendencia', 'diversificacion', 'calidad_jugadores'
    ]
    # Ensure all exist
    for m in metrics_for_sim:
        if m not in df.columns: df[m] = 0
        
    centroids = df.groupby('Clase')[metrics_for_sim].mean().to_dict('index')
    
    # Pre-calculate Distance and Gaps for each agent
    df['sim_data'] = df.apply(lambda row: calculate_similarity(row, centroids, class_order, metrics_for_sim), axis=1)
    
    # Top Agent (Rank #1) for Radar Reference
    top_agent = df[df['rank_global'] == 1].iloc[0] if not df[df['rank_global'] == 1].empty else df.iloc[0]
    
    # Convert DataFrame to list of dicts for JS
    # Handle NaNs and infinite values for JSON serialization
    df_json = df.fillna(0).replace([np.inf, -np.inf], 0).to_dict(orient='records')
    
    # Export Centroids and Configuration for Dynamic JS Analysis
    centroids_json = json.dumps(centroids)
    metrics_json = json.dumps(metrics_for_sim)
    class_order_json = json.dumps(class_order)
    weights_json = json.dumps(PESOS_METRICAS)
    
    monthly_data_js = "null"
    if df_monthly is not None and not df_monthly.empty:
        # Group by agent and convert to dict {agent_id: [{month, comision, depositos, ...}, ...]}
        monthly_cols = [
            'id_agente', 'month', 'calculo_comision', 'total_depositos', 'active_players', 'total_retiros', 'calculo_ggr', 'calculo_ngr',
            'ggr_deportiva', 'ggr_casino', 'total_apuesta_deportiva', 'total_apuesta_casino',
            'score_global', 
            'rentabilidad', 'volumen', 'fidelidad', 'estabilidad', 
            'crecimiento', 'eficiencia_casino', 'eficiencia_deportes', 
            'eficiencia_conversion', 'tendencia', 'diversificacion', 'calidad_jugadores',
            'Clase', 'Risk_Safe'
        ]
        available_cols = [c for c in monthly_cols if c in df_monthly.columns]
        
        # Sanitize Data: Fill NaNs and Replace Infs
        monthly_min = df_monthly[available_cols].copy()
        monthly_min = monthly_min.fillna(0).replace([np.inf, -np.inf], 0)
        monthly_min['month'] = monthly_min['month'].astype(str)
        
        # Build monthly dict using plain Python dicts (avoids pandas groupby issues)
        records = monthly_min.to_dict(orient='records')
        monthly_dict = {}
        for row in records:
            key = str(int(row.pop('id_agente')))
            monthly_dict.setdefault(key, []).append(row)
        
        monthly_data_js = json.dumps(monthly_dict)




    # 2. Plantilla HTML (Jinja2)
    
    template_str = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard de Clasificación de Agentes</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #fcfdfe; /* Ultra soft blue-white */
            --card-bg: #ffffff;
            --text-color: #475569; /* Slate 600 - Softer than black */
            --text-muted: #94a3b8; /* Slate 400 */
            --border: #f1f5f9; /* Slate 100 */
            
            --primary: #334155; /* Slate 700 - Soft Header */
            --accent: #60a5fa; /* Blue 400 - Soft Sky Blue */
            --accent-soft: rgba(96, 165, 250, 0.1);
            
            --accent-2: #c084fc; /* Purple 400 - Soft Purple */
            
            --success: #4ade80; /* Green 400 - Soft Mint */
            --warning: #fbbf24; /* Amber 400 - Soft Gold */
            --danger: #f87171; /* Red 400 - Soft Coral */
            
            --grid: #f1f5f9;
            --shadow: 0 2px 4px rgba(0,0,0,0.02);
            --shadow-hover: 0 8px 16px rgba(0,0,0,0.04);
        }
        body { font-family: 'Inter', sans-serif; background-color: var(--bg-color); color: var(--text-color); margin: 0; padding: 0; transition: background-color 0.3s; }

        /* Custom Plotly Rangeslider Handle Styling */
        .js-plotly-plot .rangeslider-handle-min,
        .js-plotly-plot .rangeslider-handle-max {
            fill: var(--accent) !important; /* Soft Blue */
            stroke: var(--card-bg) !important; /* White border */
            stroke-width: 2px !important;
            rx: 3px !important; /* Rounded corners */
            cursor: ew-resize;
        }
        
        .js-plotly-plot .rangeslider-handle-min:hover,
        .js-plotly-plot .rangeslider-handle-max:hover {
            fill: var(--accent-2) !important; /* Soft Purple on hover */
        }

        /* Animations */
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .animate-in { animation: fadeIn 0.5s ease-out forwards; }
        
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; height: calc(100vh - 80px); display: flex; flex-direction: column; }
        
        /* Header */
        header { 
            background: var(--card-bg); padding: 10px 24px; 
            border-bottom: 1px solid var(--border); 
            display: flex; justify-content: space-between; align-items: center; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            position: sticky; top: 0; z-index: 100;
        }
        h1 { margin: 0; font-size: 18px; font-weight: 600; color: var(--primary); letter-spacing: -0.5px; }
        .header-meta { font-size: 12px; color: var(--text-muted); margin-top: 2px; font-weight: 500; }
        
        /* Updated KPI Bar with Cards */
        .kpi-bar { display: flex; gap: 20px; }
        .kpi { text-align: center; position: relative; }
        .kpi-val { font-size: 18px; font-weight: 700; line-height: 1.2; }
        .kpi-lbl { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); font-weight: 500; }
        .tag-a { color: var(--success); } .tag-b { color: var(--warning); } .tag-c { color: var(--danger); }
        
        /* Layout Grid */
        .grid { display: grid; grid-template-columns: 280px 1fr; gap: 16px; flex: 1; overflow: hidden; margin-top: 10px; }
        
        /* Sidebar */
        .sidebar { 
            background: var(--card-bg); border-radius: 12px; 
            display: flex; flex-direction: column; 
            box-shadow: var(--shadow); border: 1px solid var(--border);
            overflow: hidden;
        }
        .sidebar-header { 
            padding: 12px 14px; 
            border-bottom: 1px solid var(--border); 
            background: #fafbfc; 
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .search-box { 
            width: 100%; padding: 10px 12px; border: 1px solid #dfe6e9; border-radius: 6px; 
            font-family: inherit; font-size: 13px; outline: none; transition: border-color 0.2s, box-shadow 0.2s;
            box-sizing: border-box;
        }
        .search-box:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
        
        .date-filter-container {
            background: white;
            border: 1px solid #e1e4e8;
            border-radius: 8px;
            padding: 10px;
        }
        .date-filter-label {
            font-size: 10px;
            font-weight: 700;
            color: var(--text-muted);
            margin-bottom: 6px;
            display: block;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .date-inputs {
            display: flex;
            gap: 5px;
            align-items: center;
        }
        
        .month-filter {
            flex: 1; 
            min-width: 0;
            padding: 6px 4px; 
            border: 1px solid #dfe6e9; border-radius: 6px;
            font-family: inherit; font-size: 11px; color: var(--text-color);
            background: #f8f9fa; outline: none; cursor: pointer;
            transition: all 0.2s;
            box-sizing: border-box;
        }
        .month-filter:hover { background: white; border-color: #cbd5e0; }
        .month-filter:focus { background: white; border-color: var(--accent); }
        
        .filter-btns { display: flex; flex-wrap: wrap; gap: 6px; }
        .filter-btn { 
            flex: 1 1 auto; min-width: 40px; padding: 6px 10px; border: 1px solid #e1e4e8; background: white; 
            border-radius: 6px; cursor: pointer; font-size: 11px; font-weight: 600; color: var(--text-muted);
            transition: all 0.2s; text-align: center;
        }
        .filter-btn:hover { background: #f1f2f6; color: var(--text-color); }
        .filter-btn.active { background: var(--primary); color: white; border-color: var(--primary); box-shadow: 0 2px 4px rgba(44, 62, 80, 0.2); }
        .filter-btn.active-A { background: var(--success); border-color: var(--success); color: white; }
        .filter-btn.active-B { background: var(--warning); border-color: var(--warning); color: white; }
        .filter-btn.active-C { background: var(--danger); border-color: var(--danger); color: white; }
        
        .agent-list { flex: 1; overflow-y: auto; padding: 0; }
        .agent-item { 
            padding: 8px 14px; border-bottom: 1px solid var(--border); cursor: pointer; 
            display: flex; justify-content: space-between; align-items: center; 
            transition: background 0.15s;
        }
        .agent-item:hover { background-color: #f8f9fa; }
        .agent-item.active { background-color: var(--accent-soft); border-left: 4px solid var(--accent); padding-left: 10px; }
        .agent-rank { font-size: 12px; font-weight: 600; color: var(--text-muted); width: 35px; }
        .agent-name { flex: 1; font-weight: 500; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--text-color); }
        .agent-badge { 
            font-size: 10px; font-weight: 700; padding: 3px 8px; border-radius: 12px; 
            min-width: 20px; text-align: center;
        }
        /* Generic Badge Styles using Attribute Selectors */
        [class*="badge-A"] { background: rgba(46, 204, 113, 0.15); color: #27ae60; }
        [class*="badge-B"] { background: rgba(241, 196, 15, 0.15); color: #d35400; }
        [class*="badge-C"] { background: rgba(231, 76, 60, 0.15); color: #c0392b; }

        /* Main Content */
        .main-content { overflow-y: auto; display: flex; flex-direction: column; gap: 16px; padding-right: 5px; }
        .main-content::-webkit-scrollbar { width: 6px; }
        .main-content::-webkit-scrollbar-thumb { background: #ccc; border-radius: 3px; }
        
        .card { 
            background: var(--card-bg); border-radius: 10px; padding: 16px; 
            box-shadow: var(--shadow); border: 1px solid var(--border);
            transition: transform 0.2s, box-shadow 0.2s;
            overflow: visible; /* Ensure content is not clipped */
            position: relative; /* Context for absolute positioning if needed */
        }
        .card:hover { box-shadow: var(--shadow-hover); }
        
        /* Profile Header */
        .profile-header { display: flex; justify-content: space-between; margin-bottom: 12px; border-bottom: 1px solid var(--border); padding-bottom: 10px; }
        .profile-name { font-size: 22px; font-weight: 700; margin: 0 0 3px 0; color: var(--primary); }
        .profile-id { font-family: monospace; color: var(--text-muted); font-size: 12px; background: #f0f2f5; padding: 2px 6px; border-radius: 4px; }
        .profile-score-val { font-size: 26px; font-weight: 800; color: var(--primary); letter-spacing: -1px; text-align: right; }
        .profile-score-lbl { font-size: 10px; color: var(--text-muted); text-transform: uppercase; text-align: right; letter-spacing: 1px; }
        
        .status-badge { display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
        .status-safe { background: #d4edda; color: #155724; }
        .status-risky { background: #f8d7da; color: #721c24; }
        
        /* Metrics Grid */
        .metrics-category { margin-bottom: 10px; }
        .metrics-title { 
            font-size: 13px; font-weight: 700; color: var(--text-color); margin-bottom: 8px; 
            display: flex; align-items: center; gap: 6px;
            padding-bottom: 6px; border-bottom: 1px dashed var(--border);
        }
        .metrics-grid { 
            display: flex; 
            flex-wrap: wrap; 
            gap: 10px; 
            width: 100%;
            min-height: 50px; /* Ensure it has height */
        }
        .metric-box { 
            background: #fafbfc; 
            padding: 10px 6px; 
            border-radius: 6px; 
            text-align: center; 
            border: 1px solid #eee;
            transition: transform 0.2s;
            flex: 1 1 110px; /* Grow, shrink, base 110px */
        }
        .metric-box:hover { transform: translateY(-2px); border-color: var(--accent); }
        .metric-val { font-size: 15px; font-weight: 700; color: var(--primary); margin-bottom: 2px; }
        .metric-lbl { font-size: 9px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; }
        
        /* Charts Area */
        .radar-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 0;
            align-items: stretch;
        }
        
        .radar-section .card {
            height: 100%;
            display: flex;
            flex-direction: column;
            /* Importante: no centrar TODO el contenido del card, solo el gráfico.
               Si centramos el card entero, la barra de etiquetas se "va" al centro. */
            align-items: stretch;
            justify-content: flex-start;
            padding: 14px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        
        #radarChart {
            width: 100%;
            height: 100%;
            min-height: 400px;
            position: relative;
        }

        .trend-section {
            width: 100%;
            margin-bottom: 0;
            margin-top: 8px;
            padding-top: 20px;
            border-top: 2px solid var(--border);
        }
        
        .chart-title { 
            width: 100%; /* Ensure full width even with centered card items */
            font-size: 15px;
            font-weight: 700; 
            letter-spacing: -0.025em;
            margin-bottom: 8px; 
            color: var(--primary); 
            display: flex; 
            align-items: center; 
            gap: 10px; 
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
        }

        .reset-btn {
            background: #fff;
            border: 1px solid #e2e8f0;
            color: #64748b;
            border-radius: 6px;
            cursor: pointer;
            padding: 6px 10px;
            font-size: 14px;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-left: 8px;
        }
        .reset-btn:hover {
            background: #f1f5f9;
            color: #0f172a;
            border-color: #cbd5e1;
        }
        
        /* Checklist */
        .checklist { display: grid; grid-template-columns: 1fr; gap: 10px; font-size: 13px; margin-top: 15px; background: #f8f9fa; padding: 15px; border-radius: 8px; }
        .check-item { display: flex; align-items: center; gap: 8px; font-weight: 500; }
        .check-icon { font-size: 14px; }
        .pass { color: var(--success); } .fail { color: var(--text-muted); opacity: 0.6; }

        /* Tables */
        table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 13px; margin-top: 10px; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #e2e8f0; }
        th { 
            font-weight: 600; color: var(--text-muted); text-transform: uppercase; font-size: 11px; 
            background: #f8fafc; position: sticky; top: 0; z-index: 10; letter-spacing: 0.5px;
        }
        tbody tr:hover { background-color: #f8fafc; transition: background 0.15s; }
        tr:last-child td { border-bottom: none; }
        .num-col { text-align: right; font-family: 'Inter', monospace; font-variant-numeric: tabular-nums; }
        
        /* Responsive */
        @media (max-width: 1200px) { 
            .radar-section { grid-template-columns: 1fr; }
            .chart-title { font-size: 16px; }
        }
        @media (max-width: 1366px) { .grid { grid-template-columns: 260px 1fr; } }
        @media (max-width: 1100px) { 
            .grid { grid-template-columns: 220px 1fr; } 
            /* .metrics-grid handled by auto-fill */
            /* .radar-section handled above */
        }
        @media (max-width: 768px) { 
            .grid { grid-template-columns: 1fr; } 
            .sidebar { height: auto; max-height: 400px; } 
            .kpi-bar { flex-wrap: wrap; gap: 15px; }
        }
        /* Multi-Select Compare */
        .ms-container { position: relative; }
        .ms-trigger {
            display: flex; align-items: center; gap: 4px;
            padding: 4px 10px; border-radius: 6px; border: 1px solid #e2e8f0;
            font-size: 13px; color: var(--text-color); cursor: pointer;
            transition: all 0.2s; background: white; min-height: 30px;
            flex-wrap: nowrap; white-space: nowrap;
        }
        .ms-trigger:hover { border-color: #cbd5e1; }
        .ms-trigger.open { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
        .ms-placeholder { color: var(--text-muted); font-size: 12px; white-space: nowrap; }
        .ms-tag {
            display: inline-flex; align-items: center; gap: 3px;
            padding: 2px 8px; border-radius: 10px; font-size: 10px;
            font-weight: 600; color: white; white-space: nowrap;
            animation: fadeIn 0.2s ease-out;
        }
        .ms-tag .ms-remove { cursor: pointer; opacity: 0.7; font-size: 12px; margin-left: 2px; }
        .ms-tag .ms-remove:hover { opacity: 1; }
        .ms-count { font-size: 11px; color: var(--text-muted); font-weight: 600; white-space: nowrap; }
        .ms-selected-bar {
            display: flex; flex-wrap: wrap; gap: 5px; align-items: center;
            padding: 8px 0 0 0;
            width: 100%;
            justify-content: flex-start;
            align-self: flex-start;
        }
        .ms-dropdown {
            position: absolute; top: calc(100% + 4px); right: 0; width: 300px;
            background: white; border: 1px solid #e2e8f0; border-radius: 8px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.12); z-index: 50;
            display: none;
        }
        .ms-dropdown.open { display: block; }
        .ms-search {
            padding: 8px; border-bottom: 1px solid #f1f5f9;
            position: sticky; top: 0; background: white; z-index: 1;
        }
        .ms-search input {
            width: 100%; border: 1px solid #e2e8f0; border-radius: 5px;
            padding: 6px 8px; font-size: 12px; outline: none;
            font-family: inherit; box-sizing: border-box;
        }
        .ms-search input:focus { border-color: var(--accent); }
        .ms-options { overflow-y: auto; max-height: 240px; }
        .ms-option {
            display: flex; align-items: center; gap: 8px;
            padding: 7px 12px; cursor: pointer; font-size: 12px;
            transition: background 0.1s; user-select: none;
        }
        .ms-option:hover { background: #f8fafc; }
        .ms-option.selected { background: var(--accent-soft); }
        .ms-option input[type="checkbox"] { accent-color: var(--accent); cursor: pointer; flex-shrink: 0; }
        .ms-option-rank { color: var(--text-muted); font-size: 11px; min-width: 28px; }
        .ms-option-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .ms-color-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
        
        /* Info Icon & Tooltip */
        .info-icon {
            display: inline-flex; align-items: center; justify-content: center;
            width: 16px; height: 16px; border-radius: 50%;
            background: #e2e8f0; color: #64748b;
            font-size: 11px; font-weight: bold; cursor: help;
            margin-left: 8px; position: relative;
            border: 1px solid #cbd5e1;
        }
        .info-icon:hover { background: #cbd5e1; color: #334155; }

        .info-icon .tooltip-text {
            visibility: hidden; width: 220px;
            background-color: #1e293b; color: #fff;
            text-align: left; border-radius: 6px; padding: 8px 10px;
            position: absolute; z-index: 100;
            bottom: 125%; left: 50%; margin-left: -110px;
            font-size: 11px; font-weight: 400; line-height: 1.4;
            opacity: 0; transition: opacity 0.2s;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
            pointer-events: none;
        }
        .info-icon .tooltip-text::after {
            content: ""; position: absolute; top: 100%; left: 50%;
            margin-left: -5px; border-width: 5px; border-style: solid;
            border-color: #1e293b transparent transparent transparent;
        }
        .info-icon:hover .tooltip-text { visibility: visible; opacity: 1; }
    </style>
</head>
<body>

<header class="animate-in">
    <div>
        <h1>Dashboard de Clasificación de Agentes</h1>
        <div class="header-meta">Total Agencias Únicas: {{ total_agents }}</div>
    </div>
    <div class="kpi-bar">
        {% for cls, count in class_counts.items() %}
        {% if count > 0 %}
        <div class="kpi" style="min-width: 40px;">
            <div class="kpi-val" style="font-size:18px;">{{ count }}</div>
            <div class="kpi-lbl">{{ cls }}</div>
        </div>
        {% endif %}
        {% endfor %}
    </div>
</header>

<div class="container grid animate-in" style="animation-delay: 0.1s;">
    <!-- Sidebar -->
    <div class="sidebar">
        <div class="sidebar-header">
            <input type="text" id="searchInput" class="search-box" placeholder="Buscar por ID o Nombre..." onkeyup="filterList()">
            

            
            <div class="filter-btns">
                <button class="filter-btn active" id="btn-all" onclick="filterClass('all')">Todos</button>
                {% for cls in class_counts.keys() %}
                <button class="filter-btn" id="btn-{{ cls }}" onclick="filterClass('{{ cls }}')">{{ cls }}</button>
                {% endfor %}
            </div>
        </div>
        <div class="agent-list" id="agentList"></div>
    </div>

    <!-- Main Content -->
    <div class="main-content">
        <!-- Profile -->
        <div class="card" id="profileCard">
            <div class="profile-header">
                <div>
                    <h2 class="profile-name" id="pName">Seleccione un agente</h2>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span class="profile-id">ID: <span id="pId">-</span></span>
                        <span style="font-size: 12px; color: var(--text-muted);">| Rank Global: <strong id="pRank" style="color:var(--text-color)">-</strong></span>
                    </div>
                    <div style="margin-top: 10px; display: flex; gap: 10px;">
                        <div id="pClassBadge" class="status-badge status-safe">-</div>
                        <div id="pRiskBadge" class="status-badge">-</div>
                    </div>
                </div>
                <div>
                    <div class="profile-score-val" id="pScore">-</div>
                    <div class="profile-score-lbl">Score Global</div>
                </div>
            </div>

            <div class="metrics-category">
                <h3 class="metrics-title">💰 Resumen Financiero</h3>
                <div class="metrics-grid">
                    <div class="metric-box"><div class="metric-val" id="valDep">-</div><div class="metric-lbl">Depósitos</div></div>
                    <div class="metric-box"><div class="metric-val" id="valRet">-</div><div class="metric-lbl">Retiros</div></div>
                    <div class="metric-box"><div class="metric-val" id="valGGR">-</div><div class="metric-lbl">GGR</div></div>
                    <div class="metric-box"><div class="metric-val" id="valNGR">-</div><div class="metric-lbl">NGR</div></div>
                    <div class="metric-box"><div class="metric-val" id="valCom">-</div><div class="metric-lbl">Comisión</div></div>
                </div>
            </div>
            
            <div class="metrics-category" style="margin-bottom:0;">
                <h3 class="metrics-title">⚙️ Indicadores de Rendimiento</h3>
                <div class="metrics-grid">
                    <div class="metric-box"><div class="metric-val" id="valRen">-</div><div class="metric-lbl">Rentabilidad</div></div>
                    <div class="metric-box"><div class="metric-val" id="valVol">-</div><div class="metric-lbl">Volumen</div></div>
                    <div class="metric-box"><div class="metric-val" id="valFid">-</div><div class="metric-lbl">Fidelidad</div></div>
                    <div class="metric-box"><div class="metric-val" id="valEst">-</div><div class="metric-lbl">Estabilidad</div></div>
                    <div class="metric-box"><div class="metric-val" id="valCre">-</div><div class="metric-lbl">Crecimiento</div></div>
                    <div class="metric-box"><div class="metric-val" id="valCas">-</div><div class="metric-lbl">Efic. Casino</div></div>
                    <div class="metric-box"><div class="metric-val" id="valDepScore">-</div><div class="metric-lbl">Efic. Deportes</div></div>
                    <div class="metric-box"><div class="metric-val" id="valConv">-</div><div class="metric-lbl">Conversión</div></div>
                    <div class="metric-box"><div class="metric-val" id="valTen">-</div><div class="metric-lbl">Tendencia</div></div>
                    <div class="metric-box"><div class="metric-val" id="valDiv">-</div><div class="metric-lbl">Diversificación</div></div>
                    <div class="metric-box"><div class="metric-val" id="valCal">-</div><div class="metric-lbl">Calidad</div></div>
                </div>
            </div>
        </div>
        
        <!-- Charts -->
        <div class="radar-section">
            <div class="card">
                <div class="chart-title">
                    <span style="font-size:24px; background:var(--bg-color); padding:8px; border-radius:8px;">🕸️</span> 
                    <div style="flex:1;">
                        <div style="line-height:1.2; display:flex; align-items:center;">
                            Análisis Radar
                            <span class="info-icon">! 
                                <span class="tooltip-text">Comparativa multidimensional del agente frente al promedio del mercado en 10 variables clave (escala 0-10).</span>
                            </span>
                        </div>
                        <div style="font-size:12px; color:var(--text-muted); font-weight:400; margin-top:2px;">Perfil vs Promedio Global</div>
                    </div>
                    <div style="display:flex; align-items:center; gap:6px;">
                        <div class="ms-container" id="msContainer">
                            <div class="ms-trigger" id="msTrigger" onclick="toggleMultiSelect(event)">
                                <span class="ms-placeholder" id="msPlaceholder">+ Comparar</span>
                            </div>
                            <div class="ms-dropdown" id="msDropdown">
                                <div class="ms-search">
                                    <input type="text" id="msSearchInput" placeholder="Buscar agente..." oninput="filterCompareOptions(this.value)">
                                </div>
                                <div class="ms-options" id="msOptions"></div>
                            </div>
                        </div>
                        <button class="reset-btn" onclick="clearCompareSelection()" title="Limpiar comparación">
                            ↺
                        </button>
                    </div>
                </div>
                <div class="ms-selected-bar" id="msSelectedBar"></div>
                <!-- Fixed height container to ensure stability -->
                <div id="radarChart" style="width: 100%; height: 500px;"></div>
            </div>
            <div class="card">
                <div class="chart-title">
                    <span style="font-size:24px; background:var(--bg-color); padding:8px; border-radius:8px;">📊</span> 
                    <div>
                        <div style="line-height:1.2; display:flex; align-items:center;">
                            Desempeño por Indicadores (KPIs)
                            <span class="info-icon">! 
                                <span class="tooltip-text">Desglose porcentual de cada métrica con delta comparativo. Muestra fortalezas y áreas donde el agente supera o está por debajo del benchmark.</span>
                            </span>
                        </div>
                        <div style="font-size:12px; color:var(--text-muted); font-weight:400; margin-top:2px;">Desglose detallado vs Benchmark</div>
                    </div>
                    <button class="reset-btn" onclick="resetChart('barChart')" title="Reiniciar gráfico" style="margin-left:auto;">
                        ↺
                    </button>
                </div>

                <div id="barChart" style="width: 100%; height: 500px;"></div>
            </div>
        </div>

        <div class="trend-section">
            <div class="card">
                <div class="chart-title">
                    <span style="font-size:20px;">📈</span> 
                    <span style="display:flex; align-items:center;">
                        Tendencia Histórica
                        <span class="info-icon">! 
                            <span class="tooltip-text">Evolución histórica de volúmenes (depósitos/retiros) y métricas financieras (GGR, comisiones) para identificar patrones de crecimiento o estabilidad.</span>
                        </span>
                    </span>
                    <div style="margin-left:auto; display:flex; gap:5px;">
                        <select id="metricSelect" onchange="updateTrendChart()" style="padding:5px; border-radius:4px; border:1px solid #ddd; font-size:12px;">
                            <option value="total_depositos">Depósitos</option>
                            <option value="total_retiros">Retiros</option>
                            <option value="calculo_ggr">GGR</option>
                            <option value="calculo_comision">Comisión</option>
                            <option value="score_global">Score Global</option>
                            <option value="active_players">Jugadores Activos</option>
                        </select>
                        <button onclick="exportTrendData()" style="padding:5px 10px; background:#2ecc71; color:white; border:none; border-radius:4px; cursor:pointer; font-size:11px;">Excel</button>
                        <button class="reset-btn" onclick="resetChart('trendChart')" title="Reiniciar gráfico" style="margin-left:0;">
                            ↺
                        </button>
                    </div>
                </div>
                <div id="trendChart" style="height: 100%; min-height: 40vh; width: 100%;"></div>
            </div>
        </div>
        


        <!-- Monthly Table -->
        <div class="card">
            <div class="chart-title">
                <span style="display:flex; align-items:center;">
                    📅 Detalle Mensual
                    <span class="info-icon">! 
                        <span class="tooltip-text">Tabla detallada con los valores mensuales de depósitos, retiros, GGR, NGR y comisiones generadas.</span>
                    </span>
                </span>
            </div>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Mes</th>
                            <th class="num-col">Jugadores</th>
                            <th class="num-col">Score</th>
                            <th class="num-col">Depósitos</th>
                            <th class="num-col">Retiros</th>
                            <th class="num-col">GGR</th>
                            <th class="num-col">NGR</th>
                            <th class="num-col">Comisión</th>
                        </tr>
                    </thead>
                    <tbody id="monthlyTableBody"></tbody>
                </table>
            </div>
        </div>
        
        <!-- Similarity Analysis -->
        <div class="card">
            <div class="chart-title">
                <span style="display:flex; align-items:center;">
                    🎯 Análisis de Mejora (Próximo Nivel)
                    <span class="info-icon">! 
                        <span class="tooltip-text">Identificación automática de las 3 métricas con mayor brecha respecto al siguiente nivel de clasificación, sugiriendo áreas prioritarias de enfoque.</span>
                    </span>
                </span>
            </div>
            <div id="simContent" style="display:none;">
                <div style="display:flex; gap:20px; align-items:center; margin-bottom:15px;">
                    <div>
                        <div style="font-size:12px; color:var(--text-muted); text-transform:uppercase;">Objetivo</div>
                        <div style="font-size:18px; font-weight:700; color:var(--accent);" id="targetClass">-</div>
                    </div>
                    <div>
                        <div style="font-size:12px; color:var(--text-muted); text-transform:uppercase;">Distancia</div>
                        <div style="font-size:24px; font-weight:700;" id="targetDist">-</div>
                    </div>
                </div>
                
                <h4 style="margin:0 0 10px 0; font-size:14px; color:var(--primary);">Top 3 Métricas a Mejorar:</h4>
                <div id="gapList" class="checklist">
                    <!-- Gaps injected via JS -->
                </div>
            </div>
            <div id="simEmpty" style="color:var(--text-muted); font-style:italic;">
                Este agente ya está en la categoría más alta o no hay datos suficientes.
            </div>
        </div>
    </div>
</div>

<script>
    const allAgents = {{ agents_json | safe }}; // Immutable source
    let displayedAgents = [...allAgents]; // Mutable display list
    
    // Dynamic Analysis Data
    const centroids = {{ centroids_json | safe }};
    const simMetrics = {{ metrics_json | safe }};
    const classOrder = {{ class_order_json | safe }};
    
    // Fallback for topAgent if needed, though we dynamically calc it now
    let staticTopAgent = {{ top_agent_json | safe }};
    
    const monthlyData = {{ monthly_json | safe }}; 
    const globalWeights = {{ weights_json | safe }}; 

    const listEl = document.getElementById('agentList');
    let currentFilter = 'all';
    let selectedCompareIds = [];
    const compareColors = ['#f59e0b','#10b981','#ef4444','#06b6d4','#ec4899','#e11d48','#84cc16','#f97316'];

    function hexToRgba(hex, alpha) {
        const r = parseInt(hex.slice(1,3), 16);
        const g = parseInt(hex.slice(3,5), 16);
        const b = parseInt(hex.slice(5,7), 16);
        return `rgba(${r},${g},${b},${alpha})`;
    }

    // Close multi-select on outside click
    document.addEventListener('click', function(e) {
        const container = document.getElementById('msContainer');
        if (container && !container.contains(e.target)) {
            const dd = document.getElementById('msDropdown');
            const trigger = document.getElementById('msTrigger');
            if (dd) dd.classList.remove('open');
            if (trigger) trigger.classList.remove('open');
        }
    });
    
    // 1. Determine Date Range
    const allMonths = new Set();
    if(monthlyData) {
        Object.values(monthlyData).forEach(list => {
            list.forEach(item => allMonths.add(item.month));
        });
    }
    const sortedMonths = Array.from(allMonths).sort(); // Ascending for range
    
    // Init Date Inputs - Removed


    function calculateSimilarityJS(agent) {
        if (!classOrder || !centroids || !simMetrics) return agent.sim_data; // Fallback
        
        const currentClass = agent.Clase;
        if (!classOrder.includes(currentClass)) return null;
        
        const currentIdx = classOrder.indexOf(currentClass);
        
        // Find next better class (lower index) that actually has a centroid
        let targetClass = null;
        let targetVals = null;
        
        for (let i = currentIdx - 1; i >= 0; i--) {
            const candidate = classOrder[i];
            if (centroids[candidate]) {
                targetClass = candidate;
                targetVals = centroids[candidate];
                break;
            }
        }
        
        if (!targetClass) return { target: "Top", dist: 0, gaps: [] };
        
        // Calculate Cosine Distance
        // 1 - Cosine Similarity
        
        let dot = 0;
        let normA = 0;
        let normB = 0;
        const gaps = [];
        
        // Use weights for gap impact
        const weights = window.weightsData || {{ weights_json | safe }} || {};
        
        simMetrics.forEach(m => {
            const valA = agent[m] || 0;
            const valB = targetVals[m] || 0;
            
            dot += valA * valB;
            normA += valA * valA;
            normB += valB * valB;
            
            const diff = valB - valA;
            if (diff > 0.5) { // Improvement needed (threshold)
                gaps.push({ 
                    metric: m, 
                    diff: diff, 
                    target: valB, 
                    current: valA,
                    impact: diff * (weights[m] || 0.1)
                });
            }
        });
        
        normA = Math.sqrt(normA);
        normB = Math.sqrt(normB);
        
        let dist = 1.0;
        if (normA > 0 && normB > 0) {
           let sim = dot / (normA * normB);
           if (sim > 1) sim = 1;
           if (sim < -1) sim = -1;
           dist = 1.0 - sim;
        }
        
        gaps.sort((a, b) => b.impact - a.impact);
        
        return {
            target: targetClass,
            dist: parseFloat(dist.toFixed(4)),
            gaps: gaps.slice(0, 3)
        };
    }

    function updateSimAnalysis(a) {
        const sim = a.sim_data;
        const container = document.getElementById('simContent');
        const emptyMsg = document.getElementById('simEmpty');
        
        if(!sim || !sim.target) {
            if(container) container.style.display = 'none';
            if(emptyMsg) emptyMsg.style.display = 'block';
            return;
        }
        
        if(container) container.style.display = 'block';
        if(emptyMsg) emptyMsg.style.display = 'none';
        
        const tClass = document.getElementById('targetClass');
        if(tClass) tClass.textContent = sim.target;
        
        const tDist = document.getElementById('targetDist');
        if(tDist) tDist.textContent = sim.dist;
        
        const list = document.getElementById('gapList');
        if(list) {
            list.innerHTML = '';
            
            if(sim.gaps && sim.gaps.length > 0) {
                sim.gaps.forEach(g => {
                    const el = document.createElement('div');
                    el.className = 'check-item';
                    el.innerHTML = `
                        <span class="check-icon fail">⚠️</span>
                        <span>Mejorar <strong>${g.metric}</strong> en <span style="color:#e74c3c">+${g.diff.toFixed(2)}</span> (Obj: ${g.target.toFixed(2)})</span>
                    `;
                    list.appendChild(el);
                });
            } else {
                list.innerHTML = '<div class="check-item"><span class="check-icon pass">✅</span><span>Cumple con los requisitos de la siguiente categoría.</span></div>';
            }
        }
    }

    function resetChart(divId) {
        const el = document.getElementById(divId);
        if(!el) return;
        
        // For Radar (Polar) and Bar Chart
        if(divId === 'radarChart' || divId === 'barChart') {
            // Re-render to ensure clean state (animation replay + view reset)
            updateRadarChart();
            return;
        }
        
        // For Trend Chart (Cartesian)
        Plotly.relayout(el, {
            'xaxis.autorange': true,
            'yaxis.autorange': true
        });
    }

    function toggleMultiSelect(e) {
        e.stopPropagation();
        const dd = document.getElementById('msDropdown');
        const trigger = document.getElementById('msTrigger');
        dd.classList.toggle('open');
        trigger.classList.toggle('open');
        if (dd.classList.contains('open')) {
            const si = document.getElementById('msSearchInput');
            if (si) { si.value = ''; filterCompareOptions(''); si.focus(); }
        }
    }

    function filterCompareOptions(text) {
        const items = document.querySelectorAll('.ms-option');
        text = text.toLowerCase();
        items.forEach(item => {
            const name = item.dataset.name || '';
            item.style.display = name.includes(text) ? 'flex' : 'none';
        });
    }

    function toggleCompareAgent(agentId, event) {
        if (event) event.stopPropagation();
        // Prevent selecting the agent that is currently active in the ranking
        const activeItem = document.querySelector('.agent-item.active');
        const activeId = activeItem ? parseInt(activeItem.dataset.id) : null;
        if (agentId === activeId) return;
        
        const idx = selectedCompareIds.indexOf(agentId);
        if (idx > -1) {
            selectedCompareIds.splice(idx, 1);
        } else {
            if (selectedCompareIds.length >= 8) return;
            selectedCompareIds.push(agentId);
        }
        renderCompareTags();
        updateCompareCheckboxes();
        updateRadarChart();
        updateTrendChart();
    }

    function clearCompareSelection() {
        selectedCompareIds = [];
        renderCompareTags();
        updateCompareCheckboxes();
        updateRadarChart();
        updateTrendChart();
    }

    function renderCompareTags() {
        const trigger = document.getElementById('msTrigger');
        const ph = document.getElementById('msPlaceholder');
        const bar = document.getElementById('msSelectedBar');
        if (!trigger || !ph) return;
        
        // Clear old count from trigger
        const oldCount = trigger.querySelector('.ms-count');
        if (oldCount) oldCount.remove();
        
        // Clear tags from bar
        if (bar) bar.innerHTML = '';
        
        // Find the currently active (ranking-selected) agent
        const activeItem = document.querySelector('.agent-item.active');
        const activeId = activeItem ? parseInt(activeItem.dataset.id) : null;
        const activeAgent = activeId ? (displayedAgents.find(a => a.id_agente == activeId) || allAgents.find(a => a.id_agente == activeId)) : null;
        
        if (bar) {
            bar.style.display = 'flex';
            
            // 1. Always show ranking-selected agent as blue tag (non-removable)
            if (activeAgent) {
                const blueTag = document.createElement('span');
                blueTag.className = 'ms-tag';
                blueTag.style.background = '#2563eb';
                const aName = activeAgent.nombre_usuario_agente || activeAgent.id_agente;
                blueTag.innerHTML = `📌 ${aName}`;
                blueTag.title = 'Seleccionado del ranking';
                bar.appendChild(blueTag);
            }
            
            // 2. Render comparison agents in palette colors (removable)
            selectedCompareIds.forEach((id, i) => {
                const agent = displayedAgents.find(a => a.id_agente == id) || allAgents.find(a => a.id_agente == id);
                if (!agent) return;
                const color = compareColors[i % compareColors.length];
                const tag = document.createElement('span');
                tag.className = 'ms-tag';
                tag.style.background = color;
                const name = agent.nombre_usuario_agente || agent.id_agente;
                tag.innerHTML = `${name}<span class="ms-remove" onclick="event.stopPropagation(); toggleCompareAgent(${id})">×</span>`;
                bar.appendChild(tag);
            });
        }
        
        // Update trigger text
        if (selectedCompareIds.length === 0) {
            ph.style.display = '';
        } else {
            ph.style.display = 'none';
            const countEl = document.createElement('span');
            countEl.className = 'ms-count';
            countEl.textContent = `${selectedCompareIds.length} agente${selectedCompareIds.length > 1 ? 's' : ''}`;
            trigger.appendChild(countEl);
        }
    }

    function updateCompareCheckboxes() {
        document.querySelectorAll('.ms-option').forEach(opt => {
            const id = parseInt(opt.dataset.id);
            const cb = opt.querySelector('input[type="checkbox"]');
            const isSelected = selectedCompareIds.includes(id);
            if (cb) cb.checked = isSelected;
            opt.classList.toggle('selected', isSelected);
            const dot = opt.querySelector('.ms-color-dot');
            if (dot) {
                const idx = selectedCompareIds.indexOf(id);
                dot.style.background = idx > -1 ? compareColors[idx % compareColors.length] : '#e2e8f0';
            }
        });
    }

    function updateCompareSelect() {
        const optionsContainer = document.getElementById('msOptions');
        if (!optionsContainer) return;
        optionsContainer.innerHTML = '';
        
        // Find active ranking agent to exclude from dropdown
        const activeItem = document.querySelector('.agent-item.active');
        const activeId = activeItem ? parseInt(activeItem.dataset.id) : null;
        
        const sorted = [...displayedAgents].sort((a,b) => a.rank_global - b.rank_global);
        sorted.forEach(a => {
            if (a._noData) return;
            if (a.id_agente === activeId) return; // Skip the ranking-selected agent
            const div = document.createElement('div');
            div.className = 'ms-option';
            div.dataset.id = a.id_agente;
            div.dataset.name = (a.nombre_usuario_agente || '').toLowerCase();
            const isSelected = selectedCompareIds.includes(a.id_agente);
            if (isSelected) div.classList.add('selected');
            const colorIdx = selectedCompareIds.indexOf(a.id_agente);
            const dotColor = colorIdx > -1 ? compareColors[colorIdx % compareColors.length] : '#e2e8f0';
            div.innerHTML = `
                <input type="checkbox" ${isSelected ? 'checked' : ''}>
                <span class="ms-color-dot" style="background:${dotColor}"></span>
                <span class="ms-option-rank">#${a.rank_global}</span>
                <span class="ms-option-name">${a.nombre_usuario_agente || a.id_agente}</span>
                <span class="agent-badge badge-${a.Clase}" style="font-size:9px; padding:1px 5px;">${a.Clase}</span>
            `;
            div.onclick = () => toggleCompareAgent(a.id_agente);
            optionsContainer.appendChild(div);
        });
        renderCompareTags();
    }

    function updateTopAgencies() {
        try {
            // Recalculate Metrics for range
        displayedAgents = allAgents.map(agent => {
            const mData = monthlyData ? monthlyData[agent.id_agente.toString()] : null;
            if (!mData) return { ...agent, _noData: true, total_depositos: 0, score_global: 0 };
            
            // Filter months
            const rangeData = mData; // Use all data
            
            if (rangeData.length === 0) {
                 return { ...agent, _noData: true, total_depositos: 0, score_global: 0, rank_global: 9999 };
            }
            
            // Aggregate
            // Sums (Keep dynamic sums for financial totals)
            const sums = {
                total_depositos: 0, total_retiros: 0, 
                calculo_ggr: 0, calculo_ngr: 0, calculo_comision: 0
            };
            
            rangeData.forEach(d => {
                sums.total_depositos += (d.total_depositos || 0);
                sums.total_retiros += (d.total_retiros || 0);
                sums.calculo_ggr += (d.calculo_ggr || 0);
                sums.calculo_ngr += (d.calculo_ngr || 0);
                sums.calculo_comision += (d.calculo_comision || 0);
            });
            
            // --- FIX: Use static metrics from Python backend instead of re-calculating from missing monthly columns ---
            // The following metrics are calculated on the full history in backend and are not present in monthlyData
            const result = { ...agent, ...sums, _noData: false };
            
            // Recalculate Dynamic Similarity
            result.sim_data = calculateSimilarityJS(result);
            
            // Use median players from range or sum? 
            const players = rangeData.map(d => d.active_players || 0);
            result.median_players = players.length ? Math.max(...players) : 0; 
            
            return result;
        });
        
        // Re-Rank based on Score (Descending)
        displayedAgents.sort((a, b) => {
            if (a._noData && !b._noData) return 1;
            if (!a._noData && b._noData) return -1;
            return b.score_global - a.score_global;
        });
        
        // Assign new Ranks
        displayedAgents.forEach((a, i) => {
            a.rank_global = i + 1;
        });
        
        // Update UI
        renderList(displayedAgents);
        updateCompareSelect();
        
        // Refresh current profile if selected
        const activeItem = document.querySelector('.agent-item.active');
        if(activeItem) {
            selectAgent(activeItem.dataset.id); 
        } else if (displayedAgents.length > 0) {
            selectAgent(displayedAgents[0].id_agente);
        }
        } catch (e) {
            console.error("Error in updateTopAgencies:", e);
            document.querySelector('header').insertAdjacentHTML('beforeend', `<div style="background:orange;color:white;font-size:10px;">Render Error: ${e.message}</div>`);
        }
    }

    function renderList(data) {
        listEl.innerHTML = '';
        data.forEach(a => {
            if (a._noData) return;
            
            const el = document.createElement('div');
            el.className = 'agent-item';
            el.dataset.id = a.id_agente;
            el.dataset.clase = a.Clase;
            el.dataset.name = (a.nombre_usuario_agente || a.id_agente).toString().toLowerCase();
            
            el.innerHTML = `
                <div style="display:flex; align-items:center; gap:12px; width:100%;">
                    <div class="agent-rank">#${a.rank_global}</div>
                    <div class="agent-name">${a.nombre_usuario_agente || a.id_agente}</div>
                    <div class="agent-badge badge-${a.Clase}">${a.Clase}</div>
                </div>
            `;
            el.onclick = () => selectAgent(a.id_agente);
            listEl.appendChild(el);
        });
        filterList();
    }

    function filterList() {
        const text = document.getElementById('searchInput').value.toLowerCase();
        const items = listEl.getElementsByClassName('agent-item');
        
        Array.from(items).forEach(item => {
            const id = item.dataset.id;
            const matchText = item.dataset.name.includes(text) || id.toString().includes(text);
            const matchClass = currentFilter === 'all' || item.dataset.clase === currentFilter;
            item.style.display = (matchText && matchClass) ? 'flex' : 'none';
        });
    }
    
    function filterClass(cls) {
        currentFilter = cls;
        document.querySelectorAll('.filter-btn').forEach(b => {
            b.className = 'filter-btn';
        });
        const activeBtn = document.getElementById('btn-' + cls);
        if(activeBtn) {
            activeBtn.classList.add('active');
            if (cls !== 'all') {
                const base = cls.charAt(0);
                activeBtn.classList.add('active-' + base);
            }
        }
        filterList();
    }

    function selectAgent(id) {
        const items = listEl.getElementsByClassName('agent-item');
        Array.from(items).forEach(i => i.classList.remove('active'));
        const activeItem = Array.from(items).find(i => i.dataset.id == id);
        if(activeItem) {
            activeItem.classList.add('active');
        }
        
        const a = displayedAgents.find(x => x.id_agente == id);
        if(!a) return;
        
        // Auto-remove this agent from compare list if it was there
        const compIdx = selectedCompareIds.indexOf(a.id_agente);
        if (compIdx > -1) selectedCompareIds.splice(compIdx, 1);
        updateCompareSelect();
        
        // Profile
        document.getElementById('pName').textContent = a.nombre_usuario_agente || 'Sin Nombre';
        document.getElementById('pId').textContent = a.id_agente;
        document.getElementById('pRank').innerHTML = '#' + a.rank_global;
        document.getElementById('pScore').textContent = a.score_global ? a.score_global.toFixed(2) : '0.00';
        
        // Badges
        const pClass = document.getElementById('pClassBadge');
        pClass.textContent = a.Clase;
        pClass.className = `status-badge badge-${a.Clase}`;
        
        const pRisk = document.getElementById('pRiskBadge');
        if(a.Risk_Safe == 1) {
            pRisk.textContent = 'Seguro';
            pRisk.className = 'status-badge status-safe';
        } else {
            pRisk.textContent = 'Riesgo';
            pRisk.className = 'status-badge status-risky';
        }
        
        // Metrics
        const fmt1 = (val) => val !== undefined ? val.toFixed(1) : '-';
        if(document.getElementById('valRen')) document.getElementById('valRen').textContent = fmt1(a.rentabilidad);
        if(document.getElementById('valVol')) document.getElementById('valVol').textContent = fmt1(a.volumen);
        if(document.getElementById('valFid')) document.getElementById('valFid').textContent = fmt1(a.fidelidad);
        if(document.getElementById('valEst')) document.getElementById('valEst').textContent = fmt1(a.estabilidad);
        if(document.getElementById('valCre')) document.getElementById('valCre').textContent = fmt1(a.crecimiento);
        if(document.getElementById('valCas')) document.getElementById('valCas').textContent = fmt1(a.eficiencia_casino);
        if(document.getElementById('valDepScore')) document.getElementById('valDepScore').textContent = fmt1(a.eficiencia_deportes); // Changed ID to avoid conflict with financial 'valDep'
        if(document.getElementById('valConv')) document.getElementById('valConv').textContent = fmt1(a.eficiencia_conversion);
        if(document.getElementById('valTen')) document.getElementById('valTen').textContent = fmt1(a.tendencia);
        if(document.getElementById('valDiv')) document.getElementById('valDiv').textContent = fmt1(a.diversificacion);
        if(document.getElementById('valCal')) document.getElementById('valCal').textContent = fmt1(a.calidad_jugadores);
        
        // Financials
        const fmt = new Intl.NumberFormat('en-US', { notation: "compact" });
        if(document.getElementById('valDep')) document.getElementById('valDep').textContent = fmt.format(a.total_depositos || 0);
        if(document.getElementById('valRet')) document.getElementById('valRet').textContent = fmt.format(a.total_retiros || 0);
        if(document.getElementById('valGGR')) document.getElementById('valGGR').textContent = fmt.format(a.calculo_ggr || 0);
        if(document.getElementById('valNGR')) document.getElementById('valNGR').textContent = fmt.format(a.calculo_ngr || 0);
        if(document.getElementById('valCom')) document.getElementById('valCom').textContent = fmt.format(a.calculo_comision || 0);

        renderCompareTags();
        updateRadarChart();
        updateTrendChart();
        updateMonthlyTable(a);
        updateSimAnalysis(a);
    }

    function renderRadar(a) {
        if (!a) return;
        
        try {
            // Updated to use global variable for robustness
            const weights = window.globalWeights || (typeof globalWeights !== 'undefined' ? globalWeights : {});
            
            // Check DOM Elements
            const radarDiv = document.getElementById('radarChart');
            const barDiv = document.getElementById('barChart');
            
            if (!radarDiv || !barDiv) {
                console.warn("Chart containers not found");
                return;
            }

            // Cleanup previous instances
            try { Plotly.purge(radarDiv); } catch(e){}
            try { Plotly.purge(barDiv); } catch(e){} 
        
            // Definitions for Metrics
            const metricKeys = [
                'rentabilidad', 'volumen', 'fidelidad', 'estabilidad', 
                'crecimiento', 'eficiencia_casino', 'eficiencia_deportes', 
                'eficiencia_conversion', 'tendencia', 'diversificacion', 'calidad_jugadores'
            ];
            
            // Display Names
            const displayNames = {
                'rentabilidad': 'Rentabilidad', 'volumen': 'Volumen', 'fidelidad': 'Fidelidad',
                'estabilidad': 'Estabilidad', 'crecimiento': 'Crecimiento', 'eficiencia_casino': 'Efic. Casino',
                'eficiencia_deportes': 'Efic. Deportes', 'eficiencia_conversion': 'Conversión',
                'tendencia': 'Tendencia', 'diversificacion': 'Diversificación', 'calidad_jugadores': 'Calidad'
            };

            let chartData = metricKeys.map(key => {
            const rawVal = a[key] || 0; // 0-10
            const weight = weights[key] || 0;
            const contribution = rawVal * weight;
            const maxContribution = 10 * weight;
            const pctWeight = (weight * 100).toFixed(0) + '%';
            
            return {
                key: key,
                name: displayNames[key] || key,
                rawVal: rawVal,
                weight: weight,
                contribution: contribution,
                maxContribution: maxContribution,
                labelRadar: displayNames[key] || key // Reverted: Labels without weight for stability
            };
        });

        // --- RADAR CHART (Profile 0-10) ---
        
        const radarMetrics = chartData.map(d => d.labelRadar);
        const radarValA = chartData.map(d => d.rawVal);
        
        // Close the loop
        radarMetrics.push(radarMetrics[0]);
        radarValA.push(radarValA[0]);
        
        const radarData = [{
            type: 'scatterpolar',
            r: radarValA,
            theta: radarMetrics,
            fill: 'toself',
            name: a.nombre_usuario_agente || 'Seleccionado',
            line: {color: '#2563eb', width: 3},
            marker: {size: 6, color: '#2563eb', line: {color: 'white', width: 1.5}},
            // SOFTER COLOR: Reduced opacity for cleaner look
            fillcolor: 'rgba(59, 130, 246, 0.35)', // More vibrant for "atracción visual"
            // Enhanced HTML Tooltip
            hovertemplate: 
                '<b>%{theta}</b><br>' +
                'Score: %{r:.1f} / 10<br>' +
                'Peso: %{customdata[0]:.0%}<br>' +
                'Impacto: %{customdata[1]:.2f} pts<br>' +
                '<extra></extra>',
            customdata: chartData.map(d => [d.weight, d.contribution]).concat([[chartData[0].weight, chartData[0].contribution]]), // Close loop data
            showlegend: false
        }];

        // --- BAR CHART (Smart Analysis) ---
        // Display raw score (0-10) but explain the WEIGHTED OPPORTUNITY (Gap)
        
        // Clone for sorting
        let barDataSorted = [...chartData].sort((a, b) => b.rawVal - a.rawVal);
        
        const barLabels = barDataSorted.map(d => d.name);
        barLabels.reverse();
        
        const barValues = barDataSorted.map(d => d.rawVal).reverse();
        const barMax = barDataSorted.map(d => 10).reverse();
        
        // Calculate Weighted Gap (Opportunity)
        // (10 - Score) * Weight = Points missing from global score
        const barMeta = barDataSorted.map(d => {
            const gap = 10 - d.rawVal;
            const weightedGap = gap * d.weight;
            return [d.weight, weightedGap];
        }).reverse();
        
        // Trace 1: Grey Background
        const traceMax = {
            x: barMax,
            y: barLabels,
            type: 'bar',
            orientation: 'h',
            marker: { color: '#f1f5f9', cornerradius: 4 },
            hoverinfo: 'skip',
            showlegend: false
        };
        
        // Trace 2: Filled Bar (Earned Score)
        const traceEarned = {
            x: barValues,
            y: barLabels,
            type: 'bar',
            orientation: 'h',
            name: 'Puntaje',
            marker: { color: '#2563eb', cornerradius: 4 },
            text: barValues.map(v => v.toFixed(1)),
            textposition: 'auto',
            textfont: { family: 'Inter', color: 'white', size: 10 },
            hovertemplate: 
                '<b>%{y}</b><br>' +
                'Score: %{x:.1f} / 10<br>' +
                'Peso: %{customdata[0]:.0%}<br>' +
                'Oportunidad: +%{customdata[1]:.2f} pts globales<br>' +
                '<extra></extra>',
            customdata: barMeta,
            showlegend: false
        };
        
        // Initial Bar Data (Single Agent)
        let barData = [traceMax, traceEarned];
        let currentBarMode = 'overlay';
        let chartHeight = 500;

        // --- MULTI-AGENT COMPARISON LOGIC ---
        
        if (typeof selectedCompareIds !== 'undefined' && selectedCompareIds.length > 0) {
            currentBarMode = 'group';
            chartHeight = 550; // Increase height for grouped bars
            
            // In comparison mode, remove background bar and show Legend
            traceEarned.name = a.nombre_usuario_agente || 'Seleccionado';
            traceEarned.showlegend = false; // Disable legend as requested
            barData = [traceEarned]; 
            
            const agentPool = (typeof allAgents !== 'undefined') ? allAgents : displayedAgents;
            
            selectedCompareIds.forEach((compId, idx) => {
                if (compId == a.id_agente) return;
                
                const compAgent = agentPool.find(ag => ag.id_agente == compId);
                if (!compAgent) return;
                
                // Radar Part
                const weights = window.globalWeights || (typeof globalWeights !== 'undefined' ? globalWeights : {}); 
                
                // Calculate detailed data for Radar tooltip
                const compChartData = metricKeys.map(key => {
                    const rawVal = compAgent[key] || 0;
                    const weight = weights[key] || 0;
                    const contribution = rawVal * weight;
                    return { key, rawVal, weight, contribution };
                });
                
                const compRawVals = compChartData.map(d => d.rawVal);
                compRawVals.push(compRawVals[0]); // Close loop
                
                // Custom Data for Radar: [Weight, Impact]
                const compRadarCustomData = compChartData.map(d => [d.weight, d.contribution]);
                compRadarCustomData.push([compChartData[0].weight, compChartData[0].contribution]); // Close loop
                
                const color = compareColors[idx % compareColors.length];
                const compName = compAgent.nombre_usuario_agente || `Agente ${compId}`;
                
                radarData.push({
                    type: 'scatterpolar',
                    r: compRawVals,
                    theta: radarMetrics,
                    fill: 'toself', 
                    fillcolor: hexToRgba(color, 0.25),
                    name: compName,
                    line: {color: color, width: 2, dash: 'dot'},
                    marker: {size: 4, color: color},
                    hovertemplate: 
                        `<b>${compName}</b><br>` +
                        '%{theta}<br>' +
                        'Score: %{r:.1f} / 10<br>' +
                        'Peso: %{customdata[0]:.0%}<br>' +
                        'Impacto: %{customdata[1]:.2f} pts<br>' +
                        '<extra></extra>',
                    customdata: compRadarCustomData,
                    showlegend: false 
                });

                // Bar Part - Map values to match the sorted order of the main agent
                const compBarValues = barDataSorted.map(d => compAgent[d.key] || 0).reverse();
                
                // Calculate detailed data for Bar tooltip (Gap Analysis)
                const compBarMeta = barDataSorted.map(d => {
                    const rawVal = compAgent[d.key] || 0;
                    const gap = 10 - rawVal;
                    const weightedGap = gap * d.weight;
                    return [d.weight, weightedGap];
                }).reverse();
                
                barData.push({
                    x: compBarValues,
                    y: barLabels,
                    type: 'bar',
                    orientation: 'h',
                    name: compName,
                    marker: { color: color, cornerradius: 4 },
                    text: compBarValues.map(v => v.toFixed(1)),
                    textposition: 'auto',
                    textfont: { family: 'Inter', color: 'white', size: 10 },
                    hovertemplate: 
                        `<b>${compName}</b><br>` +
                        '%{y}<br>' +
                        'Score: %{x:.1f} / 10<br>' +
                        'Peso: %{customdata[0]:.0%}<br>' +
                        'Oportunidad: +%{customdata[1]:.2f} pts globales<br>' +
                        '<extra></extra>',
                    customdata: compBarMeta,
                    showlegend: false
                });
            });
        }

        const radarLayout = {
            polar: {
                radialaxis: { visible: true, range: [0, 10], showticklabels: false, ticks: '' },
                angularaxis: { tickfont: { size: 11, color: '#64748b', family: 'Inter', weight: 600 } },
                gridshape: 'linear',
                domain: { x: [0, 1], y: [0, 1] }
            },
            margin: { t: 80, b: 80, l: 80, r: 80 },
            autosize: true,
            showlegend: false,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        };

        const barLayout = {
            margin: { t: 10, b: 30, l: 110, r: 20 }, 
            barmode: currentBarMode,
            xaxis: { 
                range: [0, 10.5], 
                showgrid: true, gridcolor: '#f1f5f9', zeroline: false, 
                showticklabels: true, tickfont: { size: 10, color: '#94a3b8' },
                fixedrange: true 
            },
            yaxis: { 
                tickfont: { size: 11, color: '#0f172a', family: 'Inter' },
                automargin: true, fixedrange: true
            },
            legend: { orientation: 'h', y: -0.2, font: { size: 10 } },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            showlegend: false, // Always hide legend as requested for multi-agent view
            height: chartHeight
        };
        
        Plotly.newPlot('radarChart', radarData, radarLayout, {displayModeBar: false, responsive: true});
        
        // Resize container to match strict chart height (prevents overflow)
        const barContainer = document.getElementById('barChart');
        if(barContainer) barContainer.style.height = chartHeight + 'px';
        
        Plotly.newPlot('barChart', barData, barLayout, {displayModeBar: false, responsive: true});
        
        
        } catch (e) {
            console.error("Error rendering charts:", e);
            document.getElementById('radarChart').innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:red;">Error: ${e.message}</div>`;
        }
    }


    function updateRadarChart() {

        const activeItem = document.querySelector('.agent-item.active');
        if(!activeItem) return;
        const id = activeItem.dataset.id;
        const agent = displayedAgents.find(a => a.id_agente == id);
        if(agent) renderRadar(agent);
    }

    function updateTrendChart() {
        const activeItem = document.querySelector('.agent-item.active');
        if(!activeItem) return;
        const id = activeItem.dataset.id;
        const agent = displayedAgents.find(a => a.id_agente == id);
        // Force re-render even if data seems same, to handle display toggles
        if(agent) renderTrend(agent);
    }

    function exportTrendData() {
        const activeItem = document.querySelector('.agent-item.active');
        if(!activeItem) return;
        const id = activeItem.dataset.id;
        
        if(!monthlyData || !monthlyData[id]) { alert('No hay datos para exportar'); return; }
        
        const data = monthlyData[id];
        const headers = Object.keys(data[0]);
        const csvContent = "data:text/csv;charset=utf-8," 
            + headers.join(",") + "\\n" 
            + data.map(row => headers.map(fieldName => row[fieldName]).join(",")).join("\\n");
            
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `agent_${id}_trend_data.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    function renderTrend(a) {
        if (!a) return;
        
        const divId = 'trendChart';
        const chartDiv = document.getElementById(divId);
        if (!chartDiv) return;
        
        try { Plotly.purge(chartDiv); } catch(e){}

        const id = a.id_agente.toString();
        const select = document.getElementById('metricSelect');
        const metricKey = select.value;
        const metricName = select.options[select.selectedIndex].text;
        
        let data = [];
        let layout = {};
        
        if(monthlyData && monthlyData[id]) {
             let series = monthlyData[id];
             
             // Apply Date Filter - Removed
             
             series.sort((a, b) => a.month.localeCompare(b.month));
             
             const x_vals = series.map(d => d.month);
             const y_vals = series.map(d => d[metricKey] || 0);
             
             // Determine if in multi-agent mode
             const activeItem2 = document.querySelector('.agent-item.active');
             const activeId2 = activeItem2 ? parseInt(activeItem2.dataset.id) : null;
             const compareIds = selectedCompareIds.filter(cid => cid != activeId2);
             const isMulti = compareIds.length > 0;
             const agentName = a.nombre_usuario_agente || 'Seleccionado';

             data.push({
                 x: x_vals,
                 y: y_vals,
                 type: 'scatter',
                 mode: 'lines+markers',
                 name: isMulti ? agentName : metricName,
                 line: {color: '#2563eb', shape: 'spline', width: isMulti ? 3.5 : 3},
                 marker: {size: isMulti ? 7 : 8, color: '#2563eb', line: {color: 'white', width: 2}},
                 fill: isMulti ? 'tozeroy' : 'none',
                 fillcolor: 'rgba(37, 99, 235, 0.08)',
                 hovertemplate: '<b>%{x}</b><br>' + (isMulti ? agentName : metricName) + ': %{y:,.2f}<extra></extra>',
                 showlegend: !isMulti
             });
              
             // Linear Regression - only in single agent mode
             if (series.length > 1 && !isMulti) {
                 const x_nums = x_vals.map((_, i) => i);
                 const n = y_vals.length;
                 let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
                 for (let i = 0; i < n; i++) {
                     sumX += x_nums[i]; sumY += y_vals[i];
                     sumXY += x_nums[i] * y_vals[i]; sumXX += x_nums[i] * x_nums[i];
                 }
                 const m2 = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
                 const b2 = (sumY - m2 * sumX) / n;
                 const trend_y = x_nums.map(x => m2 * x + b2);
                 const mean_y = sumY / n;
                 let trendColor = '#94a3b8';
                 const total_change = trend_y[trend_y.length - 1] - trend_y[0];
                 const rel_change = mean_y !== 0 ? total_change / Math.abs(mean_y) : 0;
                 if (Math.abs(rel_change) < 0.05) trendColor = '#94a3b8';
                 else if (m2 > 0) trendColor = '#4ade80';
                 else trendColor = '#f87171';
                 data.push({
                     x: x_vals, y: trend_y, type: 'scatter', mode: 'lines',
                     name: `Tendencia (${m2 > 0 ? '\u2197' : (m2 < 0 ? '\u2198' : '\u2192')})`,
                     line: {color: trendColor, width: 2, dash: 'dash'}, hoverinfo: 'skip'
                 });
             }
              
             // === Multi-agent comparison traces ===
             compareIds.forEach((compId, idx) => {
                 const compIdStr = compId.toString();
                 if (!monthlyData || !monthlyData[compIdStr]) return;
                 const compAgent = displayedAgents.find(x => x.id_agente == compId) || allAgents.find(x => x.id_agente == compId);
                 if (!compAgent) return;
                 const color = compareColors[selectedCompareIds.indexOf(compId) % compareColors.length];
                 let compSeries = [...monthlyData[compIdStr]];
                 compSeries.sort((s1, s2) => s1.month.localeCompare(s2.month));
                 const comp_x = compSeries.map(d => d.month);
                 const comp_y = compSeries.map(d => d[metricKey] || 0);
                 const compName = compAgent.nombre_usuario_agente || compIdStr;
                 data.push({
                     x: comp_x, y: comp_y, type: 'scatter', mode: 'lines+markers',
                     name: compName,
                     line: { color: color, shape: 'spline', width: 2.5 },
                     marker: { size: 5, color: color, line: { color: 'white', width: 1.5 } },
                     fill: 'tozeroy',
                     fillcolor: hexToRgba(color, 0.05),
                     hovertemplate: '<b>%{x}</b><br>' + compName + ': %{y:,.2f}<extra></extra>',
                     showlegend: false
                 });
             });
              
             layout = {
                margin: { t: isMulti ? 15 : 30, b: 40, l: 55, r: 20 },
                xaxis: {
                    showgrid: false,
                    rangeslider: { visible: !isMulti, thickness: 0.08, bgcolor: '#fcfdfe', bordercolor: '#f1f5f9', borderwidth: 1 },
                    type: 'date', tickformat: '%b %Y',
                    tickfont: { size: 11, color: '#64748b', family: 'Inter' },
                    automargin: true, fixedrange: false
                },
                yaxis: { showgrid: true, gridcolor: '#f1f5f9', zeroline: false, fixedrange: false,
                    tickfont: { size: 10, color: '#94a3b8', family: 'Inter' } },
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                hovermode: 'x unified',
                legend: { orientation: 'h', y: 1.1, font: { size: 11, color: '#64748b', family: 'Inter' } },
                showlegend: !isMulti,
                dragmode: 'zoom',
                transition: { duration: 300, easing: 'cubic-in-out' },
                hoverlabel: { bgcolor: "rgba(255, 255, 255, 0.95)", bordercolor: "#e2e8f0", font: { family: "Inter", size: 12, color: "#1e293b" } }
            };
             
        } else {
             data.push({
                 x: ['Sin Datos Mensuales'],
                 y: [0],
                 type: 'bar',
                 marker: {color: '#bdc3c7'},
                 text: 'No hay historial',
                 textposition: 'auto'
             });
             layout = {
                margin: { t: 10, b: 40, l: 50, r: 10 },
                xaxis: { showgrid: false },
                yaxis: { showgrid: false },
             };
        }
        
        Plotly.newPlot('trendChart', data, layout, {
            displayModeBar: false, 
            responsive: true,
            displaylogo: false
        });
        
        updateMonthlyTable(a);
    }
    

    
    function updateMonthlyTable(a) {
        const tbody = document.getElementById('monthlyTableBody');
        const id = a.id_agente.toString();
        tbody.innerHTML = '';
        
        if(monthlyData && monthlyData[id]) {
            let series = monthlyData[id];
            
            // Date Filter Removed
            
            const sorted = [...series].sort((a, b) => b.month.localeCompare(a.month));
            
            if (sorted.length === 0) {
                 tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; padding:20px; color:#94a3b8;">No hay historial mensual disponible</td></tr>';
                 return;
            }

            sorted.forEach(d => {
                const tr = document.createElement('tr');
                
                // Safe accessors
                const players = d.active_players !== undefined ? d.active_players : 0;
                const score = d.score_global !== undefined ? d.score_global : 0;
                const deps = d.total_depositos !== undefined ? d.total_depositos : 0;
                const rets = d.total_retiros !== undefined ? d.total_retiros : 0;
                const ggr = d.calculo_ggr !== undefined ? d.calculo_ggr : 0;
                const ngr = d.calculo_ngr !== undefined ? d.calculo_ngr : 0;
                const comis = d.calculo_comision !== undefined ? d.calculo_comision : 0;
                
                tr.innerHTML = `
                    <td>${d.month}</td>
                    <td class="num-col">${new Intl.NumberFormat('en-US').format(players)}</td>
                    <td class="num-col"><strong>${score.toFixed(2)}</strong></td>
                    <td class="num-col">${new Intl.NumberFormat('en-US').format(deps)}</td>
                    <td class="num-col">${new Intl.NumberFormat('en-US').format(rets)}</td>
                    <td class="num-col">${new Intl.NumberFormat('en-US').format(ggr)}</td>
                    <td class="num-col">${new Intl.NumberFormat('en-US').format(ngr)}</td>
                    <td class="num-col"><strong>${new Intl.NumberFormat('en-US').format(comis)}</strong></td>
                `;
                tbody.appendChild(tr);
            });
        } else {
            tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; padding:20px; color:#94a3b8;">No hay datos disponibles para este agente</td></tr>';
        }
    }

    // Initial Load
    document.addEventListener('DOMContentLoaded', () => {
        updateTopAgencies();
    });
    
    // Window Resize Handler
    let resizeTimeout;
    window.addEventListener('resize', function() {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(function() {
            const radarDiv = document.getElementById('radarChart');
            const barDiv = document.getElementById('barChart');
            const trendDiv = document.getElementById('trendChart');
            
            if (radarDiv) Plotly.Plots.resize(radarDiv);
            if (barDiv) Plotly.Plots.resize(barDiv);
            if (trendDiv) Plotly.Plots.resize(trendDiv);
        }, 100);
    });
</script>
</body>
</html>
    """
    
    # 3. Renderizar y Guardar
    template = Template(template_str)
    html_content = template.render(
        total_agents=total_agents,
        class_counts=class_counts,
        pct_risky=pct_risky,
        agents_json=json.dumps(df_json),
        top_agent_json=json.dumps(top_agent.fillna(0).to_dict()),
        monthly_json=monthly_data_js,
        centroids_json=centroids_json,
        metrics_json=metrics_json,
        class_order_json=class_order_json,
        weights_json=weights_json,

    )
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Dashboard saved to {out_path}")

if __name__ == "__main__":
    try:
        # Test code - won't run in production
        pass
    except Exception as e:
        print(f"Error testing: {e}")
