import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json
from jinja2 import Template

def calculate_similarity(row, centroids, class_order, metrics):
    current_class = row['Clase']
    if current_class not in class_order:
        return None
        
    try:
        current_idx = class_order.index(current_class)
        # Find next better class (lower index)
        if current_idx == 0:
            return {"target": "Top", "dist": 0, "gaps": []}
            
        target_class = class_order[current_idx - 1]
        
        if target_class not in centroids:
            return None
            
        target_vals = centroids[target_class]
        
        # Calculate Distance (Euclidean)
        dist = 0
        gaps = []
        
        for m in metrics:
            curr_val = row[m]
            targ_val = target_vals[m]
            diff = targ_val - curr_val
            dist += diff ** 2
            
            if diff > 0: # Improvement needed
                gaps.append({"metric": m, "diff": diff, "target": targ_val, "current": curr_val})
        
        dist = np.sqrt(dist)
        
        # Sort gaps by impact (largest diff)
        gaps.sort(key=lambda x: x['diff'], reverse=True)
        
        return {
            "target": target_class,
            "dist": round(dist, 2),
            "gaps": gaps[:3] # Top 3 improvements
        }
    except:
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
        'Vi', 'Tx_i', 'Gi', 'Si', 
        'Punt_Crecimiento', 'Conv_i', 'Punt_Pareto', 
        'Freq_i', 'Punt_Productos', 'Ti'
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
    
    # Monthly Data for Trend Lines (Optional)
    monthly_data_js = "null"
    if df_monthly is not None:
        # Group by agent and convert to dict {agent_id: [{month, comision, depositos, ...}, ...]}
        # Include more metrics for detailed visualization
        monthly_cols = [
            'id_agente', 'month', 'calculo_comision', 'total_depositos', 'active_players', 'total_retiros', 'calculo_ggr',
            'ggr_deportiva', 'ggr_casino', 'total_apuesta_deportiva', 'total_apuesta_casino', 'calculo_ngr',
            # Scores mapped to internal names
            'score_global', 'Si', 'Vi', 'Gi', 'Ti', 'Tx_i', 'Freq_i', 'Conv_i',
            # Unmapped new scores
            'Punt_Crecimiento', 'Punt_Pareto', 'Punt_Productos', 'Punt_Rotación',
            # Metadata
            'Clase', 'Risk_Safe'
        ]
        # Check if columns exist
        available_cols = [c for c in monthly_cols if c in df_monthly.columns]
        
        monthly_min = df_monthly[available_cols].copy()
        monthly_min['month'] = monthly_min['month'].astype(str)
        monthly_dict = {}
        for agent_id, group in monthly_min.groupby('id_agente'):
            monthly_dict[str(agent_id)] = group.drop(columns=['id_agente']).to_dict(orient='records')
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
            background: var(--card-bg); padding: 15px 30px; 
            border-bottom: 1px solid var(--border); 
            display: flex; justify-content: space-between; align-items: center; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            position: sticky; top: 0; z-index: 100;
        }
        h1 { margin: 0; font-size: 22px; font-weight: 600; color: var(--primary); letter-spacing: -0.5px; }
        .header-meta { font-size: 14px; color: var(--text-muted); margin-top: 4px; font-weight: 500; }
        
        .kpi-bar { display: flex; gap: 30px; }
        .kpi { text-align: center; position: relative; }
        .kpi-val { font-size: 22px; font-weight: 700; line-height: 1.2; }
        .kpi-lbl { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); font-weight: 500; }
        .tag-a { color: var(--success); } .tag-b { color: var(--warning); } .tag-c { color: var(--danger); }
        
        /* Layout Grid */
        .grid { display: grid; grid-template-columns: 320px 1fr; gap: 25px; flex: 1; overflow: hidden; margin-top: 20px; }
        
        /* Sidebar */
        .sidebar { 
            background: var(--card-bg); border-radius: 12px; 
            display: flex; flex-direction: column; 
            box-shadow: var(--shadow); border: 1px solid var(--border);
            overflow: hidden;
        }
        .sidebar-header { 
            padding: 20px; 
            border-bottom: 1px solid var(--border); 
            background: #fafbfc; 
            display: flex;
            flex-direction: column;
            gap: 12px;
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
            padding: 12px 20px; border-bottom: 1px solid var(--border); cursor: pointer; 
            display: flex; justify-content: space-between; align-items: center; 
            transition: background 0.15s;
        }
        .agent-item:hover { background-color: #f8f9fa; }
        .agent-item.active { background-color: var(--accent-soft); border-left: 4px solid var(--accent); padding-left: 16px; }
        .agent-rank { font-size: 12px; font-weight: 600; color: var(--text-muted); width: 35px; }
        .agent-name { flex: 1; font-weight: 500; font-size: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--text-color); }
        .agent-badge { 
            font-size: 10px; font-weight: 700; padding: 3px 8px; border-radius: 12px; 
            min-width: 20px; text-align: center;
        }
        /* Generic Badge Styles using Attribute Selectors */
        [class*="badge-A"] { background: rgba(46, 204, 113, 0.15); color: #27ae60; }
        [class*="badge-B"] { background: rgba(241, 196, 15, 0.15); color: #d35400; }
        [class*="badge-C"] { background: rgba(231, 76, 60, 0.15); color: #c0392b; }

        /* Main Content */
        .main-content { overflow-y: auto; display: flex; flex-direction: column; gap: 60px; padding-right: 5px; }
        .main-content::-webkit-scrollbar { width: 6px; }
        .main-content::-webkit-scrollbar-thumb { background: #ccc; border-radius: 3px; }
        
        .card { 
            background: var(--card-bg); border-radius: 12px; padding: 25px; 
            box-shadow: var(--shadow); border: 1px solid var(--border);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .card:hover { box-shadow: var(--shadow-hover); }
        
        /* Profile Header */
        .profile-header { display: flex; justify-content: space-between; margin-bottom: 25px; border-bottom: 1px solid var(--border); padding-bottom: 20px; }
        .profile-name { font-size: 28px; font-weight: 700; margin: 0 0 5px 0; color: var(--primary); }
        .profile-id { font-family: monospace; color: var(--text-muted); font-size: 14px; background: #f0f2f5; padding: 2px 6px; border-radius: 4px; }
        .profile-score-val { font-size: 32px; font-weight: 800; color: var(--primary); letter-spacing: -1px; text-align: right; }
        .profile-score-lbl { font-size: 12px; color: var(--text-muted); text-transform: uppercase; text-align: right; letter-spacing: 1px; }
        
        .status-badge { display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
        .status-safe { background: #d4edda; color: #155724; }
        .status-risky { background: #f8d7da; color: #721c24; }
        
        /* Metrics Grid */
        .metrics-category { margin-bottom: 25px; }
        .metrics-title { 
            font-size: 14px; font-weight: 700; color: var(--text-color); margin-bottom: 12px; 
            display: flex; align-items: center; gap: 8px;
            padding-bottom: 8px; border-bottom: 1px dashed var(--border);
        }
        .metrics-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px; }
        .metric-box { 
            background: #fafbfc; padding: 15px 10px; border-radius: 8px; text-align: center; border: 1px solid #eee;
            transition: transform 0.2s;
        }
        .metric-box:hover { transform: translateY(-2px); border-color: var(--accent); }
        .metric-val { font-size: 18px; font-weight: 700; color: var(--primary); margin-bottom: 4px; }
        .metric-lbl { font-size: 10px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; }
        
        /* Charts Area */
        .radar-section {
            display: grid;
            grid-template-columns: 5fr 7fr; /* Optimized 5:7 split for better Bar legibility */
            gap: 32px; /* Increased spacing */
            margin-bottom: 0;
            align-items: start;
        }
        
        .radar-section .card {
            height: 100%;
            display: flex;
            flex-direction: column;
            padding: 24px; /* Increased padding */
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }

        .trend-section {
            width: 100%;
            margin-bottom: 0;
        }
        
        .chart-title { 
            font-size: 18px; /* Larger title */
            font-weight: 700; 
            letter-spacing: -0.025em;
            margin-bottom: 20px; 
            color: var(--primary); 
            display: flex; 
            align-items: center; 
            gap: 12px; 
            padding-bottom: 12px;
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
        table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #eee; }
        th { font-weight: 600; color: var(--text-muted); text-transform: uppercase; font-size: 11px; background: #fafbfc; }
        tr:last-child td { border-bottom: none; }
        .num-col { text-align: right; font-family: monospace; }
        
        /* Responsive */
        @media (max-width: 1400px) { 
            .radar-section { grid-template-columns: 1fr; } /* Stack earlier for better readability */
            .chart-title { font-size: 16px; }
        }
        @media (max-width: 1366px) { .grid { grid-template-columns: 260px 1fr; } }
        @media (max-width: 1100px) { 
            .grid { grid-template-columns: 220px 1fr; } 
            .metrics-grid { grid-template-columns: repeat(3, 1fr); }
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
            flex-wrap: wrap; max-width: 280px;
        }
        .ms-trigger:hover { border-color: #cbd5e1; }
        .ms-trigger.open { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
        .ms-placeholder { color: var(--text-muted); font-size: 12px; white-space: nowrap; }
        .ms-tag {
            display: inline-flex; align-items: center; gap: 3px;
            padding: 1px 7px; border-radius: 10px; font-size: 10px;
            font-weight: 600; color: white; white-space: nowrap;
            animation: fadeIn 0.2s ease-out;
        }
        .ms-tag .ms-remove { cursor: pointer; opacity: 0.7; font-size: 12px; margin-left: 2px; }
        .ms-tag .ms-remove:hover { opacity: 1; }
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
                    <div class="metric-box"><div class="metric-val" id="valSi">-</div><div class="metric-lbl">Estabilidad</div></div>
                    <div class="metric-box"><div class="metric-val" id="valVi">-</div><div class="metric-lbl">Rentabilidad</div></div>
                    <div class="metric-box"><div class="metric-val" id="valGi">-</div><div class="metric-lbl">Fidelidad</div></div>
                    <div class="metric-box"><div class="metric-val" id="valTi">-</div><div class="metric-lbl">Tendencia</div></div>
                    <div class="metric-box"><div class="metric-val" id="valTx">-</div><div class="metric-lbl">Volumen</div></div>
                    <div class="metric-box"><div class="metric-val" id="valConv">-</div><div class="metric-lbl">Conversión</div></div>
                    <div class="metric-box"><div class="metric-val" id="valCre">-</div><div class="metric-lbl">Crecimiento</div></div>
                    <div class="metric-box"><div class="metric-val" id="valPar">-</div><div class="metric-lbl">Pareto</div></div>
                    <div class="metric-box"><div class="metric-val" id="valRot">-</div><div class="metric-lbl">Rotación</div></div>
                    <div class="metric-box"><div class="metric-val" id="valProd">-</div><div class="metric-lbl">Productos</div></div>
                </div>
            </div>
        </div>
        
        <!-- Charts -->
        <div class="radar-section">
            <div class="card">
                <div class="chart-title">
                    <span style="font-size:24px; background:var(--bg-color); padding:8px; border-radius:8px;">🕸️</span> 
                    <div>
                        <div style="line-height:1.2;">Análisis Radar</div>
                        <div style="font-size:12px; color:var(--text-muted); font-weight:400; margin-top:2px;">Perfil multidimensional del agente</div>
                    </div>
                    <div style="margin-left:auto; display:flex; align-items:center; gap:6px;">
                        <div class="ms-container" id="msContainer">
                            <div class="ms-trigger" id="msTrigger" onclick="toggleMultiSelect(event)">
                                <span class="ms-placeholder" id="msPlaceholder">+ Comparar agentes...</span>
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
                <div id="radarChart" style="height: 450px; width: 100%;"></div>
            </div>
            <div class="card">
                <div class="chart-title">
                    <span style="font-size:24px; background:var(--bg-color); padding:8px; border-radius:8px;">📊</span> 
                    <div>
                        <div style="line-height:1.2;">Métricas (Porcentaje)</div>
                        <div style="font-size:12px; color:var(--text-muted); font-weight:400; margin-top:2px;">Desglose detallado vs Benchmark</div>
                    </div>
                    <button class="reset-btn" onclick="resetChart('barChart')" title="Reiniciar gráfico" style="margin-left:auto;">
                        ↺
                    </button>
                </div>
                <div id="barChart" style="height: 450px; width: 100%;"></div>
            </div>
        </div>

        <div class="trend-section">
            <div class="card">
                <div class="chart-title">
                    <span style="font-size:20px;">📈</span> Tendencia Histórica
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
                <div id="trendChart" style="height: 450px;"></div>
            </div>
        </div>
        
        <!-- Monthly Table -->
        <div class="card">
            <div class="chart-title">📅 Detalle Mensual</div>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Mes</th>
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
            <div class="chart-title">🎯 Análisis de Mejora (Próximo Nivel)</div>
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
    
    const listEl = document.getElementById('agentList');
    let currentFilter = 'all';
    let selectedCompareIds = [];
    const compareColors = ['#7c3aed','#f59e0b','#10b981','#ef4444','#06b6d4','#ec4899','#84cc16','#f97316'];

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
        // 0 is best (A+++)
        if (currentIdx === 0) return { target: "Top", dist: 0, gaps: [] };
        
        const targetClass = classOrder[currentIdx - 1];
        if (!centroids[targetClass]) return null;
        
        const targetVals = centroids[targetClass];
        let dist = 0;
        const gaps = [];
        
        simMetrics.forEach(m => {
            const currVal = agent[m] || 0;
            const targVal = targetVals[m] || 0;
            const diff = targVal - currVal;
            dist += diff * diff;
            
            if (diff > 0) {
                gaps.push({ metric: m, diff: diff, target: targVal, current: currVal });
            }
        });
        
        dist = Math.sqrt(dist);
        gaps.sort((a, b) => b.diff - a.diff);
        
        return {
            target: targetClass,
            dist: parseFloat(dist.toFixed(2)),
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
    }

    function clearCompareSelection() {
        selectedCompareIds = [];
        renderCompareTags();
        updateCompareCheckboxes();
        updateRadarChart();
    }

    function renderCompareTags() {
        const trigger = document.getElementById('msTrigger');
        const ph = document.getElementById('msPlaceholder');
        if (!trigger || !ph) return;
        trigger.querySelectorAll('.ms-tag').forEach(t => t.remove());
        if (selectedCompareIds.length === 0) {
            ph.style.display = '';
        } else {
            ph.style.display = 'none';
            selectedCompareIds.forEach((id, i) => {
                const agent = displayedAgents.find(a => a.id_agente == id) || allAgents.find(a => a.id_agente == id);
                if (!agent) return;
                const color = compareColors[i % compareColors.length];
                const tag = document.createElement('span');
                tag.className = 'ms-tag';
                tag.style.background = color;
                const name = agent.nombre_usuario_agente || agent.id_agente;
                const shortName = String(name).length > 14 ? String(name).substring(0, 14) + '…' : name;
                tag.innerHTML = `${shortName}<span class="ms-remove" onclick="event.stopPropagation(); toggleCompareAgent(${id})">×</span>`;
                trigger.insertBefore(tag, ph);
            });
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
        const sorted = [...displayedAgents].sort((a,b) => a.rank_global - b.rank_global);
        sorted.forEach(a => {
            if (a._noData) return;
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
        // Date Filter Removed - Using full range
        
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
            // Sums
            const sums = {
                total_depositos: 0, total_retiros: 0, 
                calculo_ggr: 0, calculo_ngr: 0, calculo_comision: 0
            };
            // Averages
            const avgs = {
                score_global: [], Si: [], Vi: [], Gi: [], Ti: [], 
                Tx_i: [], Freq_i: [], Conv_i: [],
                Punt_Crecimiento: [], Punt_Pareto: [], Punt_Productos: []
            };
            
            rangeData.forEach(d => {
                sums.total_depositos += (d.total_depositos || 0);
                sums.total_retiros += (d.total_retiros || 0);
                sums.calculo_ggr += (d.calculo_ggr || 0);
                sums.calculo_ngr += (d.calculo_ngr || 0);
                sums.calculo_comision += (d.calculo_comision || 0);
                
                if(d.score_global !== undefined) avgs.score_global.push(d.score_global);
                if(d.Si !== undefined) avgs.Si.push(d.Si);
                if(d.Vi !== undefined) avgs.Vi.push(d.Vi);
                if(d.Gi !== undefined) avgs.Gi.push(d.Gi);
                if(d.Ti !== undefined) avgs.Ti.push(d.Ti);
                if(d.Tx_i !== undefined) avgs.Tx_i.push(d.Tx_i);
                if(d.Freq_i !== undefined) avgs.Freq_i.push(d.Freq_i);
                if(d.Conv_i !== undefined) avgs.Conv_i.push(d.Conv_i);
                if(d.Punt_Crecimiento !== undefined) avgs.Punt_Crecimiento.push(d.Punt_Crecimiento);
                if(d.Punt_Pareto !== undefined) avgs.Punt_Pareto.push(d.Punt_Pareto);
                if(d.Punt_Productos !== undefined) avgs.Punt_Productos.push(d.Punt_Productos);
            });
            
            const result = { ...agent, ...sums, _noData: false };
            
            // Update Class/Risk from last month of range
            if(rangeData.length > 0) {
                 const last = rangeData[rangeData.length - 1];
                 if(last.Clase) {
                     result.Clase = last.Clase;
                     // Simple Risk Logic: A/B = Safe (1), others = Risky (0)
                     const isSafe = result.Clase.startsWith('A') || result.Clase.startsWith('B');
                     result.Risk_Safe = isSafe ? 1 : 0;
                 }
            }
            
            // Calc Averages
            for (const [key, vals] of Object.entries(avgs)) {
                if (vals.length > 0) {
                    result[key] = vals.reduce((a, b) => a + b, 0) / vals.length;
                } else {
                    result[key] = 0;
                }
            }
            
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
        if(document.getElementById('valSi')) document.getElementById('valSi').textContent = fmt1(a.Si);
        if(document.getElementById('valVi')) document.getElementById('valVi').textContent = fmt1(a.Vi);
        if(document.getElementById('valGi')) document.getElementById('valGi').textContent = fmt1(a.Gi) + '%';
        if(document.getElementById('valTi')) document.getElementById('valTi').textContent = fmt1(a.Ti);
        if(document.getElementById('valTx')) document.getElementById('valTx').textContent = fmt1(a.Tx_i);
        if(document.getElementById('valConv')) document.getElementById('valConv').textContent = fmt1(a.Conv_i);
        
        // Financials
        const fmt = new Intl.NumberFormat('en-US', { notation: "compact" });
        if(document.getElementById('valDep')) document.getElementById('valDep').textContent = fmt.format(a.total_depositos || 0);
        if(document.getElementById('valRet')) document.getElementById('valRet').textContent = fmt.format(a.total_retiros || 0);
        if(document.getElementById('valGGR')) document.getElementById('valGGR').textContent = fmt.format(a.calculo_ggr || 0);
        if(document.getElementById('valNGR')) document.getElementById('valNGR').textContent = fmt.format(a.calculo_ngr || 0);
        if(document.getElementById('valCom')) document.getElementById('valCom').textContent = fmt.format(a.calculo_comision || 0);
        
        if(document.getElementById('valCre')) document.getElementById('valCre').textContent = fmt1(a.Punt_Crecimiento);
        if(document.getElementById('valPar')) document.getElementById('valPar').textContent = fmt1(a.Punt_Pareto);
        if(document.getElementById('valRot')) document.getElementById('valRot').textContent = fmt1(a.Freq_i) + '%';
        if(document.getElementById('valProd')) document.getElementById('valProd').textContent = fmt1(a.Punt_Productos);

        updateRadarChart();
        updateTrendChart();
        updateMonthlyTable(a);
        updateSimAnalysis(a);
    }

    function renderRadar(a) {
        const metrics = [
            'Rentabilidad (Vi)', 'Volumen (Tx)', 'Fidelidad (Gi)', 'Estabilidad (Si)', 
            'Crecimiento', 'Eficiencia (Conv)', 'Pareto', 'Rotación (Freq)', 
            'Productos', 'Tendencia (Ti)'
        ];
        
        const metricKeys = [
            'Vi', 'Tx_i', 'Gi', 'Si', 
            'Punt_Crecimiento', 'Conv_i', 'Punt_Pareto', 
            'Freq_i', 'Punt_Productos', 'Ti'
        ];
        
        // Find dynamic Top #1 for the current period
        const dynamicTop = displayedAgents.find(x => x.rank_global === 1) || staticTopAgent;
        
        const s = (obj, key) => (obj[key] || 0);
        const valA = metricKeys.map(k => s(a, k));
        const valTop = metricKeys.map(k => s(dynamicTop, k));
        
        // --- RADAR DATA ---
        const radarData = [
            {
                type: 'scatterpolar',
                r: valTop,
                theta: metrics,
                fill: 'toself',
                name: `Top #1 (${dynamicTop.nombre_usuario_agente || 'Global'})`,
                line: {color: '#94a3b8', width: 1, dash: 'dot'},
                fillcolor: 'rgba(148, 163, 184, 0.15)',
                hoverinfo: 'text',
                text: valTop.map((v, i) => `${metrics[i]}: ${v.toFixed(1)}`)
            },
            {
                type: 'scatterpolar',
                r: valA,
                theta: metrics,
                fill: 'toself',
                name: 'Seleccionado',
                line: {color: '#2563eb', width: 2},
                fillcolor: 'rgba(37, 99, 235, 0.2)',
                hoverinfo: 'text',
                text: valA.map((v, i) => `${metrics[i]}: ${v.toFixed(1)}`)
            }
        ];

        // --- BAR DATA (Storytelling Redesign) ---
        
        // 1. Prepare Data & Sort by Gap (Narrative)
        // Sort: Best (Positive Gap) at Bottom -> Worst (Negative Gap) at Top
        let chartData = metrics.map((m, i) => {
            const vA = valA[i] * 10; // 0-100 scale
            const vT = valTop[i] * 10;
            const diff = vA - vT;
            return {
                metric: m,
                valA: vA,
                valTop: vT,
                diff: diff,
                origIndex: i
            };
        });

        // Sort: Descending Gap (Positive -> Negative)
        // Plotly plots index 0 at bottom, so Positive (Best) will be at bottom, Negative (Priorities) at Top.
        chartData.sort((a, b) => b.diff - a.diff);

        const sortedMetrics = chartData.map(d => d.metric);
        const sortedValA = chartData.map(d => d.valA);
        const sortedValTop = chartData.map(d => d.valTop);
        const sortedDiff = chartData.map(d => d.diff);

        // 2. Generate Insight Text
        const topStrengths = chartData.filter(d => d.diff > 0).slice(0, 3).map(d => d.metric); // Top 3 positives
        const topPriorities = chartData.filter(d => d.diff < 0).slice(-3).reverse().map(d => d.metric); // Bottom 3 negatives (worst first)

        let narrativeText = "";
        if (topStrengths.length > 0) narrativeText += `<span style='color:#16a34a'><b>Fortalezas:</b> ${topStrengths.join(', ')}</span>`;
        if (topStrengths.length > 0 && topPriorities.length > 0) narrativeText += "  |  ";
        if (topPriorities.length > 0) narrativeText += `<span style='color:#dc2626'><b>Prioridades:</b> ${topPriorities.join(', ')}</span>`;
        if (!narrativeText) narrativeText = "Desempeño alineado con Benchmark";

        const barData = [];

        // Trace 1: Background Track
        barData.push({
            y: sortedMetrics,
            x: sortedMetrics.map(() => 100),
            type: 'bar',
            orientation: 'h',
            marker: { color: '#f8fafc', line: { width: 0 }, cornerradius: 5 },
            hoverinfo: 'skip',
            showlegend: false
        });

        // Trace 2: Selected Agent
        barData.push({
            y: sortedMetrics,
            x: sortedValA,
            type: 'bar',
            orientation: 'h',
            name: 'Seleccionado',
            marker: { color: '#2563eb', cornerradius: 5 },
            text: sortedValA.map(v => v.toFixed(1) + '%'),
            textposition: 'auto',
            textfont: { family: 'Inter', color: 'white' }, 
            hovertemplate: '<b>%{y}</b><br>Valor: %{x:.1f}%<extra></extra>'
        });

        // Trace 3: Delta Labels (Right Axis)
        const deltaText = sortedDiff.map(d => {
            const sign = d > 0 ? '+' : '';
            const color = d >= 0 ? '#16a34a' : '#dc2626';
            return `<span style="color:${color}; font-weight:600;">Δ ${sign}${d.toFixed(1)} pp</span>`;
        });

        barData.push({
            y: sortedMetrics,
            x: sortedMetrics.map(() => 115), 
            type: 'scatter',
            mode: 'text',
            text: deltaText,
            textposition: 'middle left',
            textfont: { family: 'Inter', size: 12 },
            hoverinfo: 'skip',
            showlegend: false
        });

        // Trace 4: Benchmark Markers (Invisible for hover)
        barData.push({
            y: sortedMetrics,
            x: sortedValTop,
            type: 'scatter',
            mode: 'markers',
            name: 'Benchmark',
            marker: { opacity: 0, size: 20 },
            hoverinfo: 'name+x'
        });

        // Handle Multiple Comparison Agents
        selectedCompareIds.forEach((compId, idx) => {
            const compAgent = displayedAgents.find(x => x.id_agente == compId) || allAgents.find(x => x.id_agente == compId);
            if (!compAgent) return;
            const color = compareColors[idx % compareColors.length];
            const valComp = metricKeys.map(k => s(compAgent, k));

            // Add to Radar
            radarData.push({
                type: 'scatterpolar',
                r: valComp,
                theta: metrics,
                fill: 'toself',
                name: compAgent.nombre_usuario_agente || 'Comparado',
                line: {color: color, width: 2},
                fillcolor: hexToRgba(color, 0.15),
                hoverinfo: 'text',
                text: valComp.map((v, i) => `${metrics[i]}: ${v.toFixed(1)}`)
            });

            // Add to Bar (Sorted)
            const sortedValComp = chartData.map(d => valComp[d.origIndex] * 10);

            barData.push({
                y: sortedMetrics,
                x: sortedValComp,
                type: 'scatter',
                mode: 'markers',
                name: compAgent.nombre_usuario_agente || 'Comparado',
                marker: {
                    symbol: 'diamond',
                    size: 10,
                    color: color,
                    line: { width: 1, color: '#ffffff' }
                },
                hoverinfo: 'x+name'
            });
        });

        // Shapes for Benchmark Lines
        const shapes = [];
        sortedValTop.forEach((val, i) => {
            shapes.push({
                type: 'line',
                x0: val, x1: val,
                y0: i - 0.4, y1: i + 0.4,
                xref: 'x', yref: 'y',
                line: { color: '#94a3b8', width: 2, dash: 'dot' }
            });
        });

        const radarLayout = {
            polar: {
                radialaxis: { visible: true, range: [0, 10], showticklabels: false, ticks: '' },
                angularaxis: { tickfont: { size: 11, color: '#64748b', family: 'Inter, sans-serif' } }
            },
            margin: { t: 40, b: 40, l: 60, r: 60 },
            showlegend: true,
            legend: { orientation: 'h', y: -0.15, xanchor: 'center', x: 0.5 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            transition: { duration: 300, easing: 'cubic-in-out' },
            height: 450
        };

        const barLayout = {
            title: {
                text: narrativeText,
                x: 0,
                y: 0.98,
                xanchor: 'left',
                yanchor: 'top',
                font: { family: 'Inter', size: 13, color: '#0f172a' },
                pad: { b: 10 }
            },
            margin: { t: 60, b: 20, l: 140, r: 80 }, 
            barmode: 'overlay',
            xaxis: { range: [0, 130], showgrid: false, zeroline: false, showticklabels: false, fixedrange: true },
            yaxis: { 
                tickfont: { size: 12, color: '#0f172a', family: 'Inter' },
                gridcolor: 'rgba(0,0,0,0)',
                automargin: true,
                fixedrange: true
            },
            shapes: shapes,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            height: 450,
            showlegend: false,
            bargap: 0.4
        };
        
        Plotly.newPlot('radarChart', radarData, radarLayout, {displayModeBar: false, responsive: true});
        Plotly.newPlot('barChart', barData, barLayout, {displayModeBar: false, responsive: true});
        
        // Sync interactions
        const radarDiv = document.getElementById('radarChart');
        const barDiv = document.getElementById('barChart');
        
        if(radarDiv && barDiv) {
            radarDiv.on('plotly_click', function(data){
                // Interaction logic if needed
            });
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
             
             data.push({
                 x: x_vals,
                 y: y_vals,
                 type: 'scatter',
                mode: 'lines+markers',
                name: metricName,
                line: {color: '#60a5fa', shape: 'spline', width: 3},
                marker: {size: 8, color: '#2563eb', line: {color: 'white', width: 2}},
                hovertemplate: '<b>%{x}</b><br>' + metricName + ': %{y:,.2f}<extra></extra>'
            });
             
             // Linear Regression
             if (series.length > 1) {
                 const x_nums = x_vals.map((_, i) => i);
                 
                 // Calculate linear regression
                 const n = y_vals.length;
                 let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
                 for (let i = 0; i < n; i++) {
                     sumX += x_nums[i];
                     sumY += y_vals[i];
                     sumXY += x_nums[i] * y_vals[i];
                     sumXX += x_nums[i] * x_nums[i];
                 }
                 const m = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
                 const b = (sumY - m * sumX) / n;
                 
                 const trend_y = x_nums.map(x => m * x + b);
                 
                 // Determine trend color
                 const start_y = trend_y[0];
                 const end_y = trend_y[trend_y.length - 1];
                 const mean_y = sumY / n;
                 
                 let trendColor = '#94a3b8'; 
                 const total_change = end_y - start_y;
                 const rel_change = mean_y !== 0 ? total_change / Math.abs(mean_y) : 0;
                 
                 if (Math.abs(rel_change) < 0.05) trendColor = '#94a3b8';
                 else if (m > 0) trendColor = '#4ade80';
                 else trendColor = '#f87171';
                 
                 data.push({
                     x: x_vals,
                     y: trend_y,
                     type: 'scatter',
                     mode: 'lines',
                     name: `Tendencia (${m > 0 ? '↗' : (m < 0 ? '↘' : '→')})`,
                     line: {color: trendColor, width: 2, dash: 'dash'},
                     hoverinfo: 'skip'
                 });
             }
             
             layout = {
               margin: { t: 30, b: 40, l: 50, r: 20 },
               xaxis: { 
                   showgrid: false,
                   rangeslider: { 
                       visible: true, 
                       thickness: 0.08,
                       bgcolor: '#fcfdfe',
                       bordercolor: '#f1f5f9',
                       borderwidth: 1
                   },
                   type: 'date',
                   tickformat: '%b %Y',
                   automargin: true,
                   fixedrange: false
               },
               yaxis: { showgrid: true, gridcolor: '#f1f5f9', zeroline: false, fixedrange: false },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                hovermode: 'x unified',
                legend: { orientation: 'h', y: 1.1 },
                dragmode: 'zoom',
                transition: { duration: 300, easing: 'cubic-in-out' }
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
            
            sorted.forEach(d => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${d.month}</td>
                    <td class="num-col">${new Intl.NumberFormat('en-US').format(d.total_depositos || 0)}</td>
                    <td class="num-col">${new Intl.NumberFormat('en-US').format(d.total_retiros || 0)}</td>
                    <td class="num-col">${new Intl.NumberFormat('en-US').format(d.calculo_ggr || 0)}</td>
                    <td class="num-col">${new Intl.NumberFormat('en-US').format(d.calculo_ngr || 0)}</td>
                    <td class="num-col"><strong>${new Intl.NumberFormat('en-US').format(d.calculo_comision || 0)}</strong></td>
                `;
                tbody.appendChild(tr);
            });
        } else {
            tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;">No hay datos disponibles</td></tr>';
        }
    }

    // Initial Load
    updateTopAgencies();
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
        class_order_json=class_order_json
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
