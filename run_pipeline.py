import pandas as pd
import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_data
from report_html import generate_html_report
from logic_analytics import (
    calcular_metricas_agente, calcular_score_total,
    categorizar_agente, calcular_credito_sugerido,
    predecir_ggr, 
    analizar_retencion_cohortes, analizar_crecimiento_organico
)

def main():
    # New CSV Input
    input_file = r"c:\Users\Miguel\Documents\Proyecto_Grafico\Data\reporte_detallado_jugadores_final.csv"
    output_file = r"c:\Users\Miguel\Documents\Proyecto_Grafico\reports\dashboard.html"
    analysis_output = r"c:\Users\Miguel\Documents\Proyecto_Grafico\reports\agent_analysis.csv"
    
    print(f"Loading data from {input_file}...")
    try:
        df = load_data(input_file)
        print(f"Data loaded. Shape: {df.shape}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("\nProcessing Agents with Unified Logic (Metrics + Deep Analysis)...")
    
    total_jugadores_global = df['jugador_id'].nunique()
    agentes = df['nombre_usuario_agente'].unique()
    
    agent_records = []
    monthly_records = []
    retention_records = []
    growth_records = []
    
    for agent_name in agentes:
        try:
            # Filter data for this agent
            df_agent = df[df['nombre_usuario_agente'] == agent_name].copy()
            
            # --- 1. CORE METRICS & SCORING ---
            metricas, df_mensual = calcular_metricas_agente(df_agent, total_jugadores_global)
            score = calcular_score_total(metricas)
            categoria, descripcion = categorizar_agente(score)
            credito, detalles = calcular_credito_sugerido(df_mensual, score, metricas)
            ggr_prediccion = predecir_ggr(df_mensual)
            
            # --- 2. DEEP ANALYSIS (RETENTION & GROWTH) ---
            df_retencion = analizar_retencion_cohortes(df_agent)
            df_crecimiento = analizar_crecimiento_organico(df_agent)
            
            # Calculamos métricas resumen
            avg_retencion = df_retencion['tasa_retencion'].mean() if df_retencion is not None else 0
            avg_pct_nuevos = df_crecimiento['pct_nuevos'].mean() if df_crecimiento is not None else 0
            
            # --- Build Agent Profile Record (df_agents) ---
            record = {
                'id_agente': df_agent['id_agente'].iloc[0] if 'id_agente' in df_agent.columns else 0,
                'nombre_usuario_agente': agent_name,
                'score_global': score,
                'Clase': categoria,
                'Risk_Safe': 1 if 'A' in categoria or 'B' in categoria else 0,
                'credito_sugerido': credito,
                'descripcion_categoria': descripcion,
                'ggr_prediccion': ggr_prediccion,
                'active_players': df_agent['jugador_id'].nunique(),
                'total_depositos': df_mensual['total_depositos'].sum(),
                'calculo_ngr': df_mensual['calculo_ngr'].sum(),
                'calculo_ggr': df_mensual['apuestas_deportivas_ggr'].sum() + df_mensual['casino_ggr'].sum(),
                
                # --- New Analytic Summary Fields ---
                'avg_retencion_cohortes': round(avg_retencion, 2),
                'avg_pct_nuevos_organico': round(avg_pct_nuevos, 2)
            }
            # Add the 11 individual metric scores
            record.update(metricas)
            agent_records.append(record)
            
            # --- Build Monthly History Record (df_monthly) ---
            if not df_mensual.empty:
                df_mensual_reset = df_mensual.reset_index()
                df_mensual_reset['id_agente'] = record['id_agente']
                df_mensual_reset['nombre_usuario_agente'] = agent_name
                df_mensual_reset['month'] = df_mensual_reset['mes'].astype(str)
                df_mensual_reset['calculo_ggr'] = df_mensual_reset['apuestas_deportivas_ggr'] + df_mensual_reset['casino_ggr']
                monthly_records.append(df_mensual_reset)

            # --- Build Retention Record ---
            if df_retencion is not None and not df_retencion.empty:
                df_retencion['id_agente'] = record['id_agente']
                retention_records.append(df_retencion)
                
            # --- Build Growth Record ---
            if df_crecimiento is not None and not df_crecimiento.empty:
                df_crecimiento['id_agente'] = record['id_agente']
                growth_records.append(df_crecimiento)
            
        except Exception as e:
            print(f"Error processing agent {agent_name}: {e}")
            continue

    # Create DataFrames
    df_agents = pd.DataFrame(agent_records)
    df_monthly = pd.concat(monthly_records, ignore_index=True) if monthly_records else pd.DataFrame()
    df_retention = pd.concat(retention_records, ignore_index=True) if retention_records else pd.DataFrame()
    df_growth = pd.concat(growth_records, ignore_index=True) if growth_records else pd.DataFrame()

    # Recalculate rank_global based on score
    if not df_agents.empty:
        df_agents['rank_global'] = df_agents['score_global'].rank(ascending=False, method='min').astype(int)
        
        # Save backend analysis for verification/export
        df_agents.to_csv(analysis_output, index=False)
        print(f"Detailed analysis saved to {analysis_output}")
    
    print(f"Global Aggregation done.")
    print(f"  Agents: {df_agents.shape}")
    print(f"  Monthly Rows: {df_monthly.shape}")
    print(f"  Retention Rows: {df_retention.shape}")
    print(f"  Growth Rows: {df_growth.shape}")

    print("\nGenerating Report...")
    try:
        # Pass new DFs to report generator
        generate_html_report(df_agents, df_monthly, df_retention, df_growth, output_file)
        print(f"Report generated at {output_file}")
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
