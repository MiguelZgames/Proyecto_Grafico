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
    predecir_ggr
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
            # Removed as per user request
            
            # Calculamos métricas resumen
            # Removed as per user request
            
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
                'total_retiros': df_mensual['total_retiros'].sum(),
                'calculo_ngr': df_mensual['calculo_ngr'].sum(),
                'calculo_ggr': df_mensual['apuestas_deportivas_ggr'].sum() + df_mensual['casino_ggr'].sum(),
                'calculo_comision': df_mensual['calculo_comision'].sum(),            }
            # Add the 11 individual metric scores
            record.update(metricas)
            agent_records.append(record)
            
            # --- Collect Monthly Data for Trend/Table ---
            if not df_mensual.empty:
                monthly_df = df_mensual.copy()
                monthly_df['id_agente'] = record['id_agente']
                monthly_df['month'] = monthly_df['mes'].astype(str)
                monthly_df['calculo_ggr'] = monthly_df['apuestas_deportivas_ggr'] + monthly_df['casino_ggr']
                # Rename columns to match report expectations
                monthly_df = monthly_df.rename(columns={
                    'apuestas_deportivas_ggr': 'ggr_deportiva',
                    'casino_ggr': 'ggr_casino',
                })
                # Add estimated bet columns
                margen = 0.05
                monthly_df['total_apuesta_deportiva'] = monthly_df['ggr_deportiva'] / margen
                monthly_df['total_apuesta_casino'] = monthly_df['ggr_casino'] / margen
                monthly_records.append(monthly_df)

        except Exception as e:
            print(f"Error processing agent {agent_name}: {e}")
            continue

    # Create DataFrames
    df_agents = pd.DataFrame(agent_records)
    df_monthly = pd.concat(monthly_records, ignore_index=True) if monthly_records else pd.DataFrame()

    # Recalculate rank_global based on score
    if not df_agents.empty:
        df_agents['rank_global'] = df_agents['score_global'].rank(ascending=False, method='min').astype(int)
        
        # Save backend analysis for verification/export
        df_agents.to_csv(analysis_output, index=False)
        print(f"Detailed analysis saved to {analysis_output}")
    
    print(f"Global Aggregation done.")
    print(f"  Agents: {df_agents.shape}")
    print(f"  Monthly Rows: {df_monthly.shape}")


    print("\nGenerating Report...")
    try:
        # Pass new DFs to report generator
        generate_html_report(df_agents, df_monthly, output_file)
        print(f"Report generated at {output_file}")
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
