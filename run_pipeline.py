import pandas as pd
import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_data
from report_html import generate_html_report
from logic_analytics import (
    calcular_metricas_agente_con_mensual, calcular_score_total,
    categorizar_agente, calcular_credito_sugerido,
    predecir_ggr
)
from metrics_dashboard_generator import load_and_validate_data, generate_metrics_dashboard

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
    
    agent_records = []
    monthly_records = []

    # === 1. CALCULAR VISTA GLOBAL (TODA LA EMPRESA) ===
    print("\nCalculando Vista Global de la Empresa...")
    try:
        metricas_g, df_mensual_orig_g, df_mensual_mets_g = calcular_metricas_agente_con_mensual(df, total_jugadores_global)
        df_mensual_g = pd.merge(df_mensual_orig_g, df_mensual_mets_g, on='mes', how='left')
        
        if 'calculo_comision' not in df_mensual_g.columns:
            df_mensual_g['calculo_comision'] = df_mensual_g['calculo_ngr']
            
        def compute_clase(s):
            if pd.notna(s): return categorizar_agente(s)[0]
            return 'C'
            
        df_mensual_g['Clase'] = df_mensual_g['score_global'].apply(compute_clase)
        df_mensual_g['Risk_Safe'] = df_mensual_g['Clase'].apply(lambda c: 1 if ('A' in c or 'B' in c) else 0)

        score_g = calcular_score_total(metricas_g)
        categoria_g, descripcion_g = categorizar_agente(score_g)
        credito_g, detalles_g = calcular_credito_sugerido(df_mensual_g, score_g, metricas_g)
        ggr_prediccion_g = predecir_ggr(df_mensual_g)
        
        record_g = {
            'id_agente': 'GLOBAL',
            'nombre_usuario_agente': '🌟 VISTA GLOBAL (Toda la Empresa)',
            'score_global': score_g,
            'Clase': categoria_g,
            'Risk_Safe': 1 if 'A' in categoria_g or 'B' in categoria_g else 0,
            'credito_sugerido': credito_g,
            'descripcion_categoria': descripcion_g,
            'ggr_prediccion': ggr_prediccion_g,
            'active_players': df['jugador_id'].nunique(),
            'total_depositos': df_mensual_g['total_depositos'].sum(),
            'total_retiros': df_mensual_g['total_retiros'].sum(),
            'calculo_ngr': df_mensual_g['calculo_ngr'].sum(),
            'calculo_ggr': df_mensual_g['apuestas_deportivas_ggr'].sum() + df_mensual_g['casino_ggr'].sum(),
            'calculo_comision': df_mensual_g['calculo_comision'].sum(),
        }
        record_g.update(metricas_g)
        agent_records.append(record_g)
        
        if not df_mensual_g.empty:
            monthly_df_g = df_mensual_g.copy()
            monthly_df_g['id_agente'] = 'GLOBAL'
            monthly_df_g['month'] = monthly_df_g['mes'].astype(str)
            monthly_df_g['calculo_ggr'] = monthly_df_g['apuestas_deportivas_ggr'] + monthly_df_g['casino_ggr']
            monthly_df_g = monthly_df_g.rename(columns={'apuestas_deportivas_ggr': 'ggr_deportiva', 'casino_ggr': 'ggr_casino'})
            margen = 0.05
            monthly_df_g['total_apuesta_deportiva'] = monthly_df_g['ggr_deportiva'] / margen
            monthly_df_g['total_apuesta_casino'] = monthly_df_g['ggr_casino'] / margen
            monthly_records.append(monthly_df_g)
    except Exception as e:
        print(f"Error procesando la Vista Global: {e}")
    # ===================================================

    # OPTIMIZACIÓN: Usar groupby para evitar escanear el DF completo por cada agente
    for agent_name, df_agent in df.groupby('nombre_usuario_agente'):
        try:
            df_agent = df_agent.copy()
            
            # --- 1. CORE METRICS & SCORING ---
            metricas, df_mensual_orig, df_mensual_mets = calcular_metricas_agente_con_mensual(df_agent, total_jugadores_global)
            df_mensual = pd.merge(df_mensual_orig, df_mensual_mets, on='mes', how='left')
            
            # Fallback for logic_analytics changes
            if 'calculo_comision' not in df_mensual.columns:
                df_mensual['calculo_comision'] = df_mensual['calculo_ngr']
            
            # Add Clase and Risk_Safe per month (since it was removed from logic_analytics inner loop)
            def compute_clase(s):
                if pd.notna(s):
                    return categorizar_agente(s)[0]
                return 'C'
            df_mensual['Clase'] = df_mensual['score_global'].apply(compute_clase)
            df_mensual['Risk_Safe'] = df_mensual['Clase'].apply(lambda c: 1 if ('A' in c or 'B' in c) else 0)

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
        # Force GLOBAL to be the #0 so it stays on top
        df_agents.loc[df_agents['id_agente'] == 'GLOBAL', 'rank_global'] = 0
        df_agents = df_agents.sort_values('rank_global').reset_index(drop=True)
        
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
        
        # Add generation of the Historic Metrics Dashboard
        print("\nGenerating Historical Metrics Dashboard...")
        historic_out_file = os.path.join(os.path.dirname(output_file), "metrics_historic_dashboard.html")
        dict_data, _ = load_and_validate_data(input_file)
        generate_metrics_dashboard(dict_data, out_path=historic_out_file)
        
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
