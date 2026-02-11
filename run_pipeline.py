import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_data
from aggregator import aggregate_by_agent_month, aggregate_by_agent_total
from report_html import generate_html_report

def main():
    input_file = r"c:\Users\Miguel\Documents\Proyecto_Grafico\Data\ranking_todos_meses.xlsx"
    output_file = r"c:\Users\Miguel\Documents\Proyecto_Grafico\reports\dashboard.html"
    
    print(f"Loading data from {input_file}...")
    try:
        df = load_data(input_file)
        print(f"Data loaded. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check generated ID
        print(f"Sample IDs: {df[['nombre_usuario_agente', 'id_agente']].head().to_dict('records')}")
        
        # Check Month Parsing
        print(f"Sample Months: {df[['date', 'month']].head()}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("\nAggregating Monthly...")
    try:
        df_monthly = aggregate_by_agent_month(df)
        print(f"Monthly Aggregation done. Shape: {df_monthly.shape}")
    except Exception as e:
        print(f"Error in monthly aggregation: {e}")
        return

    print("\nAggregating Global...")
    try:
        df_agents = aggregate_by_agent_total(df_monthly)
        print(f"Global Aggregation done. Shape: {df_agents.shape}")
        print(f"Top 5 Agents by Score:\n{df_agents.sort_values('score_global', ascending=False)[['nombre_usuario_agente', 'score_global', 'Clase']].head()}")
    except Exception as e:
        print(f"Error in global aggregation: {e}")
        return

    print("\nGenerating Report...")
    try:
        generate_html_report(df_agents, df_monthly, output_file)
        print(f"Report generated at {output_file}")
    except Exception as e:
        print(f"Error generating report: {e}")
        return

if __name__ == "__main__":
    main()
