import pandas as pd

def load_data(file_path):
    """
    Loads raw player data and performs initial preprocessing.
    Adapted for NEW schema (Excel).
    """
    try:
        # Load Excel
        df = pd.read_excel(file_path)
        
        # Rename columns to match internal schema
        rename_map = {
            'Agente': 'nombre_usuario_agente',
            '#': 'rank_global',
            'Depósitos': 'total_depositos',
            'Retiros': 'total_retiros',
            'GGR': 'calculo_ggr',
            'NGR': 'calculo_ngr',
            'Comisión': 'calculo_comision',
            'Jugadores_Únicos': 'active_players',
            'Categoría': 'Clase',
            'Score': 'score_global',
            'Tipo_Agente': 'tipo_agente',
            # Mappings for Radar Chart compatibility
            'Punt_Estabilidad': 'Si',
            'Punt_Rentabilidad': 'Vi',
            'Punt_Fidelidad': 'Gi',
            'Punt_Tendencia': 'Ti',
            'Punt_Volumen': 'Tx_i',
            'Punt_Rotación': 'Freq_i',
            'Punt_Eficiencia': 'Conv_i',
            # Keep other scores as is (or map if needed)
            'Punt_Crecimiento': 'Punt_Crecimiento',
            'Punt_Pareto': 'Punt_Pareto',
            'Punt_Productos': 'Punt_Productos'
        }
        df = df.rename(columns=rename_map)

        # Generate id_agente (int) from nombre_usuario_agente
        # Using factorization to ensure unique integer ID per unique name
        if 'nombre_usuario_agente' in df.columns:
            # We factorize based on the unique names to ensure consistency
            # However, if we process month by month, we need a global mapping.
            # Assuming load_data loads the FULL dataset (all months).
            df['id_agente'] = pd.factorize(df['nombre_usuario_agente'])[0] + 1 # Start from 1
            df['id_agente'] = df['id_agente'].astype(int)
        else:
             raise ValueError("Column 'Agente' (mapped to 'nombre_usuario_agente') not found.")

        # Parse 'Mes' to 'month' and 'date'
        # Input format example: "2025-07"
        if 'Mes' in df.columns:
            df['date'] = pd.to_datetime(df['Mes'])
            df['month'] = df['date'].dt.to_period('M')
        else:
            raise ValueError("Column 'Mes' not found.")
            
        # Fill nulls for numerical columns
        cols_to_fill = [
            'rank_global',
            'total_depositos', 'total_retiros', 
            'calculo_comision', 'calculo_ngr', 'calculo_ggr',
            'active_players', 'score_global',
            'Si', 'Vi', 'Gi', 'Ti', 'Tx_i', 'Freq_i', 'Conv_i'
        ]
        
        # Add new Punt columns to fillna list
        cols_to_fill += [c for c in df.columns if c.startswith('Punt_')]

        for col in cols_to_fill:
            if col in df.columns:
                df[col] = df[col].fillna(0)
            else:
                # If mapped column is missing, create it with 0 (e.g. if Excel lacks Punt_Fidelidad)
                if col in rename_map.values():
                     df[col] = 0
                
        return df
        
    except Exception as e:
        print(f"Error in load_data: {e}")
        raise
