import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Loads raw player data from CSV and performs initial preprocessing.
    Adapted for NEW schema (CSV) compatible with metricas_agente.py.
    """
    try:
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Rename columns to match internal schema expected by metricas_agente.py
        # and subsequent reporting steps
        rename_map = {
            'date_evento': 'creado',
            'comis_calculada': 'calculo_ngr',
            'n_deposito': 'num_depositos',
            'n_retiro': 'num_retiros',
            'deposito': 'total_depositos',
            'retiro': 'total_retiros',
            'ggr_deportiva': 'apuestas_deportivas_ggr',
            'ggr_casino': 'casino_ggr',
            'player_id': 'jugador_id',
            'agente_username': 'nombre_usuario_agente',
            'agente_id': 'id_agente'
        }
        
        # Check if required columns exist before renaming
        missing_cols = [k for k in rename_map.keys() if k not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns in CSV: {missing_cols}")
            
        df = df.rename(columns=rename_map)

        # Convert 'creado' to datetime
        if 'creado' in df.columns:
            df['creado'] = pd.to_datetime(df['creado'], errors='coerce')
            # Create a 'month' column for compatibility if needed elsewhere
            df['month'] = df['creado'].dt.to_period('M')
            df['date'] = df['creado'] # Alias for compatibility
        
        # Ensure numerical columns are floats/ints and fill NaNs
        numeric_cols = [
            'calculo_ngr', 'num_depositos', 'num_retiros', 
            'total_depositos', 'total_retiros', 
            'apuestas_deportivas_ggr', 'casino_ggr'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0.0

        # Fill ID and Username NaNs
        if 'nombre_usuario_agente' in df.columns:
            df['nombre_usuario_agente'] = df['nombre_usuario_agente'].fillna('Unknown')
            
        if 'id_agente' in df.columns:
             df['id_agente'] = df['id_agente'].fillna(0).astype(int)
        
        # Ensure jugador_id is treated as string/object to avoid float issues
        if 'jugador_id' in df.columns:
            df['jugador_id'] = df['jugador_id'].astype(str)

        return df
        
    except Exception as e:
        print(f"Error in load_data: {e}")
        raise
