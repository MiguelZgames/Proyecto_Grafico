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
            'creado': 'creado', # fix potential conflict if date_evento is not found
            'date_evento': 'creado',
            'comis_calculada': 'calculo_comision', # Correct mapping per user
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
        missing_cols = [k for k in rename_map.keys() if k not in df.columns and k != 'creado'] # 'creado' might be target
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
            'calculo_ngr', 'calculo_comision', # Added commission
            'num_depositos', 'num_retiros', 
            'total_depositos', 'total_retiros', 
            'apuestas_deportivas_ggr', 'casino_ggr'
        ]
        
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0.0
            
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        # Fallback: If calculo_ngr is 0 but calculo_comision has data, use commission as proxy for NGR logic
        # strictly to preserve existing behavior where comis_calculada was mapped to ngr
        if df['calculo_ngr'].sum() == 0 and df['calculo_comision'].sum() != 0:
             df['calculo_ngr'] = df['calculo_comision']

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
