
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data_loader import load_data

input_file = r"c:\Users\Miguel\Documents\Proyecto_Grafico\Data\reporte_detallado_jugadores_final.csv"

try:
    df = load_data(input_file)
    print("Data loaded successfully.")
    print(f"Shape: {df.shape}")
    print("\nColumns:", df.columns.tolist())
    
    print("\nSample 'creado' dates:")
    print(df['creado'].head())
    print("NaT in 'creado':", df['creado'].isna().sum())
    
    print("\nSample 'calculo_ngr':")
    print(df['calculo_ngr'].head())
    print("Sum 'calculo_ngr':", df['calculo_ngr'].sum())
    
    print("\nSample 'total_depositos':")
    print(df['total_depositos'].head())
    print("Sum 'total_depositos':", df['total_depositos'].sum())
    
except Exception as e:
    print(f"Error: {e}")
