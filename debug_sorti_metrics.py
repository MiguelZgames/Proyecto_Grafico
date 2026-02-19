
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_data
from logic_analytics import (
    calcular_metricas_agente, calcular_score_total,
    categorizar_agente, PESOS_METRICAS
)

def analyze_sorti_metrics():
    input_file = r"c:\Users\Miguel\Documents\Proyecto_Grafico\Data\reporte_detallado_jugadores_final.csv"
    print(f"Loading data from {input_file}...")
    
    try:
        df = load_data(input_file)
        print(f"Data loaded. Shape: {df.shape}")
        
        target_agent = "SortiOficial"
        agent_data = df[df['nombre_usuario_agente'] == target_agent].copy()
        
        if agent_data.empty:
            print(f"ERROR: No data found for agent '{target_agent}'")
            return

        total_jugadores_global = df['jugador_id'].nunique()
        
        # Calculate metrics using the actual function (returns overall metrics + df_mensual)
        metricas_global, df_mensual = calcular_metricas_agente(agent_data, total_jugadores_global)
        score_global = calcular_score_total(metricas_global)
        clase_global, desc_global = categorizar_agente(score_global)
        
        with open('debug_output.txt', 'w', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"  ANÁLISIS DETALLADO DE MÉTRICAS — {target_agent}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"Total Rows: {len(agent_data)}\n")
            f.write(f"Agent ID: {agent_data['id_agente'].iloc[0]}\n")
            f.write(f"Total Jugadores Global: {total_jugadores_global}\n")
            f.write(f"Meses disponibles: {len(df_mensual)}\n\n")
            
            # ── RESUMEN GLOBAL ──
            f.write(f"{'─'*80}\n")
            f.write(f"  RESUMEN GLOBAL (Todos los meses agregados)\n")
            f.write(f"{'─'*80}\n")
            f.write(f"  Score Global: {score_global:.4f}\n")
            f.write(f"  Clase: {clase_global} — {desc_global}\n\n")
            
            f.write(f"  {'Métrica':<25} {'Score (0-10)':>12} {'Peso':>8} {'Contribución':>14}\n")
            f.write(f"  {'─'*25} {'─'*12} {'─'*8} {'─'*14}\n")
            for k in PESOS_METRICAS:
                v = metricas_global.get(k, 0)
                w = PESOS_METRICAS[k]
                contrib = v * w
                f.write(f"  {k:<25} {v:>12.4f} {w:>8.2f} {contrib:>14.4f}\n")
            f.write(f"  {'─'*25} {'─'*12} {'─'*8} {'─'*14}\n")
            f.write(f"  {'TOTAL':<25} {'':>12} {'1.00':>8} {score_global:>14.4f}\n\n")
            
            # ── DETALLE MES A MES ──
            f.write(f"\n{'='*80}\n")
            f.write(f"  DETALLE MES A MES\n")
            f.write(f"{'='*80}\n\n")
            
            metric_keys = list(PESOS_METRICAS.keys())
            
            for idx, row in df_mensual.iterrows():
                mes = str(row['mes'])
                
                # Raw data for this month
                ngr = row['calculo_ngr']
                deps = row['total_depositos']
                num_deps = row['num_depositos']
                num_rets = row['num_retiros']
                ggr_c = row['casino_ggr']
                ggr_d = row['apuestas_deportivas_ggr']
                ggr_total = ggr_c + ggr_d
                players = row['active_players']
                comision = row.get('calculo_comision', 0)
                score_mes = row.get('score_global', 0)
                clase_mes = row.get('Clase', '?')
                risk_mes = row.get('Risk_Safe', '?')
                
                margen = 0.05
                bets_c = ggr_c / margen if margen else 0
                bets_d = ggr_d / margen if margen else 0
                total_bets = bets_c + bets_d
                
                f.write(f"{'━'*80}\n")
                f.write(f"  📅 MES: {mes}  |  Score: {score_mes:.4f}  |  Clase: {clase_mes}  |  Risk: {'Seguro' if risk_mes == 1 else 'Riesgo'}\n")
                f.write(f"{'━'*80}\n\n")
                
                # Raw data summary
                f.write(f"  📊 DATOS CRUDOS DEL MES:\n")
                f.write(f"    NGR (calculo_ngr):          {ngr:>15,.2f}\n")
                f.write(f"    Comisión (calculo_comision): {comision:>15,.2f}\n")
                f.write(f"    Total Depósitos ($):         {deps:>15,.2f}\n")
                f.write(f"    Num. Depósitos (txn):        {num_deps:>15,.0f}\n")
                f.write(f"    Total Retiros ($):           {row['total_retiros']:>15,.2f}\n")
                f.write(f"    Num. Retiros (txn):          {num_rets:>15,.0f}\n")
                f.write(f"    GGR Casino:                  {ggr_c:>15,.2f}\n")
                f.write(f"    GGR Deportes:                {ggr_d:>15,.2f}\n")
                f.write(f"    GGR Total:                   {ggr_total:>15,.2f}\n")
                f.write(f"    Jugadores Activos:           {players:>15,.0f}\n")
                f.write(f"    Apuestas Est. Casino:        {bets_c:>15,.2f}\n")
                f.write(f"    Apuestas Est. Deportes:      {bets_d:>15,.2f}\n")
                f.write(f"    Apuestas Est. Total:         {total_bets:>15,.2f}\n\n")
                
                # ── Detailed calc for each metric ──
                f.write(f"  ⚙️  CÁLCULOS DETALLADOS POR MÉTRICA:\n\n")
                
                # 1. Rentabilidad
                f.write(f"  1️⃣  RENTABILIDAD (Peso: {PESOS_METRICAS['rentabilidad']:.2f})\n")
                if deps > 0:
                    rent_pct = (ngr / deps) * 100
                    f.write(f"     Fórmula: (NGR / Depósitos) * 100 = ({ngr:,.2f} / {deps:,.2f}) * 100 = {rent_pct:.4f}%\n")
                    if rent_pct >= 8:
                        s_pct = 7.0
                        f.write(f"     Condición: rent_pct >= 8 → score_pct = 7.0\n")
                    elif rent_pct >= 6:
                        s_pct = 5.5
                        f.write(f"     Condición: rent_pct >= 6 → score_pct = 5.5\n")
                    elif rent_pct >= 4:
                        s_pct = 4.0
                        f.write(f"     Condición: rent_pct >= 4 → score_pct = 4.0\n")
                    else:
                        s_pct = max(0, min(7, rent_pct * 1.75))
                        f.write(f"     Condición: else → score_pct = max(0, min(7, {rent_pct:.4f} * 1.75)) = {s_pct:.4f}\n")
                    if ngr > 0:
                        s_vol = min(3.0, np.log10(ngr + 1) * 0.75)
                        f.write(f"     Bonus volumen: min(3.0, log10({ngr:,.2f}+1) * 0.75) = {s_vol:.4f}\n")
                    else:
                        s_vol = 0
                        f.write(f"     Bonus volumen: NGR <= 0 → 0\n")
                    total_rent = s_pct + s_vol
                    f.write(f"     ✅ Score = {s_pct:.4f} + {s_vol:.4f} = {total_rent:.4f}\n\n")
                else:
                    f.write(f"     Depósitos = 0 → Score = 0\n\n")
                
                # 2. Volumen
                f.write(f"  2️⃣  VOLUMEN (Peso: {PESOS_METRICAS['volumen']:.2f})\n")
                txs = num_deps + num_rets
                if txs > 0:
                    vol_score = min(10, max(0, np.log10(txs + 1) * 2.3))
                    f.write(f"     Fórmula: min(10, max(0, log10({txs:.0f}+1) * 2.3)) = {vol_score:.4f}\n")
                else:
                    vol_score = 0
                    f.write(f"     Transacciones = 0 → Score = 0\n")
                f.write(f"     ✅ Score = {vol_score:.4f}\n\n")
                
                # 3. Fidelidad
                f.write(f"  3️⃣  FIDELIDAD (Peso: {PESOS_METRICAS['fidelidad']:.2f})\n")
                if total_jugadores_global > 0:
                    fid_score = min(10, (players / total_jugadores_global) * 100 * 2.5)
                    f.write(f"     Fórmula: min(10, ({players:.0f} / {total_jugadores_global}) * 100 * 2.5) = {fid_score:.4f}\n")
                else:
                    fid_score = 0
                    f.write(f"     Total jugadores global = 0 → Score = 0\n")
                f.write(f"     ✅ Score = {fid_score:.4f}\n\n")
                
                # 4. Estabilidad
                f.write(f"  4️⃣  ESTABILIDAD (Peso: {PESOS_METRICAS['estabilidad']:.2f})\n")
                f.write(f"     Snapshot mensual → Score = 5.0 (neutral)\n")
                f.write(f"     ✅ Score = 5.0000\n\n")
                
                # 5. Crecimiento
                f.write(f"  5️⃣  CRECIMIENTO (Peso: {PESOS_METRICAS['crecimiento']:.2f})\n")
                if idx > 0:
                    prev_deps = df_mensual.iloc[idx-1]['num_depositos']
                    if prev_deps > 0:
                        pct = ((num_deps - prev_deps) / prev_deps) * 100
                        f.write(f"     Fórmula: (({num_deps:.0f} - {prev_deps:.0f}) / {prev_deps:.0f}) * 100 = {pct:.2f}%\n")
                        if pct >= 20: crec = 10
                        elif pct >= 10: crec = 8
                        elif pct >= 5: crec = 6.5
                        elif pct >= 0: crec = 5
                        elif pct >= -10: crec = 3.5
                        elif pct >= -20: crec = 2
                        else: crec = 1
                        f.write(f"     Condición: pct={pct:.2f}% → Score = {crec}\n")
                    elif num_deps > 0:
                        crec = 10
                        f.write(f"     Prev deps = 0, current > 0 → Score = 10\n")
                    else:
                        crec = 5
                        f.write(f"     Ambos = 0 → Score = 5\n")
                else:
                    crec = 5.0
                    f.write(f"     Primer mes → Score = 5.0 (default)\n")
                f.write(f"     ✅ Score = {crec:.4f}\n\n")
                
                # 6. Eficiencia Casino
                f.write(f"  6️⃣  EFICIENCIA CASINO (Peso: {PESOS_METRICAS['eficiencia_casino']:.2f})\n")
                if ggr_c > 0:
                    val_c = (num_deps / ggr_c) * 100
                    f.write(f"     Fórmula: ({num_deps:.0f} / {ggr_c:,.2f}) * 100 = {val_c:.4f}\n")
                    if val_c < 5.5: ec = 10
                    elif val_c < 10: ec = 7.5
                    elif val_c < 14: ec = 5
                    elif val_c < 20: ec = 3
                    elif val_c < 33: ec = 2
                    else: ec = max(1.0, 10 - (val_c/10))
                    f.write(f"     Condición: val={val_c:.4f} → Score = {ec}\n")
                else:
                    ec = 0
                    f.write(f"     GGR Casino <= 0 → Score = 0\n")
                f.write(f"     ✅ Score = {ec:.4f}\n\n")
                
                # 7. Eficiencia Deportes
                f.write(f"  7️⃣  EFICIENCIA DEPORTES (Peso: {PESOS_METRICAS['eficiencia_deportes']:.2f})\n")
                if ggr_d > 0:
                    val_d = (num_deps / ggr_d) * 100
                    f.write(f"     Fórmula: ({num_deps:.0f} / {ggr_d:,.2f}) * 100 = {val_d:.4f}\n")
                    if val_d < 5.5: ed = 10
                    elif val_d < 10: ed = 7.5
                    elif val_d < 14: ed = 5
                    elif val_d < 20: ed = 3
                    elif val_d < 33: ed = 2
                    else: ed = max(1.0, 10 - (val_d/10))
                    f.write(f"     Condición: val={val_d:.4f} → Score = {ed}\n")
                else:
                    ed = 0
                    f.write(f"     GGR Deportes <= 0 → Score = 0\n")
                f.write(f"     ✅ Score = {ed:.4f}\n\n")
                
                # 8. Conversión
                f.write(f"  8️⃣  CONVERSIÓN (Peso: {PESOS_METRICAS['eficiencia_conversion']:.2f})\n")
                if deps > 0:
                    val_conv = (ggr_total / deps) * 100
                    f.write(f"     Fórmula: ({ggr_total:,.2f} / {deps:,.2f}) * 100 = {val_conv:.4f}%\n")
                    if val_conv >= 15: conv = 10
                    elif val_conv >= 10: conv = 7.5
                    elif val_conv >= 7: conv = 5
                    elif val_conv >= 5: conv = 3
                    else: conv = max(0, val_conv * 2)
                    f.write(f"     Condición: val={val_conv:.4f}% → Score = {conv}\n")
                else:
                    conv = 0
                    f.write(f"     Depósitos = 0 → Score = 0\n")
                f.write(f"     ✅ Score = {conv:.4f}\n\n")
                
                # 9. Tendencia
                f.write(f"  9️⃣  TENDENCIA (Peso: {PESOS_METRICAS['tendencia']:.2f})\n")
                f.write(f"     Snapshot mensual → Score = 5.0 (neutral)\n")
                f.write(f"     ✅ Score = 5.0000\n\n")
                
                # 10. Diversificación
                f.write(f"  🔟  DIVERSIFICACIÓN (Peso: {PESOS_METRICAS['diversificacion']:.2f})\n")
                if total_bets > 0:
                    hhi = (bets_c/total_bets)**2 + (bets_d/total_bets)**2
                    div_score = (1 - hhi) * 10
                    f.write(f"     Casino bets: {bets_c:,.2f}, Deportes bets: {bets_d:,.2f}\n")
                    f.write(f"     HHI = ({bets_c:,.2f}/{total_bets:,.2f})² + ({bets_d:,.2f}/{total_bets:,.2f})² = {hhi:.6f}\n")
                    f.write(f"     Fórmula: (1 - {hhi:.6f}) * 10 = {div_score:.4f}\n")
                else:
                    div_score = 0
                    f.write(f"     Total apuestas = 0 → Score = 0\n")
                f.write(f"     ✅ Score = {div_score:.4f}\n\n")
                
                # 11. Calidad Jugadores
                f.write(f"  1️⃣1️⃣ CALIDAD JUGADORES (Peso: {PESOS_METRICAS['calidad_jugadores']:.2f})\n")
                if players > 0:
                    avg_bets = total_bets / players
                    f.write(f"     Fórmula: Apuestas / Jugadores = {total_bets:,.2f} / {players:.0f} = {avg_bets:,.2f}\n")
                    if avg_bets > 10000: cal = 8
                    elif avg_bets > 5000: cal = 6
                    elif avg_bets > 1000: cal = 4
                    else: cal = 2
                    f.write(f"     Condición: avg={avg_bets:,.2f} → Score = {cal}\n")
                else:
                    cal = 0
                    f.write(f"     Jugadores = 0 → Score = 0\n")
                f.write(f"     ✅ Score = {cal:.4f}\n\n")
                
                # Summary table for this month
                f.write(f"  📋 RESUMEN DEL MES {mes}:\n")
                f.write(f"  {'Métrica':<25} {'Score':>8} {'Peso':>8} {'Contribución':>14}\n")
                f.write(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*14}\n")
                for mk in metric_keys:
                    mv = row.get(mk, 0)
                    mw = PESOS_METRICAS[mk]
                    mc = mv * mw
                    f.write(f"  {mk:<25} {mv:>8.4f} {mw:>8.2f} {mc:>14.4f}\n")
                f.write(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*14}\n")
                f.write(f"  {'SCORE MES':<25} {score_mes:>8.4f} {'':>8} {score_mes:>14.4f}\n")
                f.write(f"  Clase: {clase_mes}  |  Risk: {'Seguro' if risk_mes == 1 else 'Riesgo'}\n")
                f.write(f"\n\n")
            
        print("Analysis written to debug_output.txt")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_sorti_metrics()
