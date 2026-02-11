import pandas as pd

def aggregate_by_agent_month(df):
    """
    Aggregates data by Agent and Month.
    For the new Excel input, data is already monthly, so this mainly ensures uniqueness
    and aggregates potential duplicates.
    """
    # Group by Agent and Month
    
    # Define aggregation rules
    agg_rules = {
        'rank_global': 'min', # Keep the best rank (or first)
        'total_depositos': 'sum',
        'total_retiros': 'sum',
        'calculo_comision': 'sum',
        'calculo_ngr': 'sum',
        'calculo_ggr': 'sum',
        'active_players': 'sum', # Sum if multiple records for same agent/month
        'score_global': 'mean',
        'Si': 'mean', 'Vi': 'mean', 'Gi': 'mean', 'Ti': 'mean',
        'Tx_i': 'mean', 'Freq_i': 'mean', 'Conv_i': 'mean',
        'Clase': 'first', # Assuming class doesn't change within same month split
        'tipo_agente': 'first'
    }
    
    # Add rules for any extra Punt columns
    for col in df.columns:
        if col.startswith('Punt_') and col not in agg_rules:
            agg_rules[col] = 'mean'
            
    # Filter only available columns
    actual_agg_rules = {k: v for k, v in agg_rules.items() if k in df.columns}
    
    agg = df.groupby(['id_agente', 'nombre_usuario_agente', 'month']).agg(actual_agg_rules)
    
    return agg.reset_index()

def aggregate_by_agent_total(df_monthly):
    """
    Aggregates monthly data into a single profile per Agent.
    """
    # Group by Agent
    
    # Define aggregation rules
    agg_rules = {
        'rank_global': 'min',
        'total_depositos': 'sum',
        'total_retiros': 'sum',
        'calculo_comision': 'sum',
        'calculo_ngr': 'sum',
        'calculo_ggr': 'sum',
        'active_players': 'median', # MEDIAN activity per month
        'score_global': 'mean', # Average score over time
        'Si': 'mean', 'Vi': 'mean', 'Gi': 'mean', 'Ti': 'mean',
        'Tx_i': 'mean', 'Freq_i': 'mean', 'Conv_i': 'mean',
        'Clase': 'last', # Take the latest class
        'tipo_agente': 'first'
    }

    # Add rules for any extra Punt columns
    for col in df_monthly.columns:
        if col.startswith('Punt_') and col not in agg_rules:
            agg_rules[col] = 'mean'
            
    # Filter only available columns
    actual_agg_rules = {k: v for k, v in agg_rules.items() if k in df_monthly.columns}
    
    # Remove rank_global from aggregation if present, we will recalculate it
    if 'rank_global' in actual_agg_rules:
        del actual_agg_rules['rank_global']

    agent_profile = df_monthly.groupby(['id_agente', 'nombre_usuario_agente']).agg(actual_agg_rules)
    
    # Rename active_players to median_players for clarity in global profile
    if 'active_players' in agent_profile.columns:
        agent_profile = agent_profile.rename(columns={'active_players': 'median_players'})
    
    # Add Risk_Safe logic
    agent_profile = agent_profile.reset_index()
    
    if 'Risk_Safe' not in agent_profile.columns:
        if 'Clase' in agent_profile.columns:
             # Example logic: A/B are Safe (1), others Risky (0)
             agent_profile['Risk_Safe'] = agent_profile['Clase'].apply(lambda x: 1 if str(x).startswith('A') or str(x).startswith('B') else 0)
        else:
             agent_profile['Risk_Safe'] = 1 # Default safe
             
    # ALWAYS Recalculate rank_global based on score_global (Descending)
    # The user explicitly requested ranking based on Score.
    if 'score_global' in agent_profile.columns:
        agent_profile['rank_global'] = agent_profile['score_global'].rank(ascending=False, method='min').astype(int)
    else:
        # Fallback if no score
        agent_profile['rank_global'] = agent_profile.index + 1

    return agent_profile
