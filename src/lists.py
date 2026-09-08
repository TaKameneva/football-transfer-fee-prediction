# ---------------------- COLUMN DEFINITIONS ----------------------
base_columns = ['player_name', 'shirt_number', 'nation', 'position', 'age', 'minutes_played']

columns_summary = base_columns + [
    'goals', 'assists', 'penalty_kicks_scored', 'penalty_kicks_attempted', 'total_shots', 'shots_on_target',
    'yellow_cards', 'red_cards', 'touches', 'tackles', 'interceptions', 'blocks', 'xG', 'npxG', 'xAG',
    'shot_creating_actions', 'goal_creating_actions', 'passes_completed', 'passes_attempted', 'pass_compl_perc',
    'prog_passes', 'carries', 'prog_carries', 'take_ons_attempted', 'take_ons_successful'
]

columns_passing = base_columns + [
    'passes_completed', 'passes_attempted', 'pass_compl_perc', 'total_pass_distance', 'total_prog_pass_distance',
    'short_pass_completed', 'short_pass_attempted', 'short_pass_compl_perc', 'medium_pass_completed',
    'medium_pass_attempted', 'medium_pass_compl_perc', 'long_pass_completed', 'long_pass_attempted',
    'long_pass_compl_perc', 'assists', 'xA', 'xAG', 'key_passes', 'passes_final_3rd',
    'pass_into_penalty_area', 'crosses_into_penalty_area', 'prog_passes'
]

columns_passTypes = base_columns + [
    'passes_attempted', 'live_ball_pass', 'dead_ball_pass', 'free_kick_pass', 'through_balls', 'switches',
    'crosses', 'throw_ins', 'corner_kick', 'in_corner', 'out_corner', 'straight_corner',
    'passes_completed', 'passes_offside', 'passes_blocked'
]

columns_defensiveActions = base_columns + [
    'tackles', 'tackles_won', 'tackles_def_3rd', 'tackles_mid_3rd', 'tackles_att_3rd',
    'dribblers_tackeled', 'dribbles_challenged', 'dribbles_tackeled_perc', 'challenges_lost',
    'total_blocks', 'passes_blocked', 'shots_blocked', 'interceptions',
    'tackles_and_interceptions', 'clearances', 'errors'
]

columns_possession = base_columns + [
    'touches', 'touches_def_pen_area', 'touches_def_3rd', 'touches_mid_3rd', 'touches_att_3rd',
    'touches_att_pen_area', 'live_ball_touches', 'take_ons_attempted', 'take_ons_successful',
    'take_ons_success_perc', 'times_tackeled_during_take_on', 'tackeled_during_take_on_perc',
    'total_carries', 'total_carrying_distance', 'prog_carrying_distance', 'total_prog_carries',
    'carries_into_final_3rd', 'carries_into_penalty_area', 'total_miscontrols', 'dispossessed',
    'total_passes_received', 'prog_passes_received'
]

columns_miscanceallousStats = base_columns + [
    'yellow_cards', 'red_cards', 'second_yellow_card', 'fouls_committed', 'fouls_drawn', 'offsides', 'crosses',
    'interceptions', 'tackles_won', 'penalty_kicks_won', 'penalty_kicks_conceded', 'own_goals',
    'ball_recoveries', 'aerials_won', 'aerials_lost', 'aerials_won_perc'
]

all_columns = [
    columns_summary,
    columns_passing,
    columns_passTypes,
    columns_defensiveActions,
    columns_possession,
    columns_miscanceallousStats
]

leagues = [
    "Premier-League",
    "La-Liga",
    "Bundesliga",
    "Serie-A",
    "Ligue-1",
    "Primeira-Liga",     
    "Eredivisie",        
    "Championship"      
]

seasons = [
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025"
]

player_data_drop = ['league', 'season', 'team', 'home_away', 'shirt_number', 'nation', 'age']
player_data_cat = ['player_name', 'position']

transfer_columns_full = ['player_name', 'inst', 'season', 'transfer_date', 'selling_club', 'buying_club', 'selling_league', 'buying_league', 'coach_past', 'coach_new', 'market_val', 'age_at_transfer', 'contract_left', 'transfer_fee', 'is_loan']
transfer_data_keep = ['player_name', 'selling_club', 'selling_league', 'market_val', 'age_at_transfer', 'transfer_fee', 'transfer_window_idx', 'contract_left_days']
transfer_data_cat = ['player_name', 'selling_club', 'selling_league']
transfer_data_num = list(set(transfer_data_keep) - set(transfer_data_cat))

cat_var = ['position', 'selling_league', 'selling_club']

league_map = {
    "Premier League": "Premier-League",
    "LaLiga": "La-Liga",
    "Bundesliga": "Bundesliga",
    "Serie A": "Serie-A",
    "Ligue 1": "Ligue-1",
    "Liga NOS": "Primeira-Liga",
    "Eredivisie": "Eredivisie", 
    "Championship": "Championship"
}
