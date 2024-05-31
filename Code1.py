import math
from statsbombpy import sb 
from datetime import datetime
import pandas as pd 
from mplsoccer import Pitch, Sbopen 
from matplotlib import rcParams 
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.stats import chi2_contingency 
import sys
from fuzzywuzzy import fuzz 
from difflib import SequenceMatcher
from mplsoccer import Sbopen


parser = Sbopen()
sys.stdout.reconfigure(encoding='utf-8')

data_players = pd.read_csv('players_22.csv')

def trouver_joueur(nom_complet, fichier_csv):
    data_players = pd.read_csv(fichier_csv, low_memory=False)
    resultats = []
    for index, row in data_players.iterrows():
        score = fuzz.token_sort_ratio(nom_complet, row['long_name']) 
        if score >= 90:  
            resultats.append(index)
    return resultats

df_compet = sb.competitions()
df_frame = sb.frames(3857288)
df_match = sb.matches(43,106) 
df_event = sb.events(3857288)
df_lineup = parser.lineup(3869151)
df_lineup = sb.lineups(3869151)
df_match = sb.matches(2,27) 

WC2022_games = [3857273, 3857271, 3857268, 3857265, 3857262, 3857261, 3857255, 3857254, 3857256, 3869151, 3857257, 3857258, 3857288, 3869486, 3857260, 3857264, 3857266, 3857289, 3857269, 3869254, 3869118, 3869519, 3869354, 3869552, 3869253, 3869152, 3857263, 3857259, 3857295, 3857284, 3857282, 3857286, 3857300, 3857299, 3857298, 3857297, 3857293, 3857292, 3857291, 3857290, 3857281, 3857280, 3857279, 3857278, 3857275]


""" Fonction donnant le nombre de joueurs packés lors d'une phase de relance contenant un seul event """
def packing_one_event(event_id, index_event, match_id):
    df_frame = sb.frames(match_id)
    df_event = sb.events(match_id)
    
    before_packed_players = 0
    after_packed_players = 10
    adv_visible = 0

    coord_x_before = df_event['location'][index_event][0] 
    if df_event['type'][index_event] == "Pass":
        coord_x_after = df_event['pass_end_location'][index_event][0] 
    if df_event['type'][index_event] == "Carry":
        coord_x_after = df_event['carry_end_location'][index_event][0]

    for i in range(len(df_frame['id'])) :
        if df_frame['id'][i] == event_id :
            if df_frame['teammate'][i] == False and df_frame['location'][i][0] <= df_event['location'][index_event][0]:
                before_packed_players += 1
    for i in range(len(df_frame['id'])) :
        if df_frame['id'][i] == event_id :
            if df_frame['teammate'][i] == False :
                adv_visible += 1
            if df_frame['teammate'][i] == False and df_frame['location'][i][0] >= df_event['pass_end_location'][index_event][0]:
                after_packed_players -= 1

    packed_players = after_packed_players - before_packed_players

    if adv_visible == 0 :
        return -10
    
    if before_packed_players >= 5 :
        return -20

    return packed_players
    

""" Fonction donnant le nombre de joueurs packés lors d'une phase de relance en particulier """
def packing_relance(first_event_id, last_event_id, index_first_event, index_last_event, match_id):
    df_frame = sb.frames(match_id)
    df_event = sb.events(match_id)
    
    before_packed_players = 0
    after_packed_players = 10
    adv_visible = 0
    coord_x_before = df_event['location'][index_first_event][0] 
    coord_x_after = df_event['location'][index_last_event][0] 

    for i in range(len(df_frame['id'])) :
        if df_frame['id'][i] == first_event_id :
            if df_frame['teammate'][i] == False and df_frame['location'][i][0] <= df_event['location'][index_first_event][0]:
                before_packed_players += 1
    for i in range(len(df_frame['id'])) :
        if df_frame['id'][i] == last_event_id :
            if df_frame['teammate'][i] == False :
                adv_visible += 1
            if df_frame['teammate'][i] == False and df_frame['location'][i][0] >= df_event['location'][index_last_event][0]:
                after_packed_players -= 1
    
    packed_players = after_packed_players - before_packed_players
    if df_event['type'][index_last_event] == "Foul Won" :
        packed_players = 0

    if adv_visible == 0 :
        return -10
    
    if before_packed_players >= 5 :
        return -20

    return packed_players


""" Fonction s'occupant uniquement des events d'une équipe spécifiée """
def timestamp_suiv(index_timestamp_spec, match_id):  
    df_event = sb.events(match_id)
    period = df_event['period'][index_timestamp_spec]
    timestamps = df_event['timestamp']
    teams = df_event['team']
    team_spec = df_event['team'][index_timestamp_spec]

    timestamps = [datetime.strptime(ts, "%H:%M:%S.%f").time() for ts in timestamps]

    timestamp_specifie = datetime.strptime(df_event['timestamp'][index_timestamp_spec], "%H:%M:%S.%f").time()

    indices_first_half = [i for i in range(len(df_event)) if df_event['period'][i] == 1]
    indices_second_half = [i for i in range(len(df_event)) if df_event['period'][i] == 2]

    indices_first_half = sorted(indices_first_half, key=lambda i: timestamps[i])
    indices_second_half = sorted(indices_second_half, key=lambda i: timestamps[i])

    if period == 1:
        indices = indices_first_half
    else:
        indices = indices_second_half

    index_timestamp_specifie = indices.index(index_timestamp_spec)

    i = index_timestamp_specifie + 1
    
    while teams[indices[i]] != team_spec: 
        i += 1
    
    if i < len(indices):
        timestamp_suivant = timestamps[indices[i]]
        index_timestamp = indices[i]
        period_suivant = df_event['period'][index_timestamp]
    else:
        return None, None, None

    return timestamp_suivant, index_timestamp, period_suivant


""" Fonction qui calcule la distance parcourue par un carry """
def distance_entre_points(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


""" Fonction qui calcule l'IMC d'un joueur en fonction de son poids et sa taille """
def calcul_imc(poids, taille_cm):
    # Conversion de la taille en mètres
    taille_m = taille_cm / 100
    
    # Calcul de l'IMC
    imc = poids / (taille_m ** 2)
    
    return imc


### DEGAGEMENT ###
liste_to_add = []
liste_reussi = []
liste_echec = []

cles = [3412, 4231, 433, 442, 3421, 352, 4141, 343, 4411, 41212]

dico_compo = {cle: [0, 0, 0, 0, 0, 0, 0] for cle in cles}

center_back = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], 
               [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]  # [0] : left center back gaucher, [1] : left center back droitier, [2] : right center back gaucher, [3] : right center back droitier

back = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], 
        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]  # [0] : left back gaucher, [1] : left back droitier, [2] : right back gaucher, [3] : right back droitier


""" On part du dégagement et on avance jusqu'à la fin de la phase de relance """
def relance_dégagement(match_id):
    df_event = sb.events(match_id)
    df_lineup = sb.lineups(match_id)

    Type = df_event['type']
    types_exclus = ["Dispossessed", "Error", "Miscontrol", "Block", "Clearance", "Interception", "Pressure", "Foul Committed"]
    numero_relance = 1

    # Variable comptant le nombre de relances ayant un packing -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    compteur_relance = [0, 0, 0, 0, 0, 0, 0]

    # Variable donnant le nombre moyen de passes d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_nb_passes = [0, 0, 0, 0, 0, 0, 0]

    # Variable donnant la longueur moyenne parcourue par la passe d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_long_par_la_passe = [0, 0, 0, 0, 0, 0, 0]

    # Variable donnant le nombre moyen de chaque type de passes pour les relances ayant un packing -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_type_passes = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]] # [0] : Ground, [1] : Low, [2] : High
    
    # variable donnant la duree moyenne d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_duration = [0, 0, 0, 0, 0, 0, 0]
    
    # Variable donnant le nombre moyen de carries d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_nb_carry = [0, 0, 0, 0, 0, 0, 0]

    # Variable donnant la longueur moyenne parcourue par le carry d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_long_par_le_carry = [0, 0, 0, 0, 0, 0, 0]

    # Variable donnant la repartition du nombre de touches moyen par un droitier / gaucher d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_pied_fort = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]] # [0] : gaucher, [1] : droitier
    
    # Variable donnant la repartition (de 1/5 à 5/5) de la qualité de pied faible des joueurs d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_pied_faible = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]] # [0] : 1/5, [1] : 2/5, [2] : 3/5, [3] : 4/5, [4] : 5/5
    
    # Variable donnant la repartition (de 1/5 à 5/5) des skills des joueurs d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_skills = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]] # [0] : 1/5, [1] : 2/5, [2] : 3/5, [3] : 4/5, [4] : 5/5
    
    # Variable donnant la répartition (de 19-20 à 25-26) de l'IMC des joueurs d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_IMC = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]] # [0] : 19-20, [1] : 20-21, [2] : 21-22, [3] : 22-23, [4] : 23-24, [5] : 24-25, [6] : 25-26
    
    # Variable donnant la qualité de passe des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_passing = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90
    
    # Variable donnant la qualité de positioning des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_mentality_pos = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90
    
    # Variable donnant la qualité de vision des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_mentality_vis = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la qualité de long passing des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_long_passing = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la qualité de vitesse des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_pace = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la qualité de controle des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_ball_control = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la qualité de passe courte des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_short_passing = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la repartition des championnats des joueurs d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_league = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]] # [0] : Ligue 1, [1] : Liga, [2] : Bundesliga, [3] : Serie A, [4] : Premier League
    
    # Variable donnant le work-rate des joueurs d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_work_rate = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]] # Le premier est l'apport off et le deuxieme l'apport def : [0] : High/High, [1] : High/Medium, [2] : High/Low, [3] : Medium/High, [4] : Medium/Medium, [5] : Medium/Low, [6] : Low/High, [7] : Low/Medium, [8] : Low/Low

    # Variable donnant la qualité physique des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_physic = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la qualité d'acceleration des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_accel = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90
    
    # Variable donnant la force des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_strength = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90
    
    # Variable donnant l'agilité' des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_agility = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90
    
    # Variable donnant l'agressivité des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_aggression = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90
    
    index_suiv = 0
    nb_nonPress = 0
    nb_press = 0
    first_index = 0

    ratio_press = 0

    compo = []
    team = []

    for index, chaine in enumerate(Type):
        if chaine == "Starting XI" :
            compo.append(df_event['tactics'][index]['formation'])
            team.append(df_event['team'][index])
        if chaine == "Pass" and df_event.loc[index, 'pass_type'] == "Goal Kick": 
            array_hauteur_pass = [0,0,0]
            nb_pass = 0
            pass_length = 0
            dist_carry = 0
            nb_carry = 0
            duration = 0
            nb_press = 0
            nb_nonPress = 0
            first_index = index

            ratio_press = 0

            pref_foot = [0, 0] # index 0 : gaucher, index 1 : droitier

            skills = [0, 0, 0, 0, 0] # 1/5, 2/5, 3/5, 4/5, 5/5

            weak_foot = [0, 0, 0, 0, 0] # 1/5, 2/5, 3/5, 4/5, 5/5

            IMC = [0, 0, 0, 0, 0, 0, 0] # 19-20, 20-21, 21-22, 22-23, 23-24, 24-25, 25-26

            passing = [0, 0, 0, 0, 0, 0]

            mentality_pos = [0, 0, 0, 0, 0, 0]

            mentality_vis = [0, 0, 0, 0, 0, 0]

            long_passing = [0, 0, 0, 0, 0, 0]

            pace = [0, 0, 0, 0, 0, 0]

            ball_control = [0, 0, 0, 0, 0, 0]

            short_passing = [0, 0, 0, 0, 0, 0]

            league = [0, 0, 0, 0, 0]

            work_rate = [0, 0, 0, 0, 0, 0, 0, 0, 0]

            physic = [0, 0, 0, 0, 0, 0]

            accel = [0, 0, 0, 0, 0, 0]

            strength = [0, 0, 0, 0, 0, 0]

            agility = [0, 0, 0, 0, 0, 0]

            aggression = [0, 0, 0, 0, 0, 0]

            def_central = [0, 0, 0, 0]

            back_aile = [0, 0, 0, 0]

            liste_attente = []
        
            time_suiv, index_suiv, periode_suiv = timestamp_suiv(index, match_id)
            start_time = datetime.strptime(df_event['timestamp'][index], "%H:%M:%S.%f")
            index_final = 0
            if len(trouver_joueur(df_event['player'][index], 'players_22.csv')) != 0 :
                team = df_event['team'][index]
                player_id = df_event['player_id'][index]
                for i in range(len(df_lineup[team]['player_id'])):
                    if player_id == df_lineup[team]['player_id'][i] and len(df_lineup[team]['positions'][i]) != 0:
                        if df_lineup[team]['positions'][i][0]['position'] == "Left Center Back" :
                            pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                            if pied == "Left" :
                                def_central[0] += 1
                            elif pied == "Right" :
                                def_central[1] += 1
                        elif df_lineup[team]['positions'][i][0]['position'] == "Right Center Back" :
                            pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                            if pied == "Left" :
                                def_central[2] += 1
                            elif pied == "Right" :
                                def_central[3] += 1
                        elif df_lineup[team]['positions'][i][0]['position'] == "Left Back" :
                            pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                            if pied == "Left" :
                                back_aile[0] += 1
                            elif pied == "Right" :
                                back_aile[1] += 1
                        elif df_lineup[team]['positions'][i][0]['position'] == "Right Back" :
                            pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                            if pied == "Left" :
                                back_aile[2] += 1
                            elif pied == "Right" :
                                back_aile[3] += 1

                if data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == "Left" :
                    pref_foot[0] += 1
                else : 
                    pref_foot[1] += 1

                if data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                    skills[0] += 1
                elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                    skills[1] += 1
                elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                    skills[2] += 1
                elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                    skills[3] += 1
                else :
                    skills[4] += 1

                player_IMC = calcul_imc(data_players['weight_kg'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]], data_players['height_cm'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]])

                if player_IMC <= 20 :
                    IMC[0] += 1
                elif 20 < player_IMC <= 21 :
                    IMC[1] += 1
                elif 21 < player_IMC <= 22 :
                    IMC[2] += 1
                elif 22 < player_IMC <= 23 :
                    IMC[3] += 1
                elif 23 < player_IMC <= 24 :
                    IMC[4] += 1
                elif 24 < player_IMC <= 25 :
                    IMC[5] += 1
                else :
                    IMC[6] += 1

                if data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                    weak_foot[0] += 1
                elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                    weak_foot[1] += 1
                elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                    weak_foot[2] += 1
                elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                    weak_foot[3] += 1
                else :
                    weak_foot[4] += 1

                player_passing = data_players['passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                if data_players['club_position'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] != "GK" :
                    if player_passing <= 50:
                        passing[0] += 1
                    elif 50 < player_passing <= 60:
                        passing[1] += 1
                    elif 60 < player_passing <= 70:
                        passing[2] += 1
                    elif 70 < player_passing <= 80:
                        passing[3] += 1
                    elif 80 < player_passing <= 90:
                        passing[4] += 1
                    else : 
                        passing[5] += 1

                player_mentality_pos = data_players['mentality_positioning'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                if player_mentality_pos <= 50:
                    mentality_pos[0] += 1
                elif 50 < player_mentality_pos <= 60:
                    mentality_pos[1] += 1
                elif 60 < player_mentality_pos <= 70:
                    mentality_pos[2] += 1
                elif 70 < player_mentality_pos <= 80:
                    mentality_pos[3] += 1
                elif 80 < player_mentality_pos <= 90:
                    mentality_pos[4] += 1
                else : 
                    mentality_pos[5] += 1

                player_mentality_vis = data_players['mentality_vision'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                if player_mentality_vis <= 50:
                    mentality_vis[0] += 1
                elif 50 < player_mentality_vis <= 60:
                    mentality_vis[1] += 1
                elif 60 < player_mentality_vis <= 70:
                    mentality_vis[2] += 1
                elif 70 < player_mentality_vis <= 80:
                    mentality_vis[3] += 1
                elif 80 < player_mentality_vis <= 90:
                    mentality_vis[4] += 1
                else : 
                    mentality_vis[5] += 1

                player_long_passing = data_players['skill_long_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                if player_long_passing <= 50:
                    long_passing[0] += 1
                elif 50 < player_long_passing <= 60:
                    long_passing[1] += 1
                elif 60 < player_long_passing <= 70:
                    long_passing[2] += 1
                elif 70 < player_long_passing <= 80:
                    long_passing[3] += 1
                elif 80 < player_long_passing <= 90:
                    long_passing[4] += 1
                else : 
                    long_passing[5] += 1

                player_pace = data_players['pace'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                if player_pace <= 50:
                    pace[0] += 1
                elif 50 < player_pace <= 60:
                    pace[1] += 1
                elif 60 < player_pace <= 70:
                    pace[2] += 1
                elif 70 < player_pace <= 80:
                    pace[3] += 1
                elif 80 < player_pace <= 90:
                    pace[4] += 1
                else : 
                    pace[5] += 1

                player_ball_control = data_players['skill_ball_control'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_ball_control <= 50:
                    ball_control[0] += 1
                elif 50 < player_ball_control <= 60:
                    ball_control[1] += 1
                elif 60 < player_ball_control <= 70:
                    ball_control[2] += 1
                elif 70 < player_ball_control <= 80:
                    ball_control[3] += 1
                elif 80 < player_ball_control <= 90:
                    ball_control[4] += 1
                else : 
                    ball_control[5] += 1
                
                player_short_passing = data_players['attacking_short_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_short_passing <= 50:
                    short_passing[0] += 1
                elif 50 < player_short_passing <= 60:
                    short_passing[1] += 1
                elif 60 < player_short_passing <= 70:
                    short_passing[2] += 1
                elif 70 < player_short_passing <= 80:
                    short_passing[3] += 1
                elif 80 < player_short_passing <= 90:
                    short_passing[4] += 1
                else : 
                    short_passing[5] += 1
                
                player_league = data_players['league_name'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_league == "French Ligue 1":
                    league[0] += 1
                elif player_league == "Spain Primera Division":
                    league[1] += 1
                elif player_league == "German 1. Bundesliga":
                    league[2] += 1
                elif player_league == "Italian Serie A":
                    league[3] += 1
                elif player_league == "English Premier League":
                    league[4] += 1

                player_work_rate = data_players['work_rate'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_work_rate == "High/High":
                    work_rate[0] += 1
                elif player_work_rate == "High/Medium":
                    work_rate[1] += 1
                elif player_work_rate == "High/Low":
                    work_rate[2] += 1
                elif player_work_rate == "Medium/High":
                    work_rate[3] += 1
                elif player_work_rate == "Medium/Medium":
                    work_rate[4] += 1
                elif player_work_rate == "Medium/Low":
                    work_rate[5] += 1
                elif player_work_rate == "Low/High":
                    work_rate[6] += 1
                elif player_work_rate == "Low/Medium":
                    work_rate[7] += 1
                elif player_work_rate == "Low/Low": 
                    work_rate[8] += 1

                player_physic = data_players['physic'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_physic <= 50:
                    physic[0] += 1
                elif 50 < player_physic <= 60:
                    physic[1] += 1
                elif 60 < player_physic <= 70:
                    physic[2] += 1
                elif 70 < player_physic <= 80:
                    physic[3] += 1
                elif 80 < player_physic <= 90:
                    physic[4] += 1
                else : 
                    physic[5] += 1
                
                player_accel = data_players['movement_acceleration'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_accel <= 50:
                    accel[0] += 1
                elif 50 < player_accel <= 60:
                    accel[1] += 1
                elif 60 < player_accel <= 70:
                    accel[2] += 1
                elif 70 < player_accel <= 80:
                    accel[3] += 1
                elif 80 < player_accel <= 90:
                    accel[4] += 1
                else : 
                    accel[5] += 1

                player_strength = data_players['power_strength'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_strength <= 50:
                    strength[0] += 1
                elif 50 < player_strength <= 60:
                    strength[1] += 1
                elif 60 < player_strength <= 70:
                    strength[2] += 1
                elif 70 < player_strength <= 80:
                    strength[3] += 1
                elif 80 < player_strength <= 90:
                    strength[4] += 1
                else : 
                    strength[5] += 1
                
                player_agility = data_players['movement_agility'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_agility <= 50:
                    agility[0] += 1
                elif 50 < player_agility <= 60:
                    agility[1] += 1
                elif 60 < player_agility <= 70:
                    agility[2] += 1
                elif 70 < player_agility <= 80:
                    agility[3] += 1
                elif 80 < player_agility <= 90:
                    agility[4] += 1
                else : 
                    agility[5] += 1

                player_aggression = data_players['mentality_aggression'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_aggression <= 50:
                    aggression[0] += 1
                elif 50 < player_aggression <= 60:
                    aggression[1] += 1
                elif 60 < player_aggression <= 70:
                    aggression[2] += 1
                elif 70 < player_aggression <= 80:
                    aggression[3] += 1
                elif 80 < player_aggression <= 90:
                    aggression[4] += 1
                else : 
                    aggression[5] += 1

            liste_attente.append([df_event['location'][index][0], df_event['location'][index][1], df_event['pass_end_location'][index][0], df_event['pass_end_location'][index][1], df_event['type'][index], numero_relance])
            if isinstance(df_event['location'][index_suiv], list):
                while (df_event['location'][index_suiv][0] < 60 and df_event.loc[index_suiv, 'type'] not in types_exclus and df_event['pass_outcome'][index_suiv] != "Incomplete" and df_event['pass_outcome'][index_suiv] != "Out" and df_event['pass_outcome'][index_suiv] != "Pass Offside" and df_event['pass_outcome'][index_suiv] != "Unknown" and df_event['dribble_outcome'][index_suiv] != "Incomplete") and df_event['ball_receipt_outcome'][index_suiv] != "Incomplete":
                    index = index_suiv
                    
                    if len(trouver_joueur(df_event['player'][index], 'players_22.csv')) != 0 :
                        team = df_event['team'][index]
                        player_id = df_event['player_id'][index]
                        for i in range(len(df_lineup[team]['player_id'])):
                            if player_id == df_lineup[team]['player_id'][i] and len(df_lineup[team]['positions'][i]) != 0:
                                if df_lineup[team]['positions'][i][0]['position'] == "Left Center Back" :
                                    pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                    if pied == "Left" :
                                        def_central[0] += 1
                                    elif pied == "Right" :
                                        def_central[1] += 1
                                elif df_lineup[team]['positions'][i][0]['position'] == "Right Center Back" :
                                    pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                    if pied == "Left" :
                                        def_central[2] += 1
                                    elif pied == "Right" :
                                        def_central[3] += 1
                                elif df_lineup[team]['positions'][i][0]['position'] == "Left Back" :
                                    pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                    if pied == "Left" :
                                        back_aile[0] += 1
                                    elif pied == "Right" :
                                        back_aile[1] += 1
                                elif df_lineup[team]['positions'][i][0]['position'] == "Right Back" :
                                    pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                    if pied == "Left" :
                                        back_aile[2] += 1
                                    elif pied == "Right" :
                                        back_aile[3] += 1
                        
                        if data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == "Left" :
                            pref_foot[0] += 1
                        else : 
                            pref_foot[1] += 1

                        if data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                            skills[0] += 1
                        elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                            skills[1] += 1
                        elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                            skills[2] += 1
                        elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                            skills[3] += 1
                        else :
                            skills[4] += 1

                        player_IMC = calcul_imc(data_players['weight_kg'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]], data_players['height_cm'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]])

                        if player_IMC <= 20 :
                            IMC[0] += 1
                        elif 20 < player_IMC <= 21 :
                            IMC[1] += 1
                        elif 21 < player_IMC <= 22 :
                            IMC[2] += 1
                        elif 22 < player_IMC <= 23 :
                            IMC[3] += 1
                        elif 23 < player_IMC <= 24 :
                            IMC[4] += 1
                        elif 24 < player_IMC <= 25 :
                            IMC[5] += 1
                        else :
                            IMC[6] += 1

                        if data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                            weak_foot[0] += 1
                        elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                            weak_foot[1] += 1
                        elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                            weak_foot[2] += 1
                        elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                            weak_foot[3] += 1
                        else :
                            weak_foot[4] += 1

                        player_passing = data_players['passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                        if data_players['club_position'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] != "GK" :
                            if player_passing <= 50:
                                passing[0] += 1
                            elif 50 < player_passing <= 60:
                                passing[1] += 1
                            elif 60 < player_passing <= 70:
                                passing[2] += 1
                            elif 70 < player_passing <= 80:
                                passing[3] += 1
                            elif 80 < player_passing <= 90:
                                passing[4] += 1
                            else : 
                                passing[5] += 1

                        player_mentality_pos = data_players['mentality_positioning'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                        if player_mentality_pos <= 50:
                            mentality_pos[0] += 1
                        elif 50 < player_mentality_pos <= 60:
                            mentality_pos[1] += 1
                        elif 60 < player_mentality_pos <= 70:
                            mentality_pos[2] += 1
                        elif 70 < player_mentality_pos <= 80:
                            mentality_pos[3] += 1
                        elif 80 < player_mentality_pos <= 90:
                            mentality_pos[4] += 1
                        else : 
                            mentality_pos[5] += 1

                        player_mentality_vis = data_players['mentality_vision'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                        if player_mentality_vis <= 50:
                            mentality_vis[0] += 1
                        elif 50 < player_mentality_vis <= 60:
                            mentality_vis[1] += 1
                        elif 60 < player_mentality_vis <= 70:
                            mentality_vis[2] += 1
                        elif 70 < player_mentality_vis <= 80:
                            mentality_vis[3] += 1
                        elif 80 < player_mentality_vis <= 90:
                            mentality_vis[4] += 1
                        else : 
                            mentality_vis[5] += 1

                        player_long_passing = data_players['skill_long_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                        if player_long_passing <= 50:
                            long_passing[0] += 1
                        elif 50 < player_long_passing <= 60:
                            long_passing[1] += 1
                        elif 60 < player_long_passing <= 70:
                            long_passing[2] += 1
                        elif 70 < player_long_passing <= 80:
                            long_passing[3] += 1
                        elif 80 < player_long_passing <= 90:
                            long_passing[4] += 1
                        else : 
                            long_passing[5] += 1

                        player_pace = data_players['pace'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                        if player_pace <= 50:
                            pace[0] += 1
                        elif 50 < player_pace <= 60:
                            pace[1] += 1
                        elif 60 < player_pace <= 70:
                            pace[2] += 1
                        elif 70 < player_pace <= 80:
                            pace[3] += 1
                        elif 80 < player_pace <= 90:
                            pace[4] += 1
                        else : 
                            pace[5] += 1

                        player_ball_control = data_players['skill_ball_control'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_ball_control <= 50:
                            ball_control[0] += 1
                        elif 50 < player_ball_control <= 60:
                            ball_control[1] += 1
                        elif 60 < player_ball_control <= 70:
                            ball_control[2] += 1
                        elif 70 < player_ball_control <= 80:
                            ball_control[3] += 1
                        elif 80 < player_ball_control <= 90:
                            ball_control[4] += 1
                        else : 
                            ball_control[5] += 1

                        player_short_passing = data_players['attacking_short_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_short_passing <= 50:
                            short_passing[0] += 1
                        elif 50 < player_short_passing <= 60:
                            short_passing[1] += 1
                        elif 60 < player_short_passing <= 70:
                            short_passing[2] += 1
                        elif 70 < player_short_passing <= 80:
                            short_passing[3] += 1
                        elif 80 < player_short_passing <= 90:
                            short_passing[4] += 1
                        else : 
                            short_passing[5] += 1

                        player_league = data_players['league_name'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_league == "French Ligue 1":
                            league[0] += 1
                        elif player_league == "Spain Primera Division":
                            league[1] += 1
                        elif player_league == "German 1. Bundesliga":
                            league[2] += 1
                        elif player_league == "Italian Serie A":
                            league[3] += 1
                        elif player_league == "English Premier League":
                            league[4] += 1

                        player_work_rate = data_players['work_rate'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_work_rate == "High/High":
                            work_rate[0] += 1
                        elif player_work_rate == "High/Medium":
                            work_rate[1] += 1
                        elif player_work_rate == "High/Low":
                            work_rate[2] += 1
                        elif player_work_rate == "Medium/High":
                            work_rate[3] += 1
                        elif player_work_rate == "Medium/Medium":
                            work_rate[4] += 1
                        elif player_work_rate == "Medium/Low":
                            work_rate[5] += 1
                        elif player_work_rate == "Low/High":
                            work_rate[6] += 1
                        elif player_work_rate == "Low/Medium":
                            work_rate[7] += 1
                        elif player_work_rate == "Low/Low": 
                            work_rate[8] += 1

                        player_physic = data_players['physic'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_physic <= 50:
                            physic[0] += 1
                        elif 50 < player_physic <= 60:
                            physic[1] += 1
                        elif 60 < player_physic <= 70:
                            physic[2] += 1
                        elif 70 < player_physic <= 80:
                            physic[3] += 1
                        elif 80 < player_physic <= 90:
                            physic[4] += 1
                        else : 
                            physic[5] += 1

                        player_accel = data_players['movement_acceleration'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_accel <= 50:
                            accel[0] += 1
                        elif 50 < player_accel <= 60:
                            accel[1] += 1
                        elif 60 < player_accel <= 70:
                            accel[2] += 1
                        elif 70 < player_accel <= 80:
                            accel[3] += 1
                        elif 80 < player_accel <= 90:
                            accel[4] += 1
                        else : 
                            accel[5] += 1

                        player_strength = data_players['power_strength'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_strength <= 50:
                            strength[0] += 1
                        elif 50 < player_strength <= 60:
                            strength[1] += 1
                        elif 60 < player_strength <= 70:
                            strength[2] += 1
                        elif 70 < player_strength <= 80:
                            strength[3] += 1
                        elif 80 < player_strength <= 90:
                            strength[4] += 1
                        else : 
                            strength[5] += 1

                        player_agility = data_players['movement_agility'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_agility <= 50:
                            agility[0] += 1
                        elif 50 < player_agility <= 60:
                            agility[1] += 1
                        elif 60 < player_agility <= 70:
                            agility[2] += 1
                        elif 70 < player_agility <= 80:
                            agility[3] += 1
                        elif 80 < player_agility <= 90:
                            agility[4] += 1
                        else : 
                            agility[5] += 1

                        player_aggression = data_players['mentality_aggression'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_aggression <= 50:
                            aggression[0] += 1
                        elif 50 < player_aggression <= 60:
                            aggression[1] += 1
                        elif 60 < player_aggression <= 70:
                            aggression[2] += 1
                        elif 70 < player_aggression <= 80:
                            aggression[3] += 1
                        elif 80 < player_aggression <= 90:
                            aggression[4] += 1
                        else : 
                            aggression[5] += 1

                    if df_event['under_pressure'][index] == True: 
                        nb_press += 1
                    else : 
                        nb_nonPress += 1

                    if df_event['type'][index] == "Pass":
                        if df_event['pass_height'][index] == "Ground Pass":
                            array_hauteur_pass[0] += 1
                        if df_event['pass_height'][index] == "Low Pass":
                            array_hauteur_pass[1] += 1
                        if df_event['pass_height'][index] == "High Pass":   
                            array_hauteur_pass[2] += 1
                        pass_length += df_event['pass_length'][index]
                        nb_pass += 1
                        duration += df_event['duration'][index]
                        liste_attente.append([df_event['location'][index][0], df_event['location'][index][1], df_event['pass_end_location'][index][0], df_event['pass_end_location'][index][1], df_event['type'][index], numero_relance])
                    if df_event['type'][index] == "Carry":
                        nb_carry += 1
                        dist_carry += distance_entre_points(df_event['location'][index][0], df_event['location'][index][1], df_event['carry_end_location'][index][0], df_event['carry_end_location'][index][1])
                        liste_attente.append([df_event['location'][index][0], df_event['location'][index][1], df_event['carry_end_location'][index][0], df_event['carry_end_location'][index][1], df_event['type'][index], numero_relance])
                        duration += df_event['duration'][index]
                    
                    time_suiv, index_suiv, periode_suiv = timestamp_suiv(index, match_id)
                    if not isinstance(df_event['location'][index_suiv], list): 
                        index_suiv = index
                        break
                    index_final = index_suiv
                    
            if index_final == 0 :
                if len(trouver_joueur(df_event['player'][index], 'players_22.csv')) != 0 :
                    team = df_event['team'][index]
                    player_id = df_event['player_id'][index]
                    for i in range(len(df_lineup[team]['player_id'])):
                        if player_id == df_lineup[team]['player_id'][i] and len(df_lineup[team]['positions'][i]) != 0:
                            if df_lineup[team]['positions'][i][0]['position'] == "Left Center Back" :
                                pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                if pied == "Left" :
                                    def_central[0] += 1
                                elif pied == "Right" :
                                    def_central[1] += 1
                            elif df_lineup[team]['positions'][i][0]['position'] == "Right Center Back" :
                                pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                if pied == "Left" :
                                    def_central[2] += 1
                                elif pied == "Right" :
                                    def_central[3] += 1
                            elif df_lineup[team]['positions'][i][0]['position'] == "Left Back" :
                                pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                if pied == "Left" :
                                    back_aile[0] += 1
                                elif pied == "Right" :
                                    back_aile[1] += 1
                            elif df_lineup[team]['positions'][i][0]['position'] == "Right Back" :
                                pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                if pied == "Left" :
                                    back_aile[2] += 1
                                elif pied == "Right" :
                                    back_aile[3] += 1

                    
                    if data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == "Left" :
                        pref_foot[0] += 1
                    else : 
                        pref_foot[1] += 1

                    if data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                        skills[0] += 1
                    elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                        skills[1] += 1
                    elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                        skills[2] += 1
                    elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                        skills[3] += 1
                    else :
                        skills[4] += 1

                    player_IMC = calcul_imc(data_players['weight_kg'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]], data_players['height_cm'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]])

                    if player_IMC <= 20 :
                        IMC[0] += 1
                    elif 20 < player_IMC <= 21 :
                        IMC[1] += 1
                    elif 21 < player_IMC <= 22 :
                        IMC[2] += 1
                    elif 22 < player_IMC <= 23 :
                        IMC[3] += 1
                    elif 23 < player_IMC <= 24 :
                        IMC[4] += 1
                    elif 24 < player_IMC <= 25 :
                        IMC[5] += 1
                    else :
                        IMC[6] += 1

                    if data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                        weak_foot[0] += 1
                    elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                        weak_foot[1] += 1
                    elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                        weak_foot[2] += 1
                    elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                        weak_foot[3] += 1
                    else :
                        weak_foot[4] += 1

                    player_passing = data_players['passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if data_players['club_position'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] != "GK" :
                        if player_passing <= 50:
                            passing[0] += 1
                        elif 50 < player_passing <= 60:
                            passing[1] += 1
                        elif 60 < player_passing <= 70:
                            passing[2] += 1
                        elif 70 < player_passing <= 80:
                            passing[3] += 1
                        elif 80 < player_passing <= 90:
                            passing[4] += 1
                        else : 
                            passing[5] += 1

                    player_mentality_pos = data_players['mentality_positioning'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if player_mentality_pos <= 50:
                        mentality_pos[0] += 1
                    elif 50 < player_mentality_pos <= 60:
                        mentality_pos[1] += 1
                    elif 60 < player_mentality_pos <= 70:
                        mentality_pos[2] += 1
                    elif 70 < player_mentality_pos <= 80:
                        mentality_pos[3] += 1
                    elif 80 < player_mentality_pos <= 90:
                        mentality_pos[4] += 1
                    else : 
                        mentality_pos[5] += 1
                    
                    player_mentality_vis = data_players['mentality_vision'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if player_mentality_vis <= 50:
                        mentality_vis[0] += 1
                    elif 50 < player_mentality_vis <= 60:
                        mentality_vis[1] += 1
                    elif 60 < player_mentality_vis <= 70:
                        mentality_vis[2] += 1
                    elif 70 < player_mentality_vis <= 80:
                        mentality_vis[3] += 1
                    elif 80 < player_mentality_vis <= 90:
                        mentality_vis[4] += 1
                    else : 
                        mentality_vis[5] += 1
                    
                    player_long_passing = data_players['skill_long_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if player_long_passing <= 50:
                        long_passing[0] += 1
                    elif 50 < player_long_passing <= 60:
                        long_passing[1] += 1
                    elif 60 < player_long_passing <= 70:
                        long_passing[2] += 1
                    elif 70 < player_long_passing <= 80:
                        long_passing[3] += 1
                    elif 80 < player_long_passing <= 90:
                        long_passing[4] += 1
                    else : 
                        long_passing[5] += 1
                    
                    player_pace = data_players['pace'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if player_pace <= 50:
                        pace[0] += 1
                    elif 50 < player_pace <= 60:
                        pace[1] += 1
                    elif 60 < player_pace <= 70:
                        pace[2] += 1
                    elif 70 < player_pace <= 80:
                        pace[3] += 1
                    elif 80 < player_pace <= 90:
                        pace[4] += 1
                    else : 
                        pace[5] += 1
                    
                    player_ball_control = data_players['skill_ball_control'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_ball_control <= 50:
                        ball_control[0] += 1
                    elif 50 < player_ball_control <= 60:
                        ball_control[1] += 1
                    elif 60 < player_ball_control <= 70:
                        ball_control[2] += 1
                    elif 70 < player_ball_control <= 80:
                        ball_control[3] += 1
                    elif 80 < player_ball_control <= 90:
                        ball_control[4] += 1
                    else : 
                        ball_control[5] += 1

                    player_short_passing = data_players['attacking_short_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_short_passing <= 50:
                        short_passing[0] += 1
                    elif 50 < player_short_passing <= 60:
                        short_passing[1] += 1
                    elif 60 < player_short_passing <= 70:
                        short_passing[2] += 1
                    elif 70 < player_short_passing <= 80:
                        short_passing[3] += 1
                    elif 80 < player_short_passing <= 90:
                        short_passing[4] += 1
                    else : 
                        short_passing[5] += 1

                    player_league = data_players['league_name'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_league == "French Ligue 1":
                        league[0] += 1
                    elif player_league == "Spain Primera Division":
                        league[1] += 1
                    elif player_league == "German 1. Bundesliga":
                        league[2] += 1
                    elif player_league == "Italian Serie A":
                        league[3] += 1
                    elif player_league == "English Premier League":
                        league[4] += 1

                    player_work_rate = data_players['work_rate'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_work_rate == "High/High":
                        work_rate[0] += 1
                    elif player_work_rate == "High/Medium":
                        work_rate[1] += 1
                    elif player_work_rate == "High/Low":
                        work_rate[2] += 1
                    elif player_work_rate == "Medium/High":
                        work_rate[3] += 1
                    elif player_work_rate == "Medium/Medium":
                        work_rate[4] += 1
                    elif player_work_rate == "Medium/Low":
                        work_rate[5] += 1
                    elif player_work_rate == "Low/High":
                        work_rate[6] += 1
                    elif player_work_rate == "Low/Medium":
                        work_rate[7] += 1
                    elif player_work_rate == "Low/Low": 
                        work_rate[8] += 1

                    player_physic = data_players['physic'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_physic <= 50:
                        physic[0] += 1
                    elif 50 < player_physic <= 60:
                        physic[1] += 1
                    elif 60 < player_physic <= 70:
                        physic[2] += 1
                    elif 70 < player_physic <= 80:
                        physic[3] += 1
                    elif 80 < player_physic <= 90:
                        physic[4] += 1
                    else : 
                        physic[5] += 1

                    player_accel = data_players['movement_acceleration'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_accel <= 50:
                        accel[0] += 1
                    elif 50 < player_accel <= 60:
                        accel[1] += 1
                    elif 60 < player_accel <= 70:
                        accel[2] += 1
                    elif 70 < player_accel <= 80:
                        accel[3] += 1
                    elif 80 < player_accel <= 90:
                        accel[4] += 1
                    else : 
                        accel[5] += 1

                    player_strength = data_players['power_strength'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_strength <= 50:
                        strength[0] += 1
                    elif 50 < player_strength <= 60:
                        strength[1] += 1
                    elif 60 < player_strength <= 70:
                        strength[2] += 1
                    elif 70 < player_strength <= 80:
                        strength[3] += 1
                    elif 80 < player_strength <= 90:
                        strength[4] += 1
                    else : 
                        strength[5] += 1

                    player_agility = data_players['movement_agility'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_agility <= 50:
                        agility[0] += 1
                    elif 50 < player_agility <= 60:
                        agility[1] += 1
                    elif 60 < player_agility <= 70:
                        agility[2] += 1
                    elif 70 < player_agility <= 80:
                        agility[3] += 1
                    elif 80 < player_agility <= 90:
                        agility[4] += 1
                    else : 
                        agility[5] += 1

                    player_aggression = data_players['mentality_aggression'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_aggression <= 50:
                        aggression[0] += 1
                    elif 50 < player_aggression <= 60:
                        aggression[1] += 1
                    elif 60 < player_aggression <= 70:
                        aggression[2] += 1
                    elif 70 < player_aggression <= 80:
                        aggression[3] += 1
                    elif 80 < player_aggression <= 90:
                        aggression[4] += 1
                    else : 
                        aggression[5] += 1
                
                if df_event['under_pressure'][index] == True: 
                    nb_press += 1
                else : 
                    nb_nonPress += 1

                if df_event['type'][index] == "Pass":
                    if df_event['pass_height'][index] == "Ground Pass":
                        array_hauteur_pass[0] += 1
                    if df_event['pass_height'][index] == "Low Pass":
                        array_hauteur_pass[1] += 1
                    if df_event['pass_height'][index] == "High Pass":
                        array_hauteur_pass[2] += 1
                    pass_length += df_event['pass_length'][index]
                    nb_pass += 1
                    duration += df_event['duration'][index]
                    liste_attente.append([df_event['location'][index][0], df_event['location'][index][1], df_event['pass_end_location'][index][0], df_event['pass_end_location'][index][1], df_event['type'][index], numero_relance])
                if df_event['type'][index] == "Carry":
                    nb_carry += 1
                    dist_carry += distance_entre_points(df_event['location'][index][0], df_event['location'][index][1], df_event['carry_end_location'][index][0], df_event['carry_end_location'][index][1])   
                    duration += df_event['duration'][index]
                    liste_attente.append([df_event['location'][index][0], df_event['location'][index][1], df_event['carry_end_location'][index][0], df_event['carry_end_location'][index][1], df_event['type'][index], numero_relance])
                
                if df_event['pass_end_location'][index][0] > 60 and df_event['pass_outcome'][index] != "Incomplete" and df_event['pass_outcome'][index] != "Out" and df_event['pass_outcome'][index] != "Pass Offside" and df_event['pass_outcome'][index] != "Unknown":
                    for i in range(len(liste_attente)):
                        liste_reussi.append(liste_attente[i])

                    packing = packing_one_event(df_event['id'][index], index, match_id)
                    if packing == 0 or packing == 1 :
                        moyenne_duration[1] += duration
                        compteur_relance[1] += 1
                        moyenne_type_passes[1][0] += array_hauteur_pass[0]
                        moyenne_type_passes[1][1] += array_hauteur_pass[1]
                        moyenne_type_passes[1][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[1] += nb_pass
                        moyenne_long_par_la_passe[1] += pass_length
                        moyenne_nb_carry[1] += nb_carry
                        moyenne_long_par_le_carry[1] += dist_carry
                        moyenne_pied_fort[1][0] += pref_foot[0]
                        moyenne_pied_fort[1][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[1][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[1][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[1][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[1][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[1][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[1][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[1][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[1][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[1][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[1][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[1][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[1][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[1][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[1][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[1][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[1][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[1][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[1][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[1][j] += back_aile[j]

                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][1] += 1
                        else : 
                            dico_compo[compo[1]][1] += 1
                        
                    if packing == 2 or packing == 3 :
                        moyenne_duration[2] += duration
                        compteur_relance[2] += 1
                        moyenne_type_passes[2][0] += array_hauteur_pass[0]
                        moyenne_type_passes[2][1] += array_hauteur_pass[1]
                        moyenne_type_passes[2][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[2] += nb_pass
                        moyenne_long_par_la_passe[2] += pass_length
                        moyenne_nb_carry[2] += nb_carry
                        moyenne_long_par_le_carry[2] += dist_carry
                        moyenne_pied_fort[2][0] += pref_foot[0]
                        moyenne_pied_fort[2][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[2][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[2][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[2][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[2][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[2][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[2][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[2][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[2][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[2][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[2][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[2][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[2][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[2][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[2][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[2][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[2][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[2][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[2][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[2][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][2] += 1
                        else : 
                            dico_compo[compo[1]][2] += 1
                    if packing == 4 or packing == 5 :
                        moyenne_duration[3] += duration
                        compteur_relance[3] += 1
                        moyenne_type_passes[3][0] += array_hauteur_pass[0]
                        moyenne_type_passes[3][1] += array_hauteur_pass[1]
                        moyenne_type_passes[3][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[3] += nb_pass
                        moyenne_long_par_la_passe[3] += pass_length
                        moyenne_nb_carry[3] += nb_carry
                        moyenne_long_par_le_carry[3] += dist_carry
                        moyenne_pied_fort[3][0] += pref_foot[0]
                        moyenne_pied_fort[3][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[3][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[3][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[3][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[3][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[3][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[3][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[3][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[3][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[3][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[3][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[3][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[3][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[3][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[3][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[3][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[3][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[3][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[3][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[3][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][3] += 1
                        else : 
                            dico_compo[compo[1]][3] += 1
                    if packing == 6 or packing == 7 :
                        moyenne_duration[4] += duration
                        compteur_relance[4] += 1
                        moyenne_type_passes[4][0] += array_hauteur_pass[0]
                        moyenne_type_passes[4][1] += array_hauteur_pass[1]
                        moyenne_type_passes[4][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[4] += nb_pass
                        moyenne_long_par_la_passe[4] += pass_length
                        moyenne_nb_carry[4] += nb_carry
                        moyenne_long_par_le_carry[4] += dist_carry
                        moyenne_pied_fort[4][0] += pref_foot[0]
                        moyenne_pied_fort[4][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[4][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[4][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[4][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[4][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[4][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[4][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[4][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[4][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[4][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[4][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[4][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[4][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[4][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[4][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[4][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[4][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[4][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[4][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[4][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][4] += 1
                        else : 
                            dico_compo[compo[1]][4] += 1
                    if packing == 8 or packing == 9 :
                        moyenne_duration[5] += duration
                        compteur_relance[5] += 1
                        moyenne_type_passes[5][0] += array_hauteur_pass[0]
                        moyenne_type_passes[5][1] += array_hauteur_pass[1]
                        moyenne_type_passes[5][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[5] += nb_pass
                        moyenne_long_par_la_passe[5] += pass_length
                        moyenne_nb_carry[5] += nb_carry
                        moyenne_long_par_le_carry[5] += dist_carry
                        moyenne_pied_fort[5][0] += pref_foot[0]
                        moyenne_pied_fort[5][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[5][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[5][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[5][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[5][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[5][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[5][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[5][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[5][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[5][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[5][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[5][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[5][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[5][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[5][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[5][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[5][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[5][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[5][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[5][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][5] += 1
                        else : 
                            dico_compo[compo[1]][5] += 1
                    if packing == 10 :
                        moyenne_duration[6] += duration
                        compteur_relance[6] += 1
                        moyenne_type_passes[6][0] += array_hauteur_pass[0]
                        moyenne_type_passes[6][1] += array_hauteur_pass[1]
                        moyenne_type_passes[6][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[6] += nb_pass                            
                        moyenne_long_par_la_passe[6] += pass_length
                        moyenne_nb_carry[6] += nb_carry
                        moyenne_long_par_le_carry[6] += dist_carry
                        moyenne_pied_fort[6][0] += pref_foot[0]
                        moyenne_pied_fort[6][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[6][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[6][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[6][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[6][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[6][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[6][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[6][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[6][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[6][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[6][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[6][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[6][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[6][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[6][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[6][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[6][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[6][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[6][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[6][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][6] += 1
                        else : 
                            dico_compo[compo[1]][6] += 1

                else :
        
                    if df_event['type'][index_suiv] == "Pass" :
                        if df_event['pass_height'][index_suiv] == "Ground Pass":
                            array_hauteur_pass[0] += 1
                        if df_event['pass_height'][index_suiv] == "Low Pass":
                            array_hauteur_pass[1] += 1
                        if df_event['pass_height'][index_suiv] == "High Pass":
                            array_hauteur_pass[2] += 1
                        pass_length += df_event['pass_length'][index_suiv]
                        nb_pass += 1
                        duration += df_event['duration'][index_suiv]
                        liste_attente.append([df_event['location'][index_suiv][0], df_event['location'][index_suiv][1], df_event['pass_end_location'][index_suiv][0], df_event['pass_end_location'][index_suiv][1], df_event['type'][index_suiv], numero_relance])
                    if df_event['type'][index_suiv] == "Carry" :
                        nb_carry += 1
                        dist_carry += distance_entre_points(df_event['location'][index_suiv][0], df_event['location'][index_suiv][1], df_event['carry_end_location'][index_suiv][0], df_event['carry_end_location'][index_suiv][1])
                        duration += df_event['duration'][index_suiv]
                        liste_attente.append([df_event['location'][index_suiv][0], df_event['location'][index_suiv][1], df_event['carry_end_location'][index_suiv][0], df_event['carry_end_location'][index_suiv][1], df_event['type'][index_suiv], numero_relance])
                    for i in range(len(liste_attente)):
                        liste_echec.append(liste_attente[i])
        
                    compteur_relance[0] += 1
                    moyenne_duration[0] += duration 
                    moyenne_type_passes[0][0] += array_hauteur_pass[0]
                    moyenne_type_passes[0][1] += array_hauteur_pass[1]
                    moyenne_type_passes[0][2] += array_hauteur_pass[2]
                    moyenne_nb_passes[0] += nb_pass
                    moyenne_long_par_la_passe[0] += pass_length
                    moyenne_nb_carry[0] += nb_carry
                    moyenne_long_par_le_carry[0] += dist_carry
                    moyenne_pied_fort[0][0] += pref_foot[0]
                    moyenne_pied_fort[0][1] += pref_foot[1]
                    for j in range(len(weak_foot)):
                        moyenne_pied_faible[0][j] += weak_foot[j]
                    for j in range(len(skills)):
                        moyenne_skills[0][j] += skills[j]
                    for j in range(len(IMC)):
                        moyenne_IMC[0][j] += IMC[j]
                    for j in range(len(passing)):
                        moyenne_passing[0][j] += passing[j]
                    for j in range(len(mentality_pos)):
                        moyenne_mentality_pos[0][j] += mentality_pos[j]
                    for j in range(len(mentality_vis)):
                        moyenne_mentality_vis[0][j] += mentality_vis[j]
                    for j in range(len(long_passing)):
                        moyenne_long_passing[0][j] += long_passing[j]
                    for j in range(len(pace)):
                        moyenne_pace[0][j] += pace[j]
                    for j in range(len(ball_control)):
                        moyenne_ball_control[0][j] += ball_control[j]
                    for j in range(len(short_passing)):
                        moyenne_short_passing[0][j] += short_passing[j]
                    for j in range(len(league)):
                        moyenne_league[0][j] += league[j]
                    for j in range(len(work_rate)):
                        moyenne_work_rate[0][j] += work_rate[j]
                    for j in range(len(physic)):
                        moyenne_physic[0][j] += physic[j]
                    for j in range(len(accel)):
                        moyenne_accel[0][j] += accel[j]
                    for j in range(len(strength)):
                        moyenne_strength[0][j] += strength[j]
                    for j in range(len(agility)):
                        moyenne_agility[0][j] += agility[j]
                    for j in range(len(aggression)):
                        moyenne_aggression[0][j] += aggression[j]
                    for j in range(len(def_central)):
                        center_back[0][j] += def_central[j]
                    for j in range(len(back_aile)):
                        back[0][j] += back_aile[j]
                    
                    if df_event['team'][index] == team[0]:
                        dico_compo[compo[0]][0] += 1
                    else : 
                        dico_compo[compo[1]][0] += 1
                    
                numero_relance += 1
                ratio_press = nb_press / (nb_press +nb_nonPress)
            else : 
                end_time = datetime.strptime(df_event['timestamp'][index_suiv], "%H:%M:%S.%f")
                
                time_difference = end_time - start_time
                time_difference_sec = time_difference.total_seconds()
                duration = time_difference_sec
                
                if (df_event['type'][index] == "Ball Receipt*" and df_event['ball_receipt_outcome'][index] == "Incomplete") or df_event['pass_outcome'][index_suiv] == "Incomplete" or df_event['dribble_outcome'][index_final] == "Incomplete" or df_event['type'][index_final] == "Dispossessed":
                    for i in range(len(liste_attente)):
                        liste_echec.append(liste_attente[i])

                    compteur_relance[0] += 1
                    moyenne_duration[0] += duration
                    moyenne_type_passes[0][0] += array_hauteur_pass[0]
                    moyenne_type_passes[0][1] += array_hauteur_pass[1]
                    moyenne_type_passes[0][2] += array_hauteur_pass[2]
                    moyenne_nb_passes[0] += nb_pass
                    moyenne_long_par_la_passe[0] += pass_length
                    moyenne_nb_carry[0] += nb_carry
                    moyenne_long_par_le_carry[0] += dist_carry
                    moyenne_pied_fort[0][0] += pref_foot[0]
                    moyenne_pied_fort[0][1] += pref_foot[1]
                    for j in range(len(weak_foot)):
                        moyenne_pied_faible[0][j] += weak_foot[j]
                    for j in range(len(skills)):
                        moyenne_skills[0][j] += skills[j]
                    for j in range(len(IMC)):
                        moyenne_IMC[0][j] += IMC[j]
                    for j in range(len(passing)):
                        moyenne_passing[0][j] += passing[j]
                    for j in range(len(mentality_pos)):
                        moyenne_mentality_pos[0][j] += mentality_pos[j]
                    for j in range(len(mentality_vis)):
                        moyenne_mentality_vis[0][j] += mentality_vis[j]
                    for j in range(len(long_passing)):
                        moyenne_long_passing[0][j] += long_passing[j]
                    for j in range(len(pace)):
                        moyenne_pace[0][j] += pace[j]
                    for j in range(len(ball_control)):
                        moyenne_ball_control[0][j] += ball_control[j]
                    for j in range(len(short_passing)):
                        moyenne_short_passing[0][j] += short_passing[j]
                    for j in range(len(league)):
                        moyenne_league[0][j] += league[j]
                    for j in range(len(work_rate)):
                        moyenne_work_rate[0][j] += work_rate[j]
                    for j in range(len(physic)):
                        moyenne_physic[0][j] += physic[j]
                    for j in range(len(accel)):
                        moyenne_accel[0][j] += accel[j]
                    for j in range(len(strength)):
                        moyenne_strength[0][j] += strength[j]
                    for j in range(len(agility)):
                        moyenne_agility[0][j] += agility[j]
                    for j in range(len(aggression)):
                        moyenne_aggression[0][j] += aggression[j]
                    for j in range(len(def_central)):
                        center_back[0][j] += def_central[j]
                    for j in range(len(back_aile)):
                        back[0][j] += back_aile[j]
                    
                    if df_event['team'][index] == team[0]:
                        dico_compo[compo[0]][0] += 1
                    else : 
                        dico_compo[compo[1]][0] += 1
                    
                else :
                    for i in range(len(liste_attente)):
                        liste_reussi.append(liste_attente[i])

                    packing = packing_relance(df_event['id'][first_index], df_event['id'][index_final], first_index, index_final, match_id)
                    if packing == 0 or packing == 1 :
                        moyenne_duration[1] += duration
                        compteur_relance[1] += 1
                        moyenne_type_passes[1][0] += array_hauteur_pass[0]
                        moyenne_type_passes[1][1] += array_hauteur_pass[1]
                        moyenne_type_passes[1][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[1] += nb_pass
                        moyenne_long_par_la_passe[1] += pass_length
                        moyenne_nb_carry[1] += nb_carry
                        moyenne_long_par_le_carry[1] += dist_carry
                        moyenne_pied_fort[1][0] += pref_foot[0]
                        moyenne_pied_fort[1][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[1][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[1][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[1][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[1][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[1][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[1][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[1][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[1][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[1][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[1][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[1][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[1][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[1][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[1][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[1][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[1][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[1][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[1][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[1][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][1] += 1
                        else : 
                            dico_compo[compo[1]][1] += 1
                    if packing == 2 or packing == 3 :
                        moyenne_duration[2] += duration
                        compteur_relance[2] += 1
                        moyenne_type_passes[2][0] += array_hauteur_pass[0]
                        moyenne_type_passes[2][1] += array_hauteur_pass[1]
                        moyenne_type_passes[2][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[2] += nb_pass
                        moyenne_long_par_la_passe[2] += pass_length
                        moyenne_nb_carry[2] += nb_carry
                        moyenne_long_par_le_carry[2] += dist_carry
                        moyenne_pied_fort[2][0] += pref_foot[0]
                        moyenne_pied_fort[2][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[2][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[2][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[2][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[2][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[2][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[2][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[2][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[2][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[2][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[2][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[2][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[2][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[2][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[2][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[2][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[2][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[2][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[2][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[2][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][2] += 1
                        else : 
                            dico_compo[compo[1]][2] += 1
                    if packing == 4 or packing == 5 :
                        moyenne_duration[3] += duration
                        compteur_relance[3] += 1
                        moyenne_type_passes[3][0] += array_hauteur_pass[0]
                        moyenne_type_passes[3][1] += array_hauteur_pass[1]
                        moyenne_type_passes[3][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[3] += nb_pass
                        moyenne_long_par_la_passe[3] += pass_length
                        moyenne_nb_carry[3] += nb_carry
                        moyenne_long_par_le_carry[3] += dist_carry
                        moyenne_pied_fort[3][0] += pref_foot[0]
                        moyenne_pied_fort[3][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[3][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[3][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[3][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[3][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[3][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[3][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[3][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[3][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[3][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[3][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[3][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[3][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[3][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[3][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[3][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[3][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[3][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[3][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[3][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][3] += 1
                        else : 
                            dico_compo[compo[1]][3] += 1
                    if packing == 6 or packing == 7 :
                        moyenne_duration[4] += duration
                        compteur_relance[4] += 1
                        moyenne_type_passes[4][0] += array_hauteur_pass[0]
                        moyenne_type_passes[4][1] += array_hauteur_pass[1]
                        moyenne_type_passes[4][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[4] += nb_pass
                        moyenne_long_par_la_passe[4] += pass_length
                        moyenne_nb_carry[4] += nb_carry
                        moyenne_long_par_le_carry[4] += dist_carry
                        moyenne_pied_fort[4][0] += pref_foot[0]
                        moyenne_pied_fort[4][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[4][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[4][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[4][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[4][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[4][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[4][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[4][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[4][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[4][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[4][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[4][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[4][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[4][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[4][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[4][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[4][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[4][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[4][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[4][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][4] += 1
                        else : 
                            dico_compo[compo[1]][4] += 1
                    if packing == 8 or packing == 9 :
                        moyenne_duration[5] += duration
                        compteur_relance[5] += 1
                        moyenne_type_passes[5][0] += array_hauteur_pass[0]
                        moyenne_type_passes[5][1] += array_hauteur_pass[1]
                        moyenne_type_passes[5][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[5] += nb_pass
                        moyenne_long_par_la_passe[5] += pass_length
                        moyenne_nb_carry[5] += nb_carry
                        moyenne_long_par_le_carry[5] += dist_carry
                        moyenne_pied_fort[5][0] += pref_foot[0]
                        moyenne_pied_fort[5][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[5][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[5][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[5][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[5][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[5][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[5][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[5][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[5][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[5][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[5][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[5][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[5][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[5][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[5][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[5][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[5][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[5][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[5][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[5][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][5] += 1
                        else : 
                            dico_compo[compo[1]][5] += 1
                    if packing == 10 :
                        moyenne_duration[6] += duration
                        compteur_relance[6] += 1
                        moyenne_type_passes[6][0] += array_hauteur_pass[0]
                        moyenne_type_passes[6][1] += array_hauteur_pass[1]
                        moyenne_type_passes[6][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[6] += nb_pass
                        moyenne_long_par_la_passe[6] += pass_length
                        moyenne_nb_carry[6] += nb_carry
                        moyenne_long_par_le_carry[6] += dist_carry
                        moyenne_pied_fort[6][0] += pref_foot[0]
                        moyenne_pied_fort[6][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[6][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[6][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[6][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[6][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[6][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[6][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[6][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[6][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[6][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[6][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[6][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[6][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[6][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[6][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[6][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[6][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[6][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[6][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[6][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][6] += 1
                        else : 
                            dico_compo[compo[1]][6] += 1


                numero_relance += 1
                ratio_press = nb_press / (nb_press +nb_nonPress)
   
    for i in range(len(moyenne_duration)):
        if compteur_relance[i] != 0 :
            moyenne_duration[i] /= compteur_relance[i]
            moyenne_nb_passes[i] /= compteur_relance[i]
            moyenne_long_par_la_passe[i] /= compteur_relance[i]
            moyenne_nb_carry[i] /= compteur_relance[i]
            moyenne_long_par_le_carry[i] /= compteur_relance[i]
            
    return moyenne_duration, moyenne_type_passes, moyenne_nb_passes, moyenne_long_par_la_passe, moyenne_nb_carry, moyenne_long_par_le_carry, moyenne_pied_fort, moyenne_pied_faible, moyenne_skills, moyenne_IMC, moyenne_passing, moyenne_mentality_pos, moyenne_mentality_vis, moyenne_long_passing, moyenne_pace, moyenne_ball_control, moyenne_short_passing, moyenne_league, moyenne_work_rate, moyenne_physic, moyenne_accel, moyenne_strength, moyenne_agility, moyenne_aggression
    
duration = [0, 0, 0, 0, 0, 0, 0]
type_passes = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
nb_passes = [0, 0, 0, 0, 0, 0, 0]
long_par_la_passe = [0, 0, 0, 0, 0, 0, 0]
nb_carry = [0, 0, 0, 0, 0, 0, 0]
long_par_le_carry = [0, 0, 0, 0, 0, 0, 0]
pied_fort = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
pied_faible = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
skills = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
IMC = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
passing = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], 
           [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
mentality_pos = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
mentality_vis = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
long_passing = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
pace = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
ball_control = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
short_passing = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
league = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
work_rate = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
physic = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], 
          [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
accel = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], 
         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
strength = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
agility = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], 
           [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
aggression = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], 
              [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]

compteur_duration = [0, 0, 0, 0, 0, 0, 0] 
compteur_pied_fort = [0, 0, 0, 0, 0, 0, 0] 

for i in WC2022_games :
    mean_duration, mean_type_passes, mean_nb_passes, mean_long_par_la_passe, mean_nb_carry, mean_long_par_le_carry, mean_pied_fort, mean_pied_faible, mean_skills, mean_IMC, mean_passing, mean_mentality_pos, mean_mentality_vis, mean_long_passing, mean_pace, mean_ball_control, mean_short_passing, mean_league, mean_work_rate, mean_physic, mean_accel, mean_strength, mean_agility, mean_aggression = relance_dégagement(i)
    
    for j in range(len(mean_duration)):
        if mean_duration[j] != 0:
            compteur_duration[j] += 1
    for j in range(len(mean_pied_fort)):
        if mean_pied_fort[j][0] != 0 or mean_pied_fort[j][1] != 0 :
            compteur_pied_fort[j] += 1

    for j in range(len(duration)):
        duration[j] += mean_duration[j]
        nb_passes[j] += mean_nb_passes[j]
        long_par_la_passe[j] += mean_long_par_la_passe[j]
        nb_carry[j] += mean_nb_carry[j]
        long_par_le_carry[j] += mean_long_par_le_carry[j]
        for k in range(len(type_passes[0])):
            type_passes[j][k] += mean_type_passes[j][k]
        for k in range(len(pied_fort[0])):
            pied_fort[j][k] += mean_pied_fort[j][k]
        for k in range(len(pied_faible[0])):
            pied_faible[j][k] += mean_pied_faible[j][k]
        for k in range(len(skills[0])):
            skills[j][k] += mean_skills[j][k]
        for k in range(len(IMC[0])):
            IMC[j][k] += mean_IMC[j][k]
        for k in range(len(passing[0])):
            passing[j][k] += mean_passing[j][k]
        for k in range(len(mentality_pos[0])):
            mentality_pos[j][k] += mean_mentality_pos[j][k]
        for k in range(len(mentality_vis[0])):
            mentality_vis[j][k] += mean_mentality_vis[j][k]
        for k in range(len(long_passing[0])):
            long_passing[j][k] += mean_long_passing[j][k]
        for k in range(len(pace[0])):
            pace[j][k] += mean_pace[j][k]
        for k in range(len(ball_control[0])):
            ball_control[j][k] += mean_ball_control[j][k]
        for k in range(len(short_passing[0])):
            short_passing[j][k] += mean_short_passing[j][k]
        for k in range(len(league[0])):
            league[j][k] += mean_league[j][k]
        for k in range(len(work_rate[0])):
            work_rate[j][k] += mean_work_rate[j][k]
        for k in range(len(physic[0])):
            physic[j][k] += mean_physic[j][k]
        for k in range(len(accel[0])):
            accel[j][k] += mean_accel[j][k]
        for k in range(len(strength[0])):
            strength[j][k] += mean_strength[j][k]
        for k in range(len(agility[0])):
            agility[j][k] += mean_agility[j][k]
        for k in range(len(aggression[0])):
            aggression[j][k] += mean_aggression[j][k]


### BALL RECOVERY ###
liste_to_add = []
liste_reussi = []
liste_echec = []


""" On part de ball recovery et on avance jusqu'à la fin de la phase de relance """
def relance_ball_recovery(match_id):
    df_event = sb.events(match_id)
    df_lineup = sb.lineups(match_id)

    Type = df_event['type']
    types_exclus = ["Dispossessed", "Error", "Miscontrol", "Block", "Clearance", "Interception", "Pressure", "Foul Committed"]
    numero_relance = 1

    # Variable comptant le nombre de relances ayant un packing -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    compteur_relance = [0, 0, 0, 0, 0, 0, 0]

    # Variable donnant le nombre moyen de passes d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_nb_passes = [0, 0, 0, 0, 0, 0, 0]

    # Variable donnant la longueur moyenne parcourue par la passe d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_long_par_la_passe = [0, 0, 0, 0, 0, 0, 0]

    # Variable donnant le nombre moyen de chaque type de passes pour les relances ayant un packing -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_type_passes = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]] # [0] : Ground, [1] : Low, [2] : High
    
    # variable donnant la duree moyenne d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_duration = [0, 0, 0, 0, 0, 0, 0]
    
    # Variable donnant le nombre moyen de carries d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_nb_carry = [0, 0, 0, 0, 0, 0, 0]

    # Variable donnant la longueur moyenne parcourue par le carry d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_long_par_le_carry = [0, 0, 0, 0, 0, 0, 0]

    # Variable donnant la repartition du nombre de touches moyen par un droitier / gaucher d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_pied_fort = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]] # [0] : gaucher, [1] : droitier
    
    # Variable donnant la repartition (de 1/5 à 5/5) de la qualité de pied faible des joueurs d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_pied_faible = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]] # [0] : 1/5, [1] : 2/5, [2] : 3/5, [3] : 4/5, [4] : 5/5
    
    # Variable donnant la repartition (de 1/5 à 5/5) des skills des joueurs d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_skills = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]] # [0] : 1/5, [1] : 2/5, [2] : 3/5, [3] : 4/5, [4] : 5/5
    
    # Variable donnant la répartition (de 19-20 à 25-26) de l'IMC des joueurs d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_IMC = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]] # [0] : 19-20, [1] : 20-21, [2] : 21-22, [3] : 22-23, [4] : 23-24, [5] : 24-25, [6] : 25-26
    
    # Variable donnant la qualité de passe des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_passing = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90
    
    # Variable donnant la qualité de mentality positioning des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_mentality_pos = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90
    
    # Variable donnant la qualité de mentality vision des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_mentality_vis = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90
    
    # Variable donnant la qualité de long passing des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_long_passing = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la qualité de vitesse des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_pace = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la qualité de controle des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_ball_control = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la qualité de passe courte des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_short_passing = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la repartition des championnats des joueurs d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_league = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]] # [0] : Ligue 1, [1] : Liga, [2] : Bundesliga, [3] : Serie A, [4] : Premier League

    # Variable donnant le work-rate des joueurs d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_work_rate = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]] # Le premier est l'apport off et le deuxieme l'apport def : [0] : High/High, [1] : High/Medium, [2] : High/Low, [3] : Medium/High, [4] : Medium/Medium, [5] : Medium/Low, [6] : Low/High, [7] : Low/Medium, [8] : Low/Low

    # Variable donnant la qualité physique des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_physic = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la qualité d'acceleration des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_accel = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90
    
    # Variable donnant la force des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_strength = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90
    
    # Variable donnant l'agilité' des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_agility = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90
    
    # Variable donnant l'agressivité des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_aggression = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90
    
    index_suiv = 0
    nb_nonPress = 0
    nb_press = 0
    first_index = 0

    nb_joueurs_trouv = 0
    nb_total = 0

    compo = []
    team = []
    
    for index, chaine in enumerate(Type):
        if chaine == "Starting XI" : 
            compo.append(df_event['tactics'][index]['formation'])
            team.append(df_event['team'][index])
        if chaine == "Ball Recovery" and df_event['location'][index][0] <= 30:
            array_hauteur_pass = [0,0,0]
            nb_pass = 0
            pass_length = 0
            dist_carry = 0
            nb_carry = 0
            duration = 0
            nb_press = 0
            nb_nonPress = 0
            first_index = index

            pref_foot = [0, 0] # index 0 : gaucher, index 1 : droitier

            skills = [0, 0, 0, 0, 0] # 1/5, 2/5, 3/5, 4/5, 5/5

            weak_foot = [0, 0, 0, 0, 0] # 1/5, 2/5, 3/5, 4/5, 5/5

            IMC = [0, 0, 0, 0, 0, 0, 0] # 19-20, 20-21, 21-22, 22-23, 23-24, 24-25, 25-26

            passing = [0, 0, 0, 0, 0, 0] # 0-50, 50-60, 60-70, 70-80, 80-90, 90-100

            mentality_pos = [0, 0, 0, 0, 0, 0] # 0-50, 50-60, 60-70, 70-80, 80-90, 90-100

            mentality_vis = [0, 0, 0, 0, 0, 0] # 0-50, 50-60, 60-70, 70-80, 80-90, 90-100

            long_passing = [0, 0, 0, 0, 0, 0]

            pace = [0, 0, 0, 0, 0, 0]

            ball_control = [0, 0, 0, 0, 0, 0]

            short_passing = [0, 0, 0, 0, 0, 0]

            league = [0, 0, 0, 0, 0]

            work_rate = [0, 0, 0, 0, 0, 0, 0, 0, 0]

            physic = [0, 0, 0, 0, 0, 0]

            accel = [0, 0, 0, 0, 0, 0]

            strength = [0, 0, 0, 0, 0, 0]

            agility = [0, 0, 0, 0, 0, 0]

            aggression = [0, 0, 0, 0, 0, 0]

            def_central = [0, 0, 0, 0]

            back_aile = [0, 0, 0, 0]

            liste_attente = []
            
            start_time = datetime.strptime(df_event['timestamp'][index], "%H:%M:%S.%f")
            
            time_suiv, index_suiv, periode_suiv = timestamp_suiv(index, match_id)
            
            index_final = 0

            nb_total += 1
            if len(trouver_joueur(df_event['player'][index], 'players_22.csv')) != 0 :
                team = df_event['team'][index]
                player_id = df_event['player_id'][index]
                for i in range(len(df_lineup[team]['player_id'])):
                    if player_id == df_lineup[team]['player_id'][i] and len(df_lineup[team]['positions'][i]) != 0:
                        if df_lineup[team]['positions'][i][0]['position'] == "Left Center Back" :
                            pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                            if pied == "Left" :
                                def_central[0] += 1
                            elif pied == "Right" :
                                def_central[1] += 1
                        elif df_lineup[team]['positions'][i][0]['position'] == "Right Center Back" :
                            pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                            if pied == "Left" :
                                def_central[2] += 1
                            elif pied == "Right" :
                                def_central[3] += 1
                        elif df_lineup[team]['positions'][i][0]['position'] == "Left Back" :
                            pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                            if pied == "Left" :
                                back_aile[0] += 1
                            elif pied == "Right" :
                                back_aile[1] += 1
                        elif df_lineup[team]['positions'][i][0]['position'] == "Right Back" :
                            pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                            if pied == "Left" :
                                back_aile[2] += 1
                            elif pied == "Right" :
                                back_aile[3] += 1


                nb_joueurs_trouv += 1
                if data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == "Left" :
                    pref_foot[0] += 1
                else : 
                    pref_foot[1] += 1

                if data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                    skills[0] += 1
                elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                    skills[1] += 1
                elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                    skills[2] += 1
                elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                    skills[3] += 1
                else :
                    skills[4] += 1

                player_IMC = calcul_imc(data_players['weight_kg'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]], data_players['height_cm'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]])

                if player_IMC <= 20 :
                    IMC[0] += 1
                elif 20 < player_IMC <= 21 :
                    IMC[1] += 1
                elif 21 < player_IMC <= 22 :
                    IMC[2] += 1
                elif 22 < player_IMC <= 23 :
                    IMC[3] += 1
                elif 23 < player_IMC <= 24 :
                    IMC[4] += 1
                elif 24 < player_IMC <= 25 :
                    IMC[5] += 1
                else :
                    IMC[6] += 1

                if data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                    weak_foot[0] += 1
                elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                    weak_foot[1] += 1
                elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                    weak_foot[2] += 1
                elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                    weak_foot[3] += 1
                else :
                    weak_foot[4] += 1

                player_passing = data_players['passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                if data_players['club_position'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] != "GK" :
                    if player_passing <= 50:
                        passing[0] += 1
                    elif 50 < player_passing <= 60:
                        passing[1] += 1
                    elif 60 < player_passing <= 70:
                        passing[2] += 1
                    elif 70 < player_passing <= 80:
                        passing[3] += 1
                    elif 80 < player_passing <= 90:
                        passing[4] += 1
                    else : 
                        passing[5] += 1

                player_mentality_pos = data_players['mentality_positioning'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                if player_mentality_pos <= 50:
                    mentality_pos[0] += 1
                elif 50 < player_mentality_pos <= 60:
                    mentality_pos[1] += 1
                elif 60 < player_mentality_pos <= 70:
                    mentality_pos[2] += 1
                elif 70 < player_mentality_pos <= 80:
                    mentality_pos[3] += 1
                elif 80 < player_mentality_pos <= 90:
                    mentality_pos[4] += 1
                else : 
                    mentality_pos[5] += 1

                player_mentality_vis = data_players['mentality_vision'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                if player_mentality_vis <= 50:
                    mentality_vis[0] += 1
                elif 50 < player_mentality_vis <= 60:
                    mentality_vis[1] += 1
                elif 60 < player_mentality_vis <= 70:
                    mentality_vis[2] += 1
                elif 70 < player_mentality_vis <= 80:
                    mentality_vis[3] += 1
                elif 80 < player_mentality_vis <= 90:
                    mentality_vis[4] += 1
                else : 
                    mentality_vis[5] += 1

                player_long_passing = data_players['skill_long_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                if player_long_passing <= 50:
                    long_passing[0] += 1
                elif 50 < player_long_passing <= 60:
                    long_passing[1] += 1
                elif 60 < player_long_passing <= 70:
                    long_passing[2] += 1
                elif 70 < player_long_passing <= 80:
                    long_passing[3] += 1
                elif 80 < player_long_passing <= 90:
                    long_passing[4] += 1
                else : 
                    long_passing[5] += 1

                player_pace = data_players['pace'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                if player_pace <= 50:
                    pace[0] += 1
                elif 50 < player_pace <= 60:
                    pace[1] += 1
                elif 60 < player_pace <= 70:
                    pace[2] += 1
                elif 70 < player_pace <= 80:
                    pace[3] += 1
                elif 80 < player_pace <= 90:
                    pace[4] += 1
                else : 
                    pace[5] += 1

                player_ball_control = data_players['skill_ball_control'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_ball_control <= 50:
                    ball_control[0] += 1
                elif 50 < player_ball_control <= 60:
                    ball_control[1] += 1
                elif 60 < player_ball_control <= 70:
                    ball_control[2] += 1
                elif 70 < player_ball_control <= 80:
                    ball_control[3] += 1
                elif 80 < player_ball_control <= 90:
                    ball_control[4] += 1
                else : 
                    ball_control[5] += 1

                player_short_passing = data_players['attacking_short_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_short_passing <= 50:
                    short_passing[0] += 1
                elif 50 < player_short_passing <= 60:
                    short_passing[1] += 1
                elif 60 < player_short_passing <= 70:
                    short_passing[2] += 1
                elif 70 < player_short_passing <= 80:
                    short_passing[3] += 1
                elif 80 < player_short_passing <= 90:
                    short_passing[4] += 1
                else : 
                    short_passing[5] += 1

                player_league = data_players['league_name'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_league == "French Ligue 1":
                    league[0] += 1
                elif player_league == "Spain Primera Division":
                    league[1] += 1
                elif player_league == "German 1. Bundesliga":
                    league[2] += 1
                elif player_league == "Italian Serie A":
                    league[3] += 1
                elif player_league == "English Premier League":
                    league[4] += 1

                player_work_rate = data_players['work_rate'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_work_rate == "High/High":
                    work_rate[0] += 1
                elif player_work_rate == "High/Medium":
                    work_rate[1] += 1
                elif player_work_rate == "High/Low":
                    work_rate[2] += 1
                elif player_work_rate == "Medium/High":
                    work_rate[3] += 1
                elif player_work_rate == "Medium/Medium":
                    work_rate[4] += 1
                elif player_work_rate == "Medium/Low":
                    work_rate[5] += 1
                elif player_work_rate == "Low/High":
                    work_rate[6] += 1
                elif player_work_rate == "Low/Medium":
                    work_rate[7] += 1
                elif player_work_rate == "Low/Low": 
                    work_rate[8] += 1

                player_physic = data_players['physic'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_physic <= 50:
                    physic[0] += 1
                elif 50 < player_physic <= 60:
                    physic[1] += 1
                elif 60 < player_physic <= 70:
                    physic[2] += 1
                elif 70 < player_physic <= 80:
                    physic[3] += 1
                elif 80 < player_physic <= 90:
                    physic[4] += 1
                else : 
                    physic[5] += 1

                player_accel = data_players['movement_acceleration'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_accel <= 50:
                    accel[0] += 1
                elif 50 < player_accel <= 60:
                    accel[1] += 1
                elif 60 < player_accel <= 70:
                    accel[2] += 1
                elif 70 < player_accel <= 80:
                    accel[3] += 1
                elif 80 < player_accel <= 90:
                    accel[4] += 1
                else : 
                    accel[5] += 1

                player_strength = data_players['power_strength'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_strength <= 50:
                    strength[0] += 1
                elif 50 < player_strength <= 60:
                    strength[1] += 1
                elif 60 < player_strength <= 70:
                    strength[2] += 1
                elif 70 < player_strength <= 80:
                    strength[3] += 1
                elif 80 < player_strength <= 90:
                    strength[4] += 1
                else : 
                    strength[5] += 1

                player_agility = data_players['movement_agility'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_agility <= 50:
                    agility[0] += 1
                elif 50 < player_agility <= 60:
                    agility[1] += 1
                elif 60 < player_agility <= 70:
                    agility[2] += 1
                elif 70 < player_agility <= 80:
                    agility[3] += 1
                elif 80 < player_agility <= 90:
                    agility[4] += 1
                else : 
                    agility[5] += 1

                player_aggression = data_players['mentality_aggression'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_aggression <= 50:
                    aggression[0] += 1
                elif 50 < player_aggression <= 60:
                    aggression[1] += 1
                elif 60 < player_aggression <= 70:
                    aggression[2] += 1
                elif 70 < player_aggression <= 80:
                    aggression[3] += 1
                elif 80 < player_aggression <= 90:
                    aggression[4] += 1
                else : 
                    aggression[5] += 1
        
            if isinstance(df_event['location'][index_suiv], list):   
                while (df_event['location'][index_suiv][0] < 60 and df_event.loc[index_suiv, 'type'] not in types_exclus and df_event['pass_outcome'][index_suiv] != "Incomplete" and df_event['pass_outcome'][index_suiv] != "Out" and df_event['pass_outcome'][index_suiv] != "Pass Offside" and df_event['pass_outcome'][index_suiv] != "Unknown" and df_event['dribble_outcome'][index_suiv] != "Incomplete"):
                    index = index_suiv
                    nb_total += 1
                    if len(trouver_joueur(df_event['player'][index], 'players_22.csv')) != 0 :
                        team = df_event['team'][index]
                        player_id = df_event['player_id'][index]
                        for i in range(len(df_lineup[team]['player_id'])):
                            if player_id == df_lineup[team]['player_id'][i] and len(df_lineup[team]['positions'][i]) != 0:
                                if df_lineup[team]['positions'][i][0]['position'] == "Left Center Back" :
                                    pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                    if pied == "Left" :
                                        def_central[0] += 1
                                    elif pied == "Right" :
                                        def_central[1] += 1
                                elif df_lineup[team]['positions'][i][0]['position'] == "Right Center Back" :
                                    pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                    if pied == "Left" :
                                        def_central[2] += 1
                                    elif pied == "Right" :
                                        def_central[3] += 1
                                elif df_lineup[team]['positions'][i][0]['position'] == "Left Back" :
                                    pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                    if pied == "Left" :
                                        back_aile[0] += 1
                                    elif pied == "Right" :
                                        back_aile[1] += 1
                                elif df_lineup[team]['positions'][i][0]['position'] == "Right Back" :
                                    pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                    if pied == "Left" :
                                        back_aile[2] += 1
                                    elif pied == "Right" :
                                        back_aile[3] += 1


                        nb_joueurs_trouv += 1
                        
                        if data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == "Left" :
                            pref_foot[0] += 1
                        else : 
                            pref_foot[1] += 1

                        if data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                            skills[0] += 1
                        elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                            skills[1] += 1
                        elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                            skills[2] += 1
                        elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                            skills[3] += 1
                        else :
                            skills[4] += 1

                        player_IMC = calcul_imc(data_players['weight_kg'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]], data_players['height_cm'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]])

                        if player_IMC <= 20 :
                            IMC[0] += 1
                        elif 20 < player_IMC <= 21 :
                            IMC[1] += 1
                        elif 21 < player_IMC <= 22 :
                            IMC[2] += 1
                        elif 22 < player_IMC <= 23 :
                            IMC[3] += 1
                        elif 23 < player_IMC <= 24 :
                            IMC[4] += 1
                        elif 24 < player_IMC <= 25 :
                            IMC[5] += 1
                        else :
                            IMC[6] += 1

                        if data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                            weak_foot[0] += 1
                        elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                            weak_foot[1] += 1
                        elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                            weak_foot[2] += 1
                        elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                            weak_foot[3] += 1
                        else :
                            weak_foot[4] += 1

                        player_passing = data_players['passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                        if data_players['club_position'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] != "GK" :
                            if player_passing <= 50:
                                passing[0] += 1
                            elif 50 < player_passing <= 60:
                                passing[1] += 1
                            elif 60 < player_passing <= 70:
                                passing[2] += 1
                            elif 70 < player_passing <= 80:
                                passing[3] += 1
                            elif 80 < player_passing <= 90:
                                passing[4] += 1
                            else : 
                                passing[5] += 1

                        player_mentality_pos = data_players['mentality_positioning'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                        if player_mentality_pos <= 50:
                            mentality_pos[0] += 1
                        elif 50 < player_mentality_pos <= 60:
                            mentality_pos[1] += 1
                        elif 60 < player_mentality_pos <= 70:
                            mentality_pos[2] += 1
                        elif 70 < player_mentality_pos <= 80:
                            mentality_pos[3] += 1
                        elif 80 < player_mentality_pos <= 90:
                            mentality_pos[4] += 1
                        else : 
                            mentality_pos[5] += 1
                        
                        player_mentality_vis = data_players['mentality_vision'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                        if player_mentality_vis <= 50:
                            mentality_vis[0] += 1
                        elif 50 < player_mentality_vis <= 60:
                            mentality_vis[1] += 1
                        elif 60 < player_mentality_vis <= 70:
                            mentality_vis[2] += 1
                        elif 70 < player_mentality_vis <= 80:
                            mentality_vis[3] += 1
                        elif 80 < player_mentality_vis <= 90:
                            mentality_vis[4] += 1
                        else : 
                            mentality_vis[5] += 1

                        player_long_passing = data_players['skill_long_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                        if player_long_passing <= 50:
                            long_passing[0] += 1
                        elif 50 < player_long_passing <= 60:
                            long_passing[1] += 1
                        elif 60 < player_long_passing <= 70:
                            long_passing[2] += 1
                        elif 70 < player_long_passing <= 80:
                            long_passing[3] += 1
                        elif 80 < player_long_passing <= 90:
                            long_passing[4] += 1
                        else : 
                            long_passing[5] += 1

                        player_pace = data_players['pace'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                        if player_pace <= 50:
                            pace[0] += 1
                        elif 50 < player_pace <= 60:
                            pace[1] += 1
                        elif 60 < player_pace <= 70:
                            pace[2] += 1
                        elif 70 < player_pace <= 80:
                            pace[3] += 1
                        elif 80 < player_pace <= 90:
                            pace[4] += 1
                        else : 
                            pace[5] += 1

                        player_ball_control = data_players['skill_ball_control'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_ball_control <= 50:
                            ball_control[0] += 1
                        elif 50 < player_ball_control <= 60:
                            ball_control[1] += 1
                        elif 60 < player_ball_control <= 70:
                            ball_control[2] += 1
                        elif 70 < player_ball_control <= 80:
                            ball_control[3] += 1
                        elif 80 < player_ball_control <= 90:
                            ball_control[4] += 1
                        else : 
                            ball_control[5] += 1

                        player_short_passing = data_players['attacking_short_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_short_passing <= 50:
                            short_passing[0] += 1
                        elif 50 < player_short_passing <= 60:
                            short_passing[1] += 1
                        elif 60 < player_short_passing <= 70:
                            short_passing[2] += 1
                        elif 70 < player_short_passing <= 80:
                            short_passing[3] += 1
                        elif 80 < player_short_passing <= 90:
                            short_passing[4] += 1
                        else : 
                            short_passing[5] += 1

                        player_league = data_players['league_name'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_league == "French Ligue 1":
                            league[0] += 1
                        elif player_league == "Spain Primera Division":
                            league[1] += 1
                        elif player_league == "German 1. Bundesliga":
                            league[2] += 1
                        elif player_league == "Italian Serie A":
                            league[3] += 1
                        elif player_league == "English Premier League":
                            league[4] += 1

                        player_work_rate = data_players['work_rate'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_work_rate == "High/High":
                            work_rate[0] += 1
                        elif player_work_rate == "High/Medium":
                            work_rate[1] += 1
                        elif player_work_rate == "High/Low":
                            work_rate[2] += 1
                        elif player_work_rate == "Medium/High":
                            work_rate[3] += 1
                        elif player_work_rate == "Medium/Medium":
                            work_rate[4] += 1
                        elif player_work_rate == "Medium/Low":
                            work_rate[5] += 1
                        elif player_work_rate == "Low/High":
                            work_rate[6] += 1
                        elif player_work_rate == "Low/Medium":
                            work_rate[7] += 1
                        elif player_work_rate == "Low/Low": 
                            work_rate[8] += 1

                        player_physic = data_players['physic'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_physic <= 50:
                            physic[0] += 1
                        elif 50 < player_physic <= 60:
                            physic[1] += 1
                        elif 60 < player_physic <= 70:
                            physic[2] += 1
                        elif 70 < player_physic <= 80:
                            physic[3] += 1
                        elif 80 < player_physic <= 90:
                            physic[4] += 1
                        else : 
                            physic[5] += 1

                        player_accel = data_players['movement_acceleration'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_accel <= 50:
                            accel[0] += 1
                        elif 50 < player_accel <= 60:
                            accel[1] += 1
                        elif 60 < player_accel <= 70:
                            accel[2] += 1
                        elif 70 < player_accel <= 80:
                            accel[3] += 1
                        elif 80 < player_accel <= 90:
                            accel[4] += 1
                        else : 
                            accel[5] += 1

                        player_strength = data_players['power_strength'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_strength <= 50:
                            strength[0] += 1
                        elif 50 < player_strength <= 60:
                            strength[1] += 1
                        elif 60 < player_strength <= 70:
                            strength[2] += 1
                        elif 70 < player_strength <= 80:
                            strength[3] += 1
                        elif 80 < player_strength <= 90:
                            strength[4] += 1
                        else : 
                            strength[5] += 1

                        player_agility = data_players['movement_agility'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_agility <= 50:
                            agility[0] += 1
                        elif 50 < player_agility <= 60:
                            agility[1] += 1
                        elif 60 < player_agility <= 70:
                            agility[2] += 1
                        elif 70 < player_agility <= 80:
                            agility[3] += 1
                        elif 80 < player_agility <= 90:
                            agility[4] += 1
                        else : 
                            agility[5] += 1

                        player_aggression = data_players['mentality_aggression'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_aggression <= 50:
                            aggression[0] += 1
                        elif 50 < player_aggression <= 60:
                            aggression[1] += 1
                        elif 60 < player_aggression <= 70:
                            aggression[2] += 1
                        elif 70 < player_aggression <= 80:
                            aggression[3] += 1
                        elif 80 < player_aggression <= 90:
                            aggression[4] += 1
                        else : 
                            aggression[5] += 1
                
                    if df_event['under_pressure'][index] == True: 
                        nb_press += 1
                    else : 
                        nb_nonPress += 1
                    if df_event['type'][index] == "Pass":
                        if df_event['pass_height'][index] == "Ground Pass":
                            array_hauteur_pass[0] += 1
                        if df_event['pass_height'][index] == "Low Pass":
                            array_hauteur_pass[1] += 1
                        if df_event['pass_height'][index] == "High Pass":
                            array_hauteur_pass[2] += 1
                        pass_length += df_event['pass_length'][index]
                        nb_pass += 1
                        duration += df_event['duration'][index]
                        liste_attente.append([df_event['location'][index][0], df_event['location'][index][1], df_event['pass_end_location'][index][0], df_event['pass_end_location'][index][1], df_event['type'][index], numero_relance])
                    if df_event['type'][index] == "Carry":
                        nb_carry += 1
                        dist_carry += distance_entre_points(df_event['location'][index][0], df_event['location'][index][1], df_event['carry_end_location'][index][0], df_event['carry_end_location'][index][1])
                        liste_attente.append([df_event['location'][index][0], df_event['location'][index][1], df_event['carry_end_location'][index][0], df_event['carry_end_location'][index][1], df_event['type'][index], numero_relance])
                        duration += df_event['duration'][index]
                    
                    time_suiv, index_suiv, periode_suiv = timestamp_suiv(index, match_id)
                    if not isinstance(df_event['location'][index_suiv], list): 
                        index_suiv = index
                        break
                    index_final = index_suiv

            elif pd.isna(df_event['location'][index_suiv]):
                
                while df_event.loc[index_suiv, 'type'] not in types_exclus:
                    index = index_suiv
                    nb_total += 1
                    if len(trouver_joueur(df_event['player'][index], 'players_22.csv')) != 0 :
                        team = df_event['team'][index]
                        player_id = df_event['player_id'][index]
                        for i in range(len(df_lineup[team]['player_id'])):
                            if player_id == df_lineup[team]['player_id'][i] and len(df_lineup[team]['positions'][i]) != 0:
                                if df_lineup[team]['positions'][i][0]['position'] == "Left Center Back" :
                                    pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                    if pied == "Left" :
                                        def_central[0] += 1
                                    elif pied == "Right" :
                                        def_central[1] += 1
                                elif df_lineup[team]['positions'][i][0]['position'] == "Right Center Back" :
                                    pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                    if pied == "Left" :
                                        def_central[2] += 1
                                    elif pied == "Right" :
                                        def_central[3] += 1
                                elif df_lineup[team]['positions'][i][0]['position'] == "Left Back" :
                                    pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                    if pied == "Left" :
                                        back_aile[0] += 1
                                    elif pied == "Right" :
                                        back_aile[1] += 1
                                elif df_lineup[team]['positions'][i][0]['position'] == "Right Back" :
                                    pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                    if pied == "Left" :
                                        back_aile[2] += 1
                                    elif pied == "Right" :
                                        back_aile[3] += 1


                        nb_joueurs_trouv += 1
                        
                        if data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == "Left" :
                            pref_foot[0] += 1
                        else : 
                            pref_foot[1] += 1

                        if data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                            skills[0] += 1
                        elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                            skills[1] += 1
                        elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                            skills[2] += 1
                        elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                            skills[3] += 1
                        else :
                            skills[4] += 1

                        player_IMC = calcul_imc(data_players['weight_kg'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]], data_players['height_cm'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]])

                        if player_IMC <= 20 :
                            IMC[0] += 1
                        elif 20 < player_IMC <= 21 :
                            IMC[1] += 1
                        elif 21 < player_IMC <= 22 :
                            IMC[2] += 1
                        elif 22 < player_IMC <= 23 :
                            IMC[3] += 1
                        elif 23 < player_IMC <= 24 :
                            IMC[4] += 1
                        elif 24 < player_IMC <= 25 :
                            IMC[5] += 1
                        else :
                            IMC[6] += 1

                        if data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                            weak_foot[0] += 1
                        elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                            weak_foot[1] += 1
                        elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                            weak_foot[2] += 1
                        elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                            weak_foot[3] += 1
                        else :
                            weak_foot[4] += 1

                        player_passing = data_players['passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                        if data_players['club_position'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] != "GK" :
                            if player_passing <= 50:
                                passing[0] += 1
                            elif 50 < player_passing <= 60:
                                passing[1] += 1
                            elif 60 < player_passing <= 70:
                                passing[2] += 1
                            elif 70 < player_passing <= 80:
                                passing[3] += 1
                            elif 80 < player_passing <= 90:
                                passing[4] += 1
                            else : 
                                passing[5] += 1

                        player_mentality_pos = data_players['mentality_positioning'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                        if player_mentality_pos <= 50:
                            mentality_pos[0] += 1
                        elif 50 < player_mentality_pos <= 60:
                            mentality_pos[1] += 1
                        elif 60 < player_mentality_pos <= 70:
                            mentality_pos[2] += 1
                        elif 70 < player_mentality_pos <= 80:
                            mentality_pos[3] += 1
                        elif 80 < player_mentality_pos <= 90:
                            mentality_pos[4] += 1
                        else : 
                            mentality_pos[5] += 1

                        player_mentality_vis = data_players['mentality_vision'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                        if player_mentality_vis <= 50:
                            mentality_vis[0] += 1
                        elif 50 < player_mentality_vis <= 60:
                            mentality_vis[1] += 1
                        elif 60 < player_mentality_vis <= 70:
                            mentality_vis[2] += 1
                        elif 70 < player_mentality_vis <= 80:
                            mentality_vis[3] += 1
                        elif 80 < player_mentality_vis <= 90:
                            mentality_vis[4] += 1
                        else : 
                            mentality_vis[5] += 1

                        player_long_passing = data_players['skill_long_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                        if player_long_passing <= 50:
                            long_passing[0] += 1
                        elif 50 < player_long_passing <= 60:
                            long_passing[1] += 1
                        elif 60 < player_long_passing <= 70:
                            long_passing[2] += 1
                        elif 70 < player_long_passing <= 80:
                            long_passing[3] += 1
                        elif 80 < player_long_passing <= 90:
                            long_passing[4] += 1
                        else : 
                            long_passing[5] += 1
                        
                        player_pace = data_players['pace'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                        if player_pace <= 50:
                            pace[0] += 1
                        elif 50 < player_pace <= 60:
                            pace[1] += 1
                        elif 60 < player_pace <= 70:
                            pace[2] += 1
                        elif 70 < player_pace <= 80:
                            pace[3] += 1
                        elif 80 < player_pace <= 90:
                            pace[4] += 1
                        else : 
                            pace[5] += 1

                        player_ball_control = data_players['skill_ball_control'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_ball_control <= 50:
                            ball_control[0] += 1
                        elif 50 < player_ball_control <= 60:
                            ball_control[1] += 1
                        elif 60 < player_ball_control <= 70:
                            ball_control[2] += 1
                        elif 70 < player_ball_control <= 80:
                            ball_control[3] += 1
                        elif 80 < player_ball_control <= 90:
                            ball_control[4] += 1
                        else : 
                            ball_control[5] += 1

                        player_short_passing = data_players['attacking_short_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_short_passing <= 50:
                            short_passing[0] += 1
                        elif 50 < player_short_passing <= 60:
                            short_passing[1] += 1
                        elif 60 < player_short_passing <= 70:
                            short_passing[2] += 1
                        elif 70 < player_short_passing <= 80:
                            short_passing[3] += 1
                        elif 80 < player_short_passing <= 90:
                            short_passing[4] += 1
                        else : 
                            short_passing[5] += 1

                        player_league = data_players['league_name'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_league == "French Ligue 1":
                            league[0] += 1
                        elif player_league == "Spain Primera Division":
                            league[1] += 1
                        elif player_league == "German 1. Bundesliga":
                            league[2] += 1
                        elif player_league == "Italian Serie A":
                            league[3] += 1
                        elif player_league == "English Premier League":
                            league[4] += 1

                        player_work_rate = data_players['work_rate'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_work_rate == "High/High":
                            work_rate[0] += 1
                        elif player_work_rate == "High/Medium":
                            work_rate[1] += 1
                        elif player_work_rate == "High/Low":
                            work_rate[2] += 1
                        elif player_work_rate == "Medium/High":
                            work_rate[3] += 1
                        elif player_work_rate == "Medium/Medium":
                            work_rate[4] += 1
                        elif player_work_rate == "Medium/Low":
                            work_rate[5] += 1
                        elif player_work_rate == "Low/High":
                            work_rate[6] += 1
                        elif player_work_rate == "Low/Medium":
                            work_rate[7] += 1
                        elif player_work_rate == "Low/Low": 
                            work_rate[8] += 1

                        player_physic = data_players['physic'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_physic <= 50:
                            physic[0] += 1
                        elif 50 < player_physic <= 60:
                            physic[1] += 1
                        elif 60 < player_physic <= 70:
                            physic[2] += 1
                        elif 70 < player_physic <= 80:
                            physic[3] += 1
                        elif 80 < player_physic <= 90:
                            physic[4] += 1
                        else : 
                            physic[5] += 1

                        player_accel = data_players['movement_acceleration'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_accel <= 50:
                            accel[0] += 1
                        elif 50 < player_accel <= 60:
                            accel[1] += 1
                        elif 60 < player_accel <= 70:
                            accel[2] += 1
                        elif 70 < player_accel <= 80:
                            accel[3] += 1
                        elif 80 < player_accel <= 90:
                            accel[4] += 1
                        else : 
                            accel[5] += 1

                        player_strength = data_players['power_strength'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_strength <= 50:
                            strength[0] += 1
                        elif 50 < player_strength <= 60:
                            strength[1] += 1
                        elif 60 < player_strength <= 70:
                            strength[2] += 1
                        elif 70 < player_strength <= 80:
                            strength[3] += 1
                        elif 80 < player_strength <= 90:
                            strength[4] += 1
                        else : 
                            strength[5] += 1

                        player_agility = data_players['movement_agility'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_agility <= 50:
                            agility[0] += 1
                        elif 50 < player_agility <= 60:
                            agility[1] += 1
                        elif 60 < player_agility <= 70:
                            agility[2] += 1
                        elif 70 < player_agility <= 80:
                            agility[3] += 1
                        elif 80 < player_agility <= 90:
                            agility[4] += 1
                        else : 
                            agility[5] += 1

                        player_aggression = data_players['mentality_aggression'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                        if player_aggression <= 50:
                            aggression[0] += 1
                        elif 50 < player_aggression <= 60:
                            aggression[1] += 1
                        elif 60 < player_aggression <= 70:
                            aggression[2] += 1
                        elif 70 < player_aggression <= 80:
                            aggression[3] += 1
                        elif 80 < player_aggression <= 90:
                            aggression[4] += 1
                        else : 
                            aggression[5] += 1

                    if df_event['under_pressure'][index] == True: 
                        nb_press += 1
                    else : 
                        nb_nonPress += 1

                    if df_event['type'][index] == "Pass":
                        if df_event['pass_height'][index] == "Ground Pass":
                            array_hauteur_pass[0] += 1
                        if df_event['pass_height'][index] == "Low Pass":
                            array_hauteur_pass[1] += 1
                        if df_event['pass_height'][index] == "High Pass":
                            array_hauteur_pass[2] += 1
                        pass_length += df_event['pass_length'][index]
                        nb_pass += 1
                        duration += df_event['duration'][index]
                        liste_attente.append([df_event['location'][index][0], df_event['location'][index][1], df_event['pass_end_location'][index][0], df_event['pass_end_location'][index][1], df_event['type'][index], numero_relance])
                    if df_event['type'][index] == "Carry":
                        nb_carry += 1
                        dist_carry += distance_entre_points(df_event['location'][index][0], df_event['location'][index][1], df_event['carry_end_location'][index][0], df_event['carry_end_location'][index][1])
                        liste_attente.append([df_event['location'][index][0], df_event['location'][index][1], df_event['carry_end_location'][index][0], df_event['carry_end_location'][index][1], df_event['pass_outcome'][index], numero_relance])
                        duration += df_event['duration'][index]
                    
                    time_suiv, index_suiv, periode_suiv = timestamp_suiv(index, match_id)
                    index_final = index_suiv
                
            if index_final == 0 :
                nb_total += 1
                if len(trouver_joueur(df_event['player'][index], 'players_22.csv')) != 0 :
                    
                    team = df_event['team'][index]
                    player_id = df_event['player_id'][index]
                    for i in range(len(df_lineup[team]['player_id'])):
                        if player_id == df_lineup[team]['player_id'][i] and len(df_lineup[team]['positions'][i]) != 0:
                            if df_lineup[team]['positions'][i][0]['position'] == "Left Center Back" :
                                pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                if pied == "Left" :
                                    def_central[0] += 1
                                elif pied == "Right" :
                                    def_central[1] += 1
                            elif df_lineup[team]['positions'][i][0]['position'] == "Right Center Back" :
                                pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                if pied == "Left" :
                                    def_central[2] += 1
                                elif pied == "Right" :
                                    def_central[3] += 1
                            elif df_lineup[team]['positions'][i][0]['position'] == "Left Back" :
                                pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                if pied == "Left" :
                                    back_aile[0] += 1
                                elif pied == "Right" :
                                    back_aile[1] += 1
                            elif df_lineup[team]['positions'][i][0]['position'] == "Right Back" :
                                pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                if pied == "Left" :
                                    back_aile[2] += 1
                                elif pied == "Right" :
                                    back_aile[3] += 1


                    nb_joueurs_trouv += 1
                
                    if data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == "Left" :
                        pref_foot[0] += 1
                    else : 
                        pref_foot[1] += 1

                    if data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                        skills[0] += 1
                    elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                        skills[1] += 1
                    elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                        skills[2] += 1
                    elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                        skills[3] += 1
                    else :
                        skills[4] += 1

                    player_IMC = calcul_imc(data_players['weight_kg'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]], data_players['height_cm'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]])

                    if player_IMC <= 20 :
                        IMC[0] += 1
                    elif 20 < player_IMC <= 21 :
                        IMC[1] += 1
                    elif 21 < player_IMC <= 22 :
                        IMC[2] += 1
                    elif 22 < player_IMC <= 23 :
                        IMC[3] += 1
                    elif 23 < player_IMC <= 24 :
                        IMC[4] += 1
                    elif 24 < player_IMC <= 25 :
                        IMC[5] += 1
                    else :
                        IMC[6] += 1

                    if data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                        weak_foot[0] += 1
                    elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                        weak_foot[1] += 1
                    elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                        weak_foot[2] += 1
                    elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                        weak_foot[3] += 1
                    else :
                        weak_foot[4] += 1

                    player_passing = data_players['passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if data_players['club_position'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] != "GK" :
                        if player_passing <= 50:
                            passing[0] += 1
                        elif 50 < player_passing <= 60:
                            passing[1] += 1
                        elif 60 < player_passing <= 70:
                            passing[2] += 1
                        elif 70 < player_passing <= 80:
                            passing[3] += 1
                        elif 80 < player_passing <= 90:
                            passing[4] += 1
                        else : 
                            passing[5] += 1

                    player_mentality_pos = data_players['mentality_positioning'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if player_mentality_pos <= 50:
                        mentality_pos[0] += 1
                    elif 50 < player_mentality_pos <= 60:
                        mentality_pos[1] += 1
                    elif 60 < player_mentality_pos <= 70:
                        mentality_pos[2] += 1
                    elif 70 < player_mentality_pos <= 80:
                        mentality_pos[3] += 1
                    elif 80 < player_mentality_pos <= 90:
                        mentality_pos[4] += 1
                    else : 
                        mentality_pos[5] += 1

                    player_mentality_vis = data_players['mentality_vision'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if player_mentality_vis <= 50:
                        mentality_vis[0] += 1
                    elif 50 < player_mentality_vis <= 60:
                        mentality_vis[1] += 1
                    elif 60 < player_mentality_vis <= 70:
                        mentality_vis[2] += 1
                    elif 70 < player_mentality_vis <= 80:
                        mentality_vis[3] += 1
                    elif 80 < player_mentality_vis <= 90:
                        mentality_vis[4] += 1
                    else : 
                        mentality_vis[5] += 1

                    player_long_passing = data_players['skill_long_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if player_long_passing <= 50:
                        long_passing[0] += 1
                    elif 50 < player_long_passing <= 60:
                        long_passing[1] += 1
                    elif 60 < player_long_passing <= 70:
                        long_passing[2] += 1
                    elif 70 < player_long_passing <= 80:
                        long_passing[3] += 1
                    elif 80 < player_long_passing <= 90:
                        long_passing[4] += 1
                    else : 
                        long_passing[5] += 1
                    
                    player_pace = data_players['pace'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if player_pace <= 50:
                        pace[0] += 1
                    elif 50 < player_pace <= 60:
                        pace[1] += 1
                    elif 60 < player_pace <= 70:
                        pace[2] += 1
                    elif 70 < player_pace <= 80:
                        pace[3] += 1
                    elif 80 < player_pace <= 90:
                        pace[4] += 1
                    else : 
                        pace[5] += 1

                    player_ball_control = data_players['skill_ball_control'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_ball_control <= 50:
                        ball_control[0] += 1
                    elif 50 < player_ball_control <= 60:
                        ball_control[1] += 1
                    elif 60 < player_ball_control <= 70:
                        ball_control[2] += 1
                    elif 70 < player_ball_control <= 80:
                        ball_control[3] += 1
                    elif 80 < player_ball_control <= 90:
                        ball_control[4] += 1
                    else : 
                        ball_control[5] += 1

                    player_short_passing = data_players['attacking_short_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_short_passing <= 50:
                        short_passing[0] += 1
                    elif 50 < player_short_passing <= 60:
                        short_passing[1] += 1
                    elif 60 < player_short_passing <= 70:
                        short_passing[2] += 1
                    elif 70 < player_short_passing <= 80:
                        short_passing[3] += 1
                    elif 80 < player_short_passing <= 90:
                        short_passing[4] += 1
                    else : 
                        short_passing[5] += 1

                    player_league = data_players['league_name'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_league == "French Ligue 1":
                        league[0] += 1
                    elif player_league == "Spain Primera Division":
                        league[1] += 1
                    elif player_league == "German 1. Bundesliga":
                        league[2] += 1
                    elif player_league == "Italian Serie A":
                        league[3] += 1
                    elif player_league == "English Premier League":
                        league[4] += 1
                    
                    player_work_rate = data_players['work_rate'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_work_rate == "High/High":
                        work_rate[0] += 1
                    elif player_work_rate == "High/Medium":
                        work_rate[1] += 1
                    elif player_work_rate == "High/Low":
                        work_rate[2] += 1
                    elif player_work_rate == "Medium/High":
                        work_rate[3] += 1
                    elif player_work_rate == "Medium/Medium":
                        work_rate[4] += 1
                    elif player_work_rate == "Medium/Low":
                        work_rate[5] += 1
                    elif player_work_rate == "Low/High":
                        work_rate[6] += 1
                    elif player_work_rate == "Low/Medium":
                        work_rate[7] += 1
                    elif player_work_rate == "Low/Low": 
                        work_rate[8] += 1

                    player_physic = data_players['physic'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_physic <= 50:
                        physic[0] += 1
                    elif 50 < player_physic <= 60:
                        physic[1] += 1
                    elif 60 < player_physic <= 70:
                        physic[2] += 1
                    elif 70 < player_physic <= 80:
                        physic[3] += 1
                    elif 80 < player_physic <= 90:
                        physic[4] += 1
                    else : 
                        physic[5] += 1

                    player_accel = data_players['movement_acceleration'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_accel <= 50:
                        accel[0] += 1
                    elif 50 < player_accel <= 60:
                        accel[1] += 1
                    elif 60 < player_accel <= 70:
                        accel[2] += 1
                    elif 70 < player_accel <= 80:
                        accel[3] += 1
                    elif 80 < player_accel <= 90:
                        accel[4] += 1
                    else : 
                        accel[5] += 1

                    player_strength = data_players['power_strength'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_strength <= 50:
                        strength[0] += 1
                    elif 50 < player_strength <= 60:
                        strength[1] += 1
                    elif 60 < player_strength <= 70:
                        strength[2] += 1
                    elif 70 < player_strength <= 80:
                        strength[3] += 1
                    elif 80 < player_strength <= 90:
                        strength[4] += 1
                    else : 
                        strength[5] += 1

                    player_agility = data_players['movement_agility'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_agility <= 50:
                        agility[0] += 1
                    elif 50 < player_agility <= 60:
                        agility[1] += 1
                    elif 60 < player_agility <= 70:
                        agility[2] += 1
                    elif 70 < player_agility <= 80:
                        agility[3] += 1
                    elif 80 < player_agility <= 90:
                        agility[4] += 1
                    else : 
                        agility[5] += 1

                    player_aggression = data_players['mentality_aggression'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_aggression <= 50:
                        aggression[0] += 1
                    elif 50 < player_aggression <= 60:
                        aggression[1] += 1
                    elif 60 < player_aggression <= 70:
                        aggression[2] += 1
                    elif 70 < player_aggression <= 80:
                        aggression[3] += 1
                    elif 80 < player_aggression <= 90:
                        aggression[4] += 1
                    else : 
                        aggression[5] += 1
                
                if df_event['under_pressure'][index] == True: 
                    nb_press += 1
                else : 
                    nb_nonPress += 1
                if df_event['type'][index] == "Pass":
                    if df_event['pass_height'][index] == "Ground Pass":
                        array_hauteur_pass[0] += 1
                    if df_event['pass_height'][index] == "Low Pass":
                        array_hauteur_pass[1] += 1
                    if df_event['pass_height'][index] == "High Pass":
                        array_hauteur_pass[2] += 1
                    pass_length += df_event['pass_length'][index]
                    nb_pass += 1
                    duration += df_event['duration'][index]
                    liste_attente.append([df_event['location'][index][0], df_event['location'][index][1], df_event['pass_end_location'][index][0], df_event['pass_end_location'][index][1], df_event['type'][index], numero_relance])
                if df_event['type'][index] == "Carry":
                    nb_carry += 1
                    dist_carry += distance_entre_points(df_event['location'][index][0], df_event['location'][index][1], df_event['carry_end_location'][index][0], df_event['carry_end_location'][index][1])   
                    duration += df_event['duration'][index]
                    liste_attente.append([df_event['location'][index][0], df_event['location'][index][1], df_event['carry_end_location'][index][0], df_event['carry_end_location'][index][1], df_event['type'][index], numero_relance])
                if df_event['type'][index] == "Pass":
                    if df_event['pass_end_location'][index][0] > 60 :
                        for i in range(len(liste_attente)):
                            liste_reussi.append(liste_attente[i])

                        packing = packing_one_event(df_event['id'][index], index, match_id)
                        if packing == 0 or packing == 1 :
                            moyenne_duration[1] += duration
                            compteur_relance[1] += 1
                            moyenne_type_passes[1][0] += array_hauteur_pass[0]
                            moyenne_type_passes[1][1] += array_hauteur_pass[1]
                            moyenne_type_passes[1][2] += array_hauteur_pass[2]
                            moyenne_nb_passes[1] += nb_pass
                            moyenne_long_par_la_passe[1] += pass_length
                            moyenne_nb_carry[1] += nb_carry
                            moyenne_long_par_le_carry[1] += dist_carry
                            moyenne_pied_fort[1][0] += pref_foot[0]
                            moyenne_pied_fort[1][1] += pref_foot[1]
                            for j in range(len(weak_foot)):
                                moyenne_pied_faible[1][j] += weak_foot[j]
                            for j in range(len(skills)):
                                moyenne_skills[1][j] += skills[j]
                            for j in range(len(IMC)):
                                moyenne_IMC[1][j] += IMC[j]
                            for j in range(len(passing)):
                                moyenne_passing[1][j] += passing[j]
                            for j in range(len(mentality_pos)):
                                moyenne_mentality_pos[1][j] += mentality_pos[j]
                            for j in range(len(mentality_vis)):
                                moyenne_mentality_vis[1][j] += mentality_vis[j]
                            for j in range(len(long_passing)):
                                moyenne_long_passing[1][j] += long_passing[j]
                            for j in range(len(pace)):
                                moyenne_pace[1][j] += pace[j]
                            for j in range(len(ball_control)):
                                moyenne_ball_control[1][j] += ball_control[j]
                            for j in range(len(short_passing)):
                                moyenne_short_passing[1][j] += short_passing[j]
                            for j in range(len(league)):
                                moyenne_league[1][j] += league[j]
                            for j in range(len(work_rate)):
                                moyenne_work_rate[1][j] += work_rate[j]
                            for j in range(len(physic)):
                                moyenne_physic[1][j] += physic[j]
                            for j in range(len(accel)):
                                moyenne_accel[1][j] += accel[j]
                            for j in range(len(strength)):
                                moyenne_strength[1][j] += strength[j]
                            for j in range(len(agility)):
                                moyenne_agility[1][j] += agility[j]
                            for j in range(len(aggression)):
                                moyenne_aggression[1][j] += aggression[j]
                            for j in range(len(def_central)):
                                center_back[1][j] += def_central[j]
                            for j in range(len(back_aile)):
                                back[1][j] += back_aile[j]
                            
                            if df_event['team'][index] == team[0]:
                                dico_compo[compo[0]][1] += 1
                            else : 
                                dico_compo[compo[1]][1] += 1
                        if packing == 2 or packing == 3 :
                            moyenne_duration[2] += duration
                            compteur_relance[2] += 1
                            moyenne_type_passes[2][0] += array_hauteur_pass[0]
                            moyenne_type_passes[2][1] += array_hauteur_pass[1]
                            moyenne_type_passes[2][2] += array_hauteur_pass[2]
                            moyenne_nb_passes[2] += nb_pass
                            moyenne_long_par_la_passe[2] += pass_length
                            moyenne_nb_carry[2] += nb_carry
                            moyenne_long_par_le_carry[2] += dist_carry
                            moyenne_pied_fort[2][0] += pref_foot[0]
                            moyenne_pied_fort[2][1] += pref_foot[1]
                            for j in range(len(weak_foot)):
                                moyenne_pied_faible[2][j] += weak_foot[j]
                            for j in range(len(skills)):
                                moyenne_skills[2][j] += skills[j]
                            for j in range(len(IMC)):
                                moyenne_IMC[2][j] += IMC[j]
                            for j in range(len(passing)):
                                moyenne_passing[2][j] += passing[j]
                            for j in range(len(mentality_pos)):
                                moyenne_mentality_pos[2][j] += mentality_pos[j]
                            for j in range(len(mentality_vis)):
                                moyenne_mentality_vis[2][j] += mentality_vis[j]
                            for j in range(len(long_passing)):
                                moyenne_long_passing[2][j] += long_passing[j]
                            for j in range(len(pace)):
                                moyenne_pace[2][j] += pace[j]
                            for j in range(len(ball_control)):
                                moyenne_ball_control[2][j] += ball_control[j]
                            for j in range(len(short_passing)):
                                moyenne_short_passing[2][j] += short_passing[j]
                            for j in range(len(league)):
                                moyenne_league[2][j] += league[j]
                            for j in range(len(work_rate)):
                                moyenne_work_rate[2][j] += work_rate[j]
                            for j in range(len(physic)):
                                moyenne_physic[2][j] += physic[j]
                            for j in range(len(accel)):
                                moyenne_accel[2][j] += accel[j]
                            for j in range(len(strength)):
                                moyenne_strength[2][j] += strength[j]
                            for j in range(len(agility)):
                                moyenne_agility[2][j] += agility[j]
                            for j in range(len(aggression)):
                                moyenne_aggression[2][j] += aggression[j]
                            for j in range(len(def_central)):
                                center_back[2][j] += def_central[j]
                            for j in range(len(back_aile)):
                                back[2][j] += back_aile[j]
                            
                            if df_event['team'][index] == team[0]:
                                dico_compo[compo[0]][2] += 1
                            else : 
                                dico_compo[compo[1]][2] += 1
                        if packing == 4 or packing == 5 :
                            moyenne_duration[3] += duration
                            compteur_relance[3] += 1
                            moyenne_type_passes[3][0] += array_hauteur_pass[0]
                            moyenne_type_passes[3][1] += array_hauteur_pass[1]
                            moyenne_type_passes[3][2] += array_hauteur_pass[2]
                            moyenne_nb_passes[3] += nb_pass
                            moyenne_long_par_la_passe[3] += pass_length
                            moyenne_nb_carry[3] += nb_carry
                            moyenne_long_par_le_carry[3] += dist_carry
                            moyenne_pied_fort[3][0] += pref_foot[0]
                            moyenne_pied_fort[3][1] += pref_foot[1]
                            for j in range(len(weak_foot)):
                                moyenne_pied_faible[3][j] += weak_foot[j]
                            for j in range(len(skills)):
                                moyenne_skills[3][j] += skills[j]
                            for j in range(len(IMC)):
                                moyenne_IMC[3][j] += IMC[j]
                            for j in range(len(passing)):
                                moyenne_passing[3][j] += passing[j]
                            for j in range(len(mentality_pos)):
                                moyenne_mentality_pos[3][j] += mentality_pos[j]
                            for j in range(len(mentality_vis)):
                                moyenne_mentality_vis[3][j] += mentality_vis[j]
                            for j in range(len(long_passing)):
                                moyenne_long_passing[3][j] += long_passing[j]
                            for j in range(len(pace)):
                                moyenne_pace[3][j] += pace[j]
                            for j in range(len(ball_control)):
                                moyenne_ball_control[3][j] += ball_control[j]
                            for j in range(len(short_passing)):
                                moyenne_short_passing[3][j] += short_passing[j]
                            for j in range(len(league)):
                                moyenne_league[3][j] += league[j]
                            for j in range(len(work_rate)):
                                moyenne_work_rate[3][j] += work_rate[j]
                            for j in range(len(physic)):
                                moyenne_physic[3][j] += physic[j]
                            for j in range(len(accel)):
                                moyenne_accel[3][j] += accel[j]
                            for j in range(len(strength)):
                                moyenne_strength[3][j] += strength[j]
                            for j in range(len(agility)):
                                moyenne_agility[3][j] += agility[j]
                            for j in range(len(aggression)):
                                moyenne_aggression[3][j] += aggression[j]
                            for j in range(len(def_central)):
                                center_back[3][j] += def_central[j]
                            for j in range(len(back_aile)):
                                back[3][j] += back_aile[j]
                            
                            if df_event['team'][index] == team[0]:
                                dico_compo[compo[0]][3] += 1
                            else : 
                                dico_compo[compo[1]][3] += 1
                        if packing == 6 or packing == 7 :
                            moyenne_duration[4] += duration
                            compteur_relance[4] += 1
                            moyenne_type_passes[4][0] += array_hauteur_pass[0]
                            moyenne_type_passes[4][1] += array_hauteur_pass[1]
                            moyenne_type_passes[4][2] += array_hauteur_pass[2]
                            moyenne_nb_passes[4] += nb_pass
                            moyenne_long_par_la_passe[4] += pass_length
                            moyenne_nb_carry[4] += nb_carry
                            moyenne_long_par_le_carry[4] += dist_carry
                            moyenne_pied_fort[4][0] += pref_foot[0]
                            moyenne_pied_fort[4][1] += pref_foot[1]
                            for j in range(len(weak_foot)):
                                moyenne_pied_faible[4][j] += weak_foot[j]
                            for j in range(len(skills)):
                                moyenne_skills[4][j] += skills[j]
                            for j in range(len(IMC)):
                                moyenne_IMC[4][j] += IMC[j]
                            for j in range(len(passing)):
                                moyenne_passing[4][j] += passing[j]
                            for j in range(len(mentality_pos)):
                                moyenne_mentality_pos[4][j] += mentality_pos[j]
                            for j in range(len(mentality_vis)):
                                moyenne_mentality_vis[4][j] += mentality_vis[j]
                            for j in range(len(long_passing)):
                                moyenne_long_passing[4][j] += long_passing[j]
                            for j in range(len(pace)):
                                moyenne_pace[4][j] += pace[j]
                            for j in range(len(ball_control)):
                                moyenne_ball_control[4][j] += ball_control[j]
                            for j in range(len(short_passing)):
                                moyenne_short_passing[4][j] += short_passing[j]
                            for j in range(len(league)):
                                moyenne_league[4][j] += league[j]
                            for j in range(len(work_rate)):
                                moyenne_work_rate[4][j] += work_rate[j]
                            for j in range(len(physic)):
                                moyenne_physic[4][j] += physic[j]
                            for j in range(len(accel)):
                                moyenne_accel[4][j] += accel[j]
                            for j in range(len(strength)):
                                moyenne_strength[4][j] += strength[j]
                            for j in range(len(agility)):
                                moyenne_agility[4][j] += agility[j]
                            for j in range(len(aggression)):
                                moyenne_aggression[4][j] += aggression[j]
                            for j in range(len(def_central)):
                                center_back[4][j] += def_central[j]
                            for j in range(len(back_aile)):
                                back[4][j] += back_aile[j]
                            
                            if df_event['team'][index] == team[0]:
                                dico_compo[compo[0]][4] += 1
                            else : 
                                dico_compo[compo[1]][4] += 1
                        if packing == 8 or packing == 9 :
                            moyenne_duration[5] += duration
                            compteur_relance[5] += 1
                            moyenne_type_passes[5][0] += array_hauteur_pass[0]
                            moyenne_type_passes[5][1] += array_hauteur_pass[1]
                            moyenne_type_passes[5][2] += array_hauteur_pass[2]
                            moyenne_nb_passes[5] += nb_pass
                            moyenne_long_par_la_passe[5] += pass_length
                            moyenne_nb_carry[5] += nb_carry
                            moyenne_long_par_le_carry[5] += dist_carry
                            moyenne_pied_fort[5][0] += pref_foot[0]
                            moyenne_pied_fort[5][1] += pref_foot[1]
                            for j in range(len(weak_foot)):
                                moyenne_pied_faible[5][j] += weak_foot[j]
                            for j in range(len(skills)):
                                moyenne_skills[5][j] += skills[j]
                            for j in range(len(IMC)):
                                moyenne_IMC[5][j] += IMC[j]
                            for j in range(len(passing)):
                                moyenne_passing[5][j] += passing[j]
                            for j in range(len(mentality_pos)):
                                moyenne_mentality_pos[5][j] += mentality_pos[j]
                            for j in range(len(mentality_vis)):
                                moyenne_mentality_vis[5][j] += mentality_vis[j]
                            for j in range(len(long_passing)):
                                moyenne_long_passing[5][j] += long_passing[j]
                            for j in range(len(pace)):
                                moyenne_pace[5][j] += pace[j]
                            for j in range(len(ball_control)):
                                moyenne_ball_control[5][j] += ball_control[j]
                            for j in range(len(short_passing)):
                                moyenne_short_passing[5][j] += short_passing[j]
                            for j in range(len(league)):
                                moyenne_league[5][j] += league[j]
                            for j in range(len(work_rate)):
                                moyenne_work_rate[5][j] += work_rate[j]
                            for j in range(len(physic)):
                                moyenne_physic[5][j] += physic[j]
                            for j in range(len(accel)):
                                moyenne_accel[5][j] += accel[j]
                            for j in range(len(strength)):
                                moyenne_strength[5][j] += strength[j]
                            for j in range(len(agility)):
                                moyenne_agility[5][j] += agility[j]
                            for j in range(len(aggression)):
                                moyenne_aggression[5][j] += aggression[j]
                            for j in range(len(def_central)):
                                center_back[5][j] += def_central[j]
                            for j in range(len(back_aile)):
                                back[5][j] += back_aile[j]
                            
                            if df_event['team'][index] == team[0]:
                                dico_compo[compo[0]][5] += 1
                            else : 
                                dico_compo[compo[1]][5] += 1
                        if packing == 10 :
                            moyenne_duration[6] += duration
                            compteur_relance[6] += 1
                            moyenne_type_passes[6][0] += array_hauteur_pass[0]
                            moyenne_type_passes[6][1] += array_hauteur_pass[1]
                            moyenne_type_passes[6][2] += array_hauteur_pass[2]
                            moyenne_nb_passes[6] += nb_pass
                            moyenne_long_par_la_passe[6] += pass_length
                            moyenne_nb_carry[6] += nb_carry
                            moyenne_long_par_le_carry[6] += dist_carry
                            moyenne_pied_fort[6][0] += pref_foot[0]
                            moyenne_pied_fort[6][1] += pref_foot[1]
                            for j in range(len(weak_foot)):
                                moyenne_pied_faible[6][j] += weak_foot[j]
                            for j in range(len(skills)):
                                moyenne_skills[6][j] += skills[j]
                            for j in range(len(IMC)):
                                moyenne_IMC[6][j] += IMC[j]
                            for j in range(len(passing)):
                                moyenne_passing[6][j] += passing[j]
                            for j in range(len(mentality_pos)):
                                moyenne_mentality_pos[6][j] += mentality_pos[j]
                            for j in range(len(mentality_vis)):
                                moyenne_mentality_vis[6][j] += mentality_vis[j]
                            for j in range(len(long_passing)):
                                moyenne_long_passing[6][j] += long_passing[j]
                            for j in range(len(pace)):
                                moyenne_pace[6][j] += pace[j]
                            for j in range(len(ball_control)):
                                moyenne_ball_control[6][j] += ball_control[j]
                            for j in range(len(short_passing)):
                                moyenne_short_passing[6][j] += short_passing[j]
                            for j in range(len(league)):
                                moyenne_league[6][j] += league[j]
                            for j in range(len(work_rate)):
                                moyenne_work_rate[6][j] += work_rate[j]
                            for j in range(len(physic)):
                                moyenne_physic[6][j] += physic[j]
                            for j in range(len(accel)):
                                moyenne_accel[6][j] += accel[j]
                            for j in range(len(strength)):
                                moyenne_strength[6][j] += strength[j]
                            for j in range(len(agility)):
                                moyenne_agility[6][j] += agility[j]
                            for j in range(len(aggression)):
                                moyenne_aggression[6][j] += aggression[j]
                            for j in range(len(def_central)):
                                center_back[6][j] += def_central[j]
                            for j in range(len(back_aile)):
                                back[6][j] += back_aile[j]
                            
                            if df_event['team'][index] == team[0]:
                                dico_compo[compo[0]][6] += 1
                            else : 
                                dico_compo[compo[1]][6] += 1

                else :
                    if df_event['type'][index_suiv] == "Pass" :
                        if df_event['pass_height'][index_suiv] == "Ground Pass":
                            array_hauteur_pass[0] += 1
                        if df_event['pass_height'][index_suiv] == "Low Pass":
                            array_hauteur_pass[1] += 1
                        if df_event['pass_height'][index_suiv] == "High Pass":
                            array_hauteur_pass[2] += 1
                        pass_length += df_event['pass_length'][index_suiv]
                        nb_pass += 1
                        duration += df_event['duration'][index_suiv]
                        liste_attente.append([df_event['location'][index_suiv][0], df_event['location'][index_suiv][1], df_event['pass_end_location'][index_suiv][0], df_event['pass_end_location'][index_suiv][1], df_event['type'][index_suiv], numero_relance])
                    if df_event['type'][index_suiv] == "Carry" :
                        nb_carry += 1
                        dist_carry += distance_entre_points(df_event['location'][index_suiv][0], df_event['location'][index_suiv][1], df_event['carry_end_location'][index_suiv][0], df_event['carry_end_location'][index_suiv][1])
                        duration += df_event['duration'][index_suiv]
                        liste_attente.append([df_event['location'][index_suiv][0], df_event['location'][index_suiv][1], df_event['carry_end_location'][index_suiv][0], df_event['carry_end_location'][index_suiv][1], df_event['type'][index_suiv], numero_relance])
                    for i in range(len(liste_attente)):
                        liste_echec.append(liste_attente[i])
        
                    compteur_relance[0] += 1
                    moyenne_duration[0] += duration 
                    moyenne_type_passes[0][0] += array_hauteur_pass[0]
                    moyenne_type_passes[0][1] += array_hauteur_pass[1]
                    moyenne_type_passes[0][2] += array_hauteur_pass[2]
                    moyenne_nb_passes[0] += nb_pass
                    moyenne_long_par_la_passe[0] += pass_length
                    moyenne_nb_carry[0] += nb_carry
                    moyenne_long_par_le_carry[0] += dist_carry
                    moyenne_pied_fort[0][0] += pref_foot[0]
                    moyenne_pied_fort[0][1] += pref_foot[1]
                    for j in range(len(weak_foot)):
                        moyenne_pied_faible[0][j] += weak_foot[j]
                    for j in range(len(skills)):
                        moyenne_skills[0][j] += skills[j]
                    for j in range(len(IMC)):
                        moyenne_IMC[0][j] += IMC[j]
                    for j in range(len(passing)):
                        moyenne_passing[0][j] += passing[j]
                    for j in range(len(mentality_pos)):
                        moyenne_mentality_pos[0][j] += mentality_pos[j]
                    for j in range(len(mentality_vis)):
                        moyenne_mentality_vis[0][j] += mentality_vis[j]
                    for j in range(len(long_passing)):
                        moyenne_long_passing[0][j] += long_passing[j]
                    for j in range(len(pace)):
                        moyenne_pace[0][j] += pace[j]
                    for j in range(len(ball_control)):
                        moyenne_ball_control[0][j] += ball_control[j]
                    for j in range(len(short_passing)):
                        moyenne_short_passing[0][j] += short_passing[j]
                    for j in range(len(league)):
                        moyenne_league[0][j] += league[j]
                    for j in range(len(work_rate)):
                        moyenne_work_rate[0][j] += work_rate[j]
                    for j in range(len(physic)):
                        moyenne_physic[0][j] += physic[j]
                    for j in range(len(accel)):
                        moyenne_accel[0][j] += accel[j]
                    for j in range(len(strength)):
                        moyenne_strength[0][j] += strength[j]
                    for j in range(len(agility)):
                        moyenne_agility[0][j] += agility[j]
                    for j in range(len(aggression)):
                        moyenne_aggression[0][j] += aggression[j]
                    for j in range(len(def_central)):
                        center_back[0][j] += def_central[j]
                    for j in range(len(back_aile)):
                        back[0][j] += back_aile[j]
                    
                    if df_event['team'][index] == team[0]:
                        dico_compo[compo[0]][0] += 1
                    else : 
                        dico_compo[compo[1]][0] += 1
                    
                numero_relance += 1
            if index_final != 0 :
                end_time = datetime.strptime(df_event['timestamp'][index_suiv], "%H:%M:%S.%f")
                
                time_difference = end_time - start_time
                time_difference_sec = time_difference.total_seconds()
                duration = time_difference_sec
                if df_event['location'][index_suiv][0] > 60 and df_event['type'][index_suiv] != "Foul Committed":
                    for i in range(len(liste_attente)):
                        liste_reussi.append(liste_attente[i])

                    packing = packing_relance(df_event['id'][first_index], df_event['id'][index_final], first_index, index_final, match_id)
                    if packing == 0 or packing == 1 :
                        moyenne_duration[1] += duration
                        compteur_relance[1] += 1
                        moyenne_type_passes[1][0] += array_hauteur_pass[0]
                        moyenne_type_passes[1][1] += array_hauteur_pass[1]
                        moyenne_type_passes[1][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[1] += nb_pass
                        moyenne_long_par_la_passe[1] += pass_length
                        moyenne_nb_carry[1] += nb_carry
                        moyenne_long_par_le_carry[1] += dist_carry
                        moyenne_pied_fort[1][0] += pref_foot[0]
                        moyenne_pied_fort[1][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[1][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[1][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[1][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[1][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[1][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[1][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[1][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[1][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[1][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[1][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[1][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[1][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[1][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[1][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[1][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[1][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[1][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[1][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[1][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][1] += 1
                        else : 
                            dico_compo[compo[1]][1] += 1
                    if packing == 2 or packing == 3 :
                        moyenne_duration[2] += duration
                        compteur_relance[2] += 1
                        moyenne_type_passes[2][0] += array_hauteur_pass[0]
                        moyenne_type_passes[2][1] += array_hauteur_pass[1]
                        moyenne_type_passes[2][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[2] += nb_pass
                        moyenne_long_par_la_passe[2] += pass_length
                        moyenne_nb_carry[2] += nb_carry
                        moyenne_long_par_le_carry[2] += dist_carry
                        moyenne_pied_fort[2][0] += pref_foot[0]
                        moyenne_pied_fort[2][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[2][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[2][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[2][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[2][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[2][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[2][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[2][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[2][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[2][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[2][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[2][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[2][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[2][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[2][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[2][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[2][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[2][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[2][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[2][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][2] += 1
                        else : 
                            dico_compo[compo[1]][2] += 1
                    if packing == 4 or packing == 5 :
                        moyenne_duration[3] += duration
                        compteur_relance[3] += 1
                        moyenne_type_passes[3][0] += array_hauteur_pass[0]
                        moyenne_type_passes[3][1] += array_hauteur_pass[1]
                        moyenne_type_passes[3][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[3] += nb_pass
                        moyenne_long_par_la_passe[3] += pass_length
                        moyenne_nb_carry[3] += nb_carry
                        moyenne_long_par_le_carry[3] += dist_carry
                        moyenne_pied_fort[3][0] += pref_foot[0]
                        moyenne_pied_fort[3][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[3][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[3][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[3][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[3][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[3][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[3][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[3][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[3][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[3][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[3][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[3][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[3][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[3][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[3][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[3][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[3][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[3][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[3][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[3][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][3] += 1
                        else : 
                            dico_compo[compo[1]][3] += 1
                    if packing == 6 or packing == 7 :
                        moyenne_duration[4] += duration
                        compteur_relance[4] += 1
                        moyenne_type_passes[4][0] += array_hauteur_pass[0]
                        moyenne_type_passes[4][1] += array_hauteur_pass[1]
                        moyenne_type_passes[4][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[4] += nb_pass
                        moyenne_long_par_la_passe[4] += pass_length
                        moyenne_nb_carry[4] += nb_carry
                        moyenne_long_par_le_carry[4] += dist_carry
                        moyenne_pied_fort[4][0] += pref_foot[0]
                        moyenne_pied_fort[4][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[4][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[4][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[4][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[4][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[4][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[4][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[4][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[4][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[4][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[4][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[4][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[4][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[4][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[4][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[4][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[4][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[4][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[4][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[4][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][4] += 1
                        else : 
                            dico_compo[compo[1]][4] += 1
                    if packing == 8 or packing == 9 :
                        moyenne_duration[5] += duration
                        compteur_relance[5] += 1
                        moyenne_type_passes[5][0] += array_hauteur_pass[0]
                        moyenne_type_passes[5][1] += array_hauteur_pass[1]
                        moyenne_type_passes[5][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[5] += nb_pass
                        moyenne_long_par_la_passe[5] += pass_length
                        moyenne_nb_carry[5] += nb_carry
                        moyenne_long_par_le_carry[5] += dist_carry
                        moyenne_pied_fort[5][0] += pref_foot[0]
                        moyenne_pied_fort[5][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[5][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[5][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[5][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[5][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[5][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[5][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[5][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[5][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[5][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[5][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[5][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[5][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[5][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[5][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[5][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[5][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[5][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[5][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[5][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][5] += 1
                        else : 
                            dico_compo[compo[1]][5] += 1
                    if packing == 10 :
                        moyenne_duration[6] += duration
                        compteur_relance[6] += 1
                        moyenne_type_passes[6][0] += array_hauteur_pass[0]
                        moyenne_type_passes[6][1] += array_hauteur_pass[1]
                        moyenne_type_passes[6][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[6] += nb_pass
                        moyenne_long_par_la_passe[6] += pass_length
                        moyenne_nb_carry[6] += nb_carry
                        moyenne_long_par_le_carry[6] += dist_carry
                        moyenne_pied_fort[6][0] += pref_foot[0]
                        moyenne_pied_fort[6][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[6][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[6][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[6][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[6][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[6][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[6][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[6][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[6][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[6][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[6][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[6][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[6][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[6][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[6][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[6][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[6][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[6][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[6][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[6][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][6] += 1
                        else : 
                            dico_compo[compo[1]][6] += 1

                else : 
                    for i in range(len(liste_attente)):
                        liste_echec.append(liste_attente[i])

                    compteur_relance[0] += 1
                    moyenne_duration[0] += duration
                    moyenne_type_passes[0][0] += array_hauteur_pass[0]
                    moyenne_type_passes[0][1] += array_hauteur_pass[1]
                    moyenne_type_passes[0][2] += array_hauteur_pass[2]
                    moyenne_nb_passes[0] += nb_pass
                    moyenne_long_par_la_passe[0] += pass_length
                    moyenne_nb_carry[0] += nb_carry
                    moyenne_long_par_le_carry[0] += dist_carry
                    moyenne_pied_fort[0][0] += pref_foot[0]
                    moyenne_pied_fort[0][1] += pref_foot[1]
                    for j in range(len(weak_foot)):
                        moyenne_pied_faible[0][j] += weak_foot[j]
                    for j in range(len(skills)):
                        moyenne_skills[0][j] += skills[j]
                    for j in range(len(IMC)):
                        moyenne_IMC[0][j] += IMC[j]
                    for j in range(len(passing)):
                        moyenne_passing[0][j] += passing[j]
                    for j in range(len(mentality_pos)):
                        moyenne_mentality_pos[0][j] += mentality_pos[j]
                    for j in range(len(mentality_vis)):
                        moyenne_mentality_vis[0][j] += mentality_vis[j]
                    for j in range(len(long_passing)):
                        moyenne_long_passing[0][j] += long_passing[j]
                    for j in range(len(pace)):
                        moyenne_pace[0][j] += pace[j]
                    for j in range(len(ball_control)):
                        moyenne_ball_control[0][j] += ball_control[j]
                    for j in range(len(short_passing)):
                        moyenne_short_passing[0][j] += short_passing[j]
                    for j in range(len(league)):
                        moyenne_league[0][j] += league[j]
                    for j in range(len(work_rate)):
                        moyenne_work_rate[0][j] += work_rate[j]
                    for j in range(len(physic)):
                        moyenne_physic[0][j] += physic[j]
                    for j in range(len(accel)):
                        moyenne_accel[0][j] += accel[j]
                    for j in range(len(strength)):
                        moyenne_strength[0][j] += strength[j]
                    for j in range(len(agility)):
                        moyenne_agility[0][j] += agility[j]
                    for j in range(len(aggression)):
                        moyenne_aggression[0][j] += aggression[j]
                    for j in range(len(def_central)):
                        center_back[0][j] += def_central[j]
                    for j in range(len(back_aile)):
                        back[0][j] += back_aile[j]
                    
                    if df_event['team'][index] == team[0]:
                        dico_compo[compo[0]][0] += 1
                    else : 
                        dico_compo[compo[1]][0] += 1
                
                numero_relance += 1
    
    for i in range(len(moyenne_duration)):
        if compteur_relance[i] != 0 :
            moyenne_duration[i] /= compteur_relance[i]
            moyenne_nb_passes[i] /= compteur_relance[i]
            moyenne_long_par_la_passe[i] /= compteur_relance[i]
            moyenne_nb_carry[i] /= compteur_relance[i]
            moyenne_long_par_le_carry[i] /= compteur_relance[i]
            
    return moyenne_duration, moyenne_type_passes, moyenne_nb_passes, moyenne_long_par_la_passe, moyenne_nb_carry, moyenne_long_par_le_carry, moyenne_pied_fort, moyenne_pied_faible, moyenne_skills, moyenne_IMC, moyenne_passing, moyenne_mentality_pos, moyenne_mentality_vis, moyenne_long_passing, moyenne_pace, moyenne_ball_control, moyenne_short_passing, moyenne_league, moyenne_work_rate, moyenne_physic, moyenne_accel, moyenne_strength, moyenne_agility, moyenne_aggression

for i in WC2022_games :
    mean_duration, mean_type_passes, mean_nb_passes, mean_long_par_la_passe, mean_nb_carry, mean_long_par_le_carry, mean_pied_fort, mean_pied_faible, mean_skills, mean_IMC, mean_passing, mean_mentality_pos, mean_mentality_vis, mean_long_passing, mean_pace, mean_ball_control, mean_short_passing, mean_league, mean_work_rate, mean_physic, mean_accel, mean_strength, mean_agility, mean_aggression = relance_ball_recovery(i)
    
    for j in range(len(mean_duration)):
        if mean_duration[j] != 0:
            compteur_duration[j] += 1
    for j in range(len(mean_pied_fort)):
        if mean_pied_fort[j][0] != 0 or mean_pied_fort[j][1] != 0 :
            compteur_pied_fort[j] += 1

    for j in range(len(duration)):
        duration[j] += mean_duration[j]
        nb_passes[j] += mean_nb_passes[j]
        long_par_la_passe[j] += mean_long_par_la_passe[j]
        nb_carry[j] += mean_nb_carry[j]
        long_par_le_carry[j] += mean_long_par_le_carry[j]
        for k in range(len(type_passes[0])):
            type_passes[j][k] += mean_type_passes[j][k]
        for k in range(len(pied_fort[0])):
            pied_fort[j][k] += mean_pied_fort[j][k]
        for k in range(len(pied_faible[0])):
            pied_faible[j][k] += mean_pied_faible[j][k]
        for k in range(len(skills[0])):
            skills[j][k] += mean_skills[j][k]
        for k in range(len(IMC[0])):
            IMC[j][k] += mean_IMC[j][k]
        for k in range(len(passing[0])):
            passing[j][k] += mean_passing[j][k]
        for k in range(len(mentality_pos[0])):
            mentality_pos[j][k] += mean_mentality_pos[j][k]
        for k in range(len(mentality_vis[0])):
            mentality_vis[j][k] += mean_mentality_vis[j][k]
        for k in range(len(long_passing[0])):
            long_passing[j][k] += mean_long_passing[j][k]
        for k in range(len(pace[0])):
            pace[j][k] += mean_pace[j][k]
        for k in range(len(ball_control[0])):
            ball_control[j][k] += mean_ball_control[j][k]
        for k in range(len(short_passing[0])):
            short_passing[j][k] += mean_short_passing[j][k]
        for k in range(len(league[0])):
            league[j][k] += mean_league[j][k]
        for k in range(len(work_rate[0])):
            work_rate[j][k] += mean_work_rate[j][k]
        for k in range(len(physic[0])):
            physic[j][k] += mean_physic[j][k]
        for k in range(len(accel[0])):
            accel[j][k] += mean_accel[j][k]
        for k in range(len(strength[0])):
            strength[j][k] += mean_strength[j][k]
        for k in range(len(agility[0])):
            agility[j][k] += mean_agility[j][k]
        for k in range(len(aggression[0])):
            aggression[j][k] += mean_aggression[j][k]


### THROW-IN ###
liste_reussi = []
liste_echec = []


""" On part d'une rentrée en touche et on avance jusqu'à la fin de la phase de relance """
def relance_rentrée(match_id):
    df_event = sb.events(match_id)
    df_lineup = sb.lineups(match_id)
    Type = df_event['type']
    types_exclus = ["Dispossessed", "Error", "Miscontrol", "Block", "Clearance", "Interception", "Pressure",  "Foul Committed"]
    numero_relance = 1
    
    # Variable comptant le nombre de relances ayant un packing -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    compteur_relance = [0, 0, 0, 0, 0, 0, 0]

    # Variable donnant le nombre moyen de passes d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_nb_passes = [0, 0, 0, 0, 0, 0, 0]

    # Variable donnant la longueur moyenne parcourue par la passe d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_long_par_la_passe = [0, 0, 0, 0, 0, 0, 0]

    # Variable donnant le nombre moyen de chaque type de passes pour les relances ayant un packing -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_type_passes = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]] # [0] : Ground, [1] : Low, [2] : High
    
    # variable donnant la duree moyenne d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_duration = [0, 0, 0, 0, 0, 0, 0]
    
    # Variable donnant le nombre moyen de carries d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_nb_carry = [0, 0, 0, 0, 0, 0, 0]

    # Variable donnant la longueur moyenne parcourue par le carry d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_long_par_le_carry = [0, 0, 0, 0, 0, 0, 0]

    # Variable donnant la repartition du nombre de touches moyen par un droitier / gaucher d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_pied_fort = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]] # [0] : gaucher, [1] : droitier
    
    # Variable donnant la repartition (de 1/5 à 5/5) de la qualité de pied faible des joueurs d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_pied_faible = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]] # [0] : 1/5, [1] : 2/5, [2] : 3/5, [3] : 4/5, [4] : 5/5
    
    # Variable donnant la repartition (de 1/5 à 5/5) des skills des joueurs d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_skills = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]] # [0] : 1/5, [1] : 2/5, [2] : 3/5, [3] : 4/5, [4] : 5/5
    
    # Variable donnant la répartition (de 19-20 à 25-26) de l'IMC des joueurs d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_IMC = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]] # [0] : 19-20, [1] : 20-21, [2] : 21-22, [3] : 22-23, [4] : 23-24, [5] : 24-25, [6] : 25-26
    
    # Variable donnant la qualité de passe des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_passing = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90
    
    # Variable donnant la qualité de mentality positioning des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_mentality_pos = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la qualité de mentality positioning des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_mentality_vis = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la qualité de long passing des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_long_passing = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la qualité de vitesse des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_pace = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la qualité de controle des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_ball_control = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la qualité de passe courte des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_short_passing = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la repartition des championnats des joueurs d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_league = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]] # [0] : Ligue 1, [1] : Liga, [2] : Bundesliga, [3] : Serie A, [4] : Premier League

    # Variable donnant le work-rate des joueurs d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_work_rate = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]] # Le premier est l'apport off et le deuxieme l'apport def : [0] : High/High, [1] : High/Medium, [2] : High/Low, [3] : Medium/High, [4] : Medium/Medium, [5] : Medium/Low, [6] : Low/High, [7] : Low/Medium, [8] : Low/Low

    # Variable donnant la qualité physique des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_physic = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la qualité d'acceleration des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_accel = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90
    
    # Variable donnant la force des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_strength = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90
    
    # Variable donnant l'agilité' des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_agility = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90
    
    # Variable donnant l'agressivité des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_aggression = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90
    
    index_suiv = 0
    nb_nonPress = 0
    nb_press = 0
    first_index = 0

    compo = []
    team = []
    
    for index, chaine in enumerate(Type):
        if chaine == "Starting XI":
            compo.append(df_event['tactics'][index]['formation'])
            team.append(df_event['team'][index])
        if chaine == "Pass" and df_event.loc[index, 'pass_type'] == "Throw-in" and df_event['location'][index][0] <= 30: #and df_event['location'][index][0] <= 20
            array_hauteur_pass = [0,0,0]
            nb_pass = 0
            pass_length = 0
            dist_carry = 0
            nb_carry = 0
            duration = 0
            nb_press = 0
            nb_nonPress = 0
            first_index = index
            liste_attente = []

            pref_foot = [0, 0]

            skills = [0, 0, 0, 0, 0] # 1/5, 2/5, 3/5, 4/5, 5/5

            weak_foot = [0, 0, 0, 0, 0] # 1/5, 2/5, 3/5, 4/5, 5/5

            IMC = [0, 0, 0, 0, 0, 0, 0] # 19-20, 20-21, 21-22, 22-23, 23-24, 24-25, 25-26

            passing = [0, 0, 0, 0, 0, 0]

            mentality_pos = [0, 0, 0, 0, 0, 0]

            mentality_vis = [0, 0, 0, 0, 0, 0]

            long_passing = [0, 0, 0, 0, 0, 0]

            pace = [0, 0, 0, 0, 0, 0]

            ball_control = [0, 0, 0, 0, 0, 0]

            short_passing = [0, 0, 0, 0, 0, 0]

            league = [0, 0, 0, 0, 0]

            work_rate = [0, 0, 0, 0, 0, 0, 0, 0, 0]

            physic = [0, 0, 0, 0, 0, 0]

            accel = [0, 0, 0, 0, 0, 0]

            strength = [0, 0, 0, 0, 0, 0]

            agility = [0, 0, 0, 0, 0, 0]

            aggression = [0, 0, 0, 0, 0, 0]

            def_central = [0, 0, 0, 0]

            back_aile = [0, 0, 0, 0]

            start_time = datetime.strptime(df_event['timestamp'][index], "%H:%M:%S.%f")
            
            time_suiv, index_suiv, periode_suiv = timestamp_suiv(index, match_id)
        
            if len(trouver_joueur(df_event['player'][index], 'players_22.csv')) != 0 :
                team = df_event['team'][index]
                player_id = df_event['player_id'][index]
                for i in range(len(df_lineup[team]['player_id'])):
                    if player_id == df_lineup[team]['player_id'][i] and len(df_lineup[team]['positions'][i]) != 0:
                        if df_lineup[team]['positions'][i][0]['position'] == "Left Center Back" :
                            pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                            if pied == "Left" :
                                def_central[0] += 1
                            elif pied == "Right" :
                                def_central[1] += 1
                        elif df_lineup[team]['positions'][i][0]['position'] == "Right Center Back" :
                            pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                            if pied == "Left" :
                                def_central[2] += 1
                            elif pied == "Right" :
                                def_central[3] += 1
                        elif df_lineup[team]['positions'][i][0]['position'] == "Left Back" :
                            pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                            if pied == "Left" :
                                back_aile[0] += 1
                            elif pied == "Right" :
                                back_aile[1] += 1
                        elif df_lineup[team]['positions'][i][0]['position'] == "Right Back" :
                            pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                            if pied == "Left" :
                                back_aile[2] += 1
                            elif pied == "Right" :
                                back_aile[3] += 1


                if data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == "Left" :
                    pref_foot[0] += 1
                else : 
                    pref_foot[1] += 1

                if data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                    skills[0] += 1
                elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                    skills[1] += 1
                elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                    skills[2] += 1
                elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                    skills[3] += 1
                else :
                    skills[4] += 1

                player_IMC = calcul_imc(data_players['weight_kg'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]], data_players['height_cm'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]])

                if player_IMC <= 20 :
                    IMC[0] += 1
                elif 20 < player_IMC <= 21 :
                    IMC[1] += 1
                elif 21 < player_IMC <= 22 :
                    IMC[2] += 1
                elif 22 < player_IMC <= 23 :
                    IMC[3] += 1
                elif 23 < player_IMC <= 24 :
                    IMC[4] += 1
                elif 24 < player_IMC <= 25 :
                    IMC[5] += 1
                else :
                    IMC[6] += 1

                if data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                    weak_foot[0] += 1
                elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                    weak_foot[1] += 1
                elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                    weak_foot[2] += 1
                elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                    weak_foot[3] += 1
                else :
                    weak_foot[4] += 1

                player_passing = data_players['passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                if data_players['club_position'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] != "GK" :
                    if player_passing <= 50:
                        passing[0] += 1
                    elif 50 < player_passing <= 60:
                        passing[1] += 1
                    elif 60 < player_passing <= 70:
                        passing[2] += 1
                    elif 70 < player_passing <= 80:
                        passing[3] += 1
                    elif 80 < player_passing <= 90:
                        passing[4] += 1
                    else : 
                        passing[5] += 1

                player_mentality_pos = data_players['mentality_positioning'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                if player_mentality_pos <= 50:
                    mentality_pos[0] += 1
                elif 50 < player_mentality_pos <= 60:
                    mentality_pos[1] += 1
                elif 60 < player_mentality_pos <= 70:
                    mentality_pos[2] += 1
                elif 70 < player_mentality_pos <= 80:
                    mentality_pos[3] += 1
                elif 80 < player_mentality_pos <= 90:
                    mentality_pos[4] += 1
                else : 
                    mentality_pos[5] += 1

                player_mentality_vis = data_players['mentality_vision'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                if player_mentality_vis <= 50:
                    mentality_vis[0] += 1
                elif 50 < player_mentality_vis <= 60:
                    mentality_vis[1] += 1
                elif 60 < player_mentality_vis <= 70:
                    mentality_vis[2] += 1
                elif 70 < player_mentality_vis <= 80:
                    mentality_vis[3] += 1
                elif 80 < player_mentality_vis <= 90:
                    mentality_vis[4] += 1
                else : 
                    mentality_vis[5] += 1

                player_long_passing = data_players['skill_long_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                if player_long_passing <= 50:
                    long_passing[0] += 1
                elif 50 < player_long_passing <= 60:
                    long_passing[1] += 1
                elif 60 < player_long_passing <= 70:
                    long_passing[2] += 1
                elif 70 < player_long_passing <= 80:
                    long_passing[3] += 1
                elif 80 < player_long_passing <= 90:
                    long_passing[4] += 1
                else : 
                    long_passing[5] += 1

                player_pace = data_players['pace'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                if player_pace <= 50:
                    pace[0] += 1
                elif 50 < player_pace <= 60:
                    pace[1] += 1
                elif 60 < player_pace <= 70:
                    pace[2] += 1
                elif 70 < player_pace <= 80:
                    pace[3] += 1
                elif 80 < player_pace <= 90:
                    pace[4] += 1
                else : 
                    pace[5] += 1

                player_ball_control = data_players['skill_ball_control'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_ball_control <= 50:
                    ball_control[0] += 1
                elif 50 < player_ball_control <= 60:
                    ball_control[1] += 1
                elif 60 < player_ball_control <= 70:
                    ball_control[2] += 1
                elif 70 < player_ball_control <= 80:
                    ball_control[3] += 1
                elif 80 < player_ball_control <= 90:
                    ball_control[4] += 1
                else : 
                    ball_control[5] += 1

                player_short_passing = data_players['attacking_short_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_short_passing <= 50:
                    short_passing[0] += 1
                elif 50 < player_short_passing <= 60:
                    short_passing[1] += 1
                elif 60 < player_short_passing <= 70:
                    short_passing[2] += 1
                elif 70 < player_short_passing <= 80:
                    short_passing[3] += 1
                elif 80 < player_short_passing <= 90:
                    short_passing[4] += 1
                else : 
                    short_passing[5] += 1

                player_league = data_players['league_name'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_league == "French Ligue 1":
                    league[0] += 1
                elif player_league == "Spain Primera Division":
                    league[1] += 1
                elif player_league == "German 1. Bundesliga":
                    league[2] += 1
                elif player_league == "Italian Serie A":
                    league[3] += 1
                elif player_league == "English Premier League":
                    league[4] += 1

                player_work_rate = data_players['work_rate'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_work_rate == "High/High":
                    work_rate[0] += 1
                elif player_work_rate == "High/Medium":
                    work_rate[1] += 1
                elif player_work_rate == "High/Low":
                    work_rate[2] += 1
                elif player_work_rate == "Medium/High":
                    work_rate[3] += 1
                elif player_work_rate == "Medium/Medium":
                    work_rate[4] += 1
                elif player_work_rate == "Medium/Low":
                    work_rate[5] += 1
                elif player_work_rate == "Low/High":
                    work_rate[6] += 1
                elif player_work_rate == "Low/Medium":
                    work_rate[7] += 1
                elif player_work_rate == "Low/Low": 
                    work_rate[8] += 1

                player_physic = data_players['physic'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_physic <= 50:
                    physic[0] += 1
                elif 50 < player_physic <= 60:
                    physic[1] += 1
                elif 60 < player_physic <= 70:
                    physic[2] += 1
                elif 70 < player_physic <= 80:
                    physic[3] += 1
                elif 80 < player_physic <= 90:
                    physic[4] += 1
                else : 
                    physic[5] += 1

                player_accel = data_players['movement_acceleration'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_accel <= 50:
                    accel[0] += 1
                elif 50 < player_accel <= 60:
                    accel[1] += 1
                elif 60 < player_accel <= 70:
                    accel[2] += 1
                elif 70 < player_accel <= 80:
                    accel[3] += 1
                elif 80 < player_accel <= 90:
                    accel[4] += 1
                else : 
                    accel[5] += 1

                player_strength = data_players['power_strength'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_strength <= 50:
                    strength[0] += 1
                elif 50 < player_strength <= 60:
                    strength[1] += 1
                elif 60 < player_strength <= 70:
                    strength[2] += 1
                elif 70 < player_strength <= 80:
                    strength[3] += 1
                elif 80 < player_strength <= 90:
                    strength[4] += 1
                else : 
                    strength[5] += 1

                player_agility = data_players['movement_agility'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_agility <= 50:
                    agility[0] += 1
                elif 50 < player_agility <= 60:
                    agility[1] += 1
                elif 60 < player_agility <= 70:
                    agility[2] += 1
                elif 70 < player_agility <= 80:
                    agility[3] += 1
                elif 80 < player_agility <= 90:
                    agility[4] += 1
                else : 
                    agility[5] += 1

                player_aggression = data_players['mentality_aggression'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_aggression <= 50:
                    aggression[0] += 1
                elif 50 < player_aggression <= 60:
                    aggression[1] += 1
                elif 60 < player_aggression <= 70:
                    aggression[2] += 1
                elif 70 < player_aggression <= 80:
                    aggression[3] += 1
                elif 80 < player_aggression <= 90:
                    aggression[4] += 1
                else : 
                    aggression[5] += 1

            index_final = 0
            liste_attente.append([df_event['location'][index][0], df_event['location'][index][1], df_event['pass_end_location'][index][0], df_event['pass_end_location'][index][1], df_event['type'][index], numero_relance])
            while (df_event['location'][index_suiv][0] < 60 and df_event.loc[index_suiv, 'type'] not in types_exclus and df_event['pass_outcome'][index_suiv] != "Incomplete" and df_event['pass_outcome'][index_suiv] != "Out" and df_event['pass_outcome'][index_suiv] != "Pass Offside" and df_event['pass_outcome'][index_suiv] != "Unknown" and df_event['dribble_outcome'][index_suiv] != "Incomplete" and df_event['type'][index_suiv] != "Foul Won"):
                index = index_suiv
                
                if len(trouver_joueur(df_event['player'][index], 'players_22.csv')) != 0 :
                    
                    team = df_event['team'][index]
                    player_id = df_event['player_id'][index]
                    for i in range(len(df_lineup[team]['player_id'])):
                        if player_id == df_lineup[team]['player_id'][i] and len(df_lineup[team]['positions'][i]) != 0:
                            if df_lineup[team]['positions'][i][0]['position'] == "Left Center Back" :
                                pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                if pied == "Left" :
                                    def_central[0] += 1
                                elif pied == "Right" :
                                    def_central[1] += 1
                            elif df_lineup[team]['positions'][i][0]['position'] == "Right Center Back" :
                                pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                if pied == "Left" :
                                    def_central[2] += 1
                                elif pied == "Right" :
                                    def_central[3] += 1
                            elif df_lineup[team]['positions'][i][0]['position'] == "Left Back" :
                                pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                if pied == "Left" :
                                    back_aile[0] += 1
                                elif pied == "Right" :
                                    back_aile[1] += 1
                            elif df_lineup[team]['positions'][i][0]['position'] == "Right Back" :
                                pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                if pied == "Left" :
                                    back_aile[2] += 1
                                elif pied == "Right" :
                                    back_aile[3] += 1


                    if data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == "Left" :
                        pref_foot[0] += 1
                    else : 
                        pref_foot[1] += 1

                    if data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                        skills[0] += 1
                    elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                        skills[1] += 1
                    elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                        skills[2] += 1
                    elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                        skills[3] += 1
                    else :
                        skills[4] += 1

                    player_IMC = calcul_imc(data_players['weight_kg'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]], data_players['height_cm'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]])

                    if player_IMC <= 20 :
                        IMC[0] += 1
                    elif 20 < player_IMC <= 21 :
                        IMC[1] += 1
                    elif 21 < player_IMC <= 22 :
                        IMC[2] += 1
                    elif 22 < player_IMC <= 23 :
                        IMC[3] += 1
                    elif 23 < player_IMC <= 24 :
                        IMC[4] += 1
                    elif 24 < player_IMC <= 25 :
                        IMC[5] += 1
                    else :
                        IMC[6] += 1

                    if data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                        weak_foot[0] += 1
                    elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                        weak_foot[1] += 1
                    elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                        weak_foot[2] += 1
                    elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                        weak_foot[3] += 1
                    else :
                        weak_foot[4] += 1
                    
                    player_passing = data_players['passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if data_players['club_position'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] != "GK" :
                        if player_passing <= 50:
                            passing[0] += 1
                        elif 50 < player_passing <= 60:
                            passing[1] += 1
                        elif 60 < player_passing <= 70:
                            passing[2] += 1
                        elif 70 < player_passing <= 80:
                            passing[3] += 1
                        elif 80 < player_passing <= 90:
                            passing[4] += 1
                        else : 
                            passing[5] += 1
                    
                    player_mentality_pos = data_players['mentality_positioning'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if player_mentality_pos <= 50:
                        mentality_pos[0] += 1
                    elif 50 < player_mentality_pos <= 60:
                        mentality_pos[1] += 1
                    elif 60 < player_mentality_pos <= 70:
                        mentality_pos[2] += 1
                    elif 70 < player_mentality_pos <= 80:
                        mentality_pos[3] += 1
                    elif 80 < player_mentality_pos <= 90:
                        mentality_pos[4] += 1
                    else : 
                        mentality_pos[5] += 1

                    player_mentality_vis = data_players['mentality_vision'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if player_mentality_vis <= 50:
                        mentality_vis[0] += 1
                    elif 50 < player_mentality_vis <= 60:
                        mentality_vis[1] += 1
                    elif 60 < player_mentality_vis <= 70:
                        mentality_vis[2] += 1
                    elif 70 < player_mentality_vis <= 80:
                        mentality_vis[3] += 1
                    elif 80 < player_mentality_vis <= 90:
                        mentality_vis[4] += 1
                    else : 
                        mentality_vis[5] += 1

                    player_long_passing = data_players['skill_long_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if player_long_passing <= 50:
                        long_passing[0] += 1
                    elif 50 < player_long_passing <= 60:
                        long_passing[1] += 1
                    elif 60 < player_long_passing <= 70:
                        long_passing[2] += 1
                    elif 70 < player_long_passing <= 80:
                        long_passing[3] += 1
                    elif 80 < player_long_passing <= 90:
                        long_passing[4] += 1
                    else : 
                        long_passing[5] += 1
                    
                    player_pace = data_players['pace'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if player_pace <= 50:
                        pace[0] += 1
                    elif 50 < player_pace <= 60:
                        pace[1] += 1
                    elif 60 < player_pace <= 70:
                        pace[2] += 1
                    elif 70 < player_pace <= 80:
                        pace[3] += 1
                    elif 80 < player_pace <= 90:
                        pace[4] += 1
                    else : 
                        pace[5] += 1

                    player_ball_control = data_players['skill_ball_control'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_ball_control <= 50:
                        ball_control[0] += 1
                    elif 50 < player_ball_control <= 60:
                        ball_control[1] += 1
                    elif 60 < player_ball_control <= 70:
                        ball_control[2] += 1
                    elif 70 < player_ball_control <= 80:
                        ball_control[3] += 1
                    elif 80 < player_ball_control <= 90:
                        ball_control[4] += 1
                    else : 
                        ball_control[5] += 1

                    player_short_passing = data_players['attacking_short_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_short_passing <= 50:
                        short_passing[0] += 1
                    elif 50 < player_short_passing <= 60:
                        short_passing[1] += 1
                    elif 60 < player_short_passing <= 70:
                        short_passing[2] += 1
                    elif 70 < player_short_passing <= 80:
                        short_passing[3] += 1
                    elif 80 < player_short_passing <= 90:
                        short_passing[4] += 1
                    else : 
                        short_passing[5] += 1

                    player_league = data_players['league_name'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_league == "French Ligue 1":
                        league[0] += 1
                    elif player_league == "Spain Primera Division":
                        league[1] += 1
                    elif player_league == "German 1. Bundesliga":
                        league[2] += 1
                    elif player_league == "Italian Serie A":
                        league[3] += 1
                    elif player_league == "English Premier League":
                        league[4] += 1

                    player_work_rate = data_players['work_rate'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_work_rate == "High/High":
                        work_rate[0] += 1
                    elif player_work_rate == "High/Medium":
                        work_rate[1] += 1
                    elif player_work_rate == "High/Low":
                        work_rate[2] += 1
                    elif player_work_rate == "Medium/High":
                        work_rate[3] += 1
                    elif player_work_rate == "Medium/Medium":
                        work_rate[4] += 1
                    elif player_work_rate == "Medium/Low":
                        work_rate[5] += 1
                    elif player_work_rate == "Low/High":
                        work_rate[6] += 1
                    elif player_work_rate == "Low/Medium":
                        work_rate[7] += 1
                    elif player_work_rate == "Low/Low": 
                        work_rate[8] += 1

                    player_physic = data_players['physic'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_physic <= 50:
                        physic[0] += 1
                    elif 50 < player_physic <= 60:
                        physic[1] += 1
                    elif 60 < player_physic <= 70:
                        physic[2] += 1
                    elif 70 < player_physic <= 80:
                        physic[3] += 1
                    elif 80 < player_physic <= 90:
                        physic[4] += 1
                    else : 
                        physic[5] += 1

                    player_accel = data_players['movement_acceleration'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_accel <= 50:
                        accel[0] += 1
                    elif 50 < player_accel <= 60:
                        accel[1] += 1
                    elif 60 < player_accel <= 70:
                        accel[2] += 1
                    elif 70 < player_accel <= 80:
                        accel[3] += 1
                    elif 80 < player_accel <= 90:
                        accel[4] += 1
                    else : 
                        accel[5] += 1

                    player_strength = data_players['power_strength'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_strength <= 50:
                        strength[0] += 1
                    elif 50 < player_strength <= 60:
                        strength[1] += 1
                    elif 60 < player_strength <= 70:
                        strength[2] += 1
                    elif 70 < player_strength <= 80:
                        strength[3] += 1
                    elif 80 < player_strength <= 90:
                        strength[4] += 1
                    else : 
                        strength[5] += 1

                    player_agility = data_players['movement_agility'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_agility <= 50:
                        agility[0] += 1
                    elif 50 < player_agility <= 60:
                        agility[1] += 1
                    elif 60 < player_agility <= 70:
                        agility[2] += 1
                    elif 70 < player_agility <= 80:
                        agility[3] += 1
                    elif 80 < player_agility <= 90:
                        agility[4] += 1
                    else : 
                        agility[5] += 1

                    player_aggression = data_players['mentality_aggression'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_aggression <= 50:
                        aggression[0] += 1
                    elif 50 < player_aggression <= 60:
                        aggression[1] += 1
                    elif 60 < player_aggression <= 70:
                        aggression[2] += 1
                    elif 70 < player_aggression <= 80:
                        aggression[3] += 1
                    elif 80 < player_aggression <= 90:
                        aggression[4] += 1
                    else : 
                        aggression[5] += 1
                
                if df_event['under_pressure'][index] == True: 
                    nb_press += 1
                else : 
                    nb_nonPress += 1

                if df_event['type'][index] == "Pass":
                    if df_event['pass_height'][index] == "Ground Pass":
                        array_hauteur_pass[0] += 1
                    if df_event['pass_height'][index] == "Low Pass":
                        array_hauteur_pass[1] += 1
                    if df_event['pass_height'][index] == "High Pass":
                        array_hauteur_pass[2] += 1
                    pass_length += df_event['pass_length'][index]
                    nb_pass += 1
                    duration += df_event['duration'][index]
                    liste_attente.append([df_event['location'][index][0], df_event['location'][index][1], df_event['pass_end_location'][index][0], df_event['pass_end_location'][index][1], df_event['type'][index], numero_relance])
                if df_event['type'][index] == "Carry":
                    nb_carry += 1
                    dist_carry += distance_entre_points(df_event['location'][index][0], df_event['location'][index][1], df_event['carry_end_location'][index][0], df_event['carry_end_location'][index][1])
                    liste_attente.append([df_event['location'][index][0], df_event['location'][index][1], df_event['carry_end_location'][index][0], df_event['carry_end_location'][index][1], df_event['type'][index], numero_relance])
                    duration += df_event['duration'][index]
                
                time_suiv, index_suiv, periode_suiv = timestamp_suiv(index, match_id)
                if not isinstance(df_event['location'][index_suiv], list): 
                    index_suiv = index
                    break
                index_final = index_suiv

            if index_final == 0 :
                
                if len(trouver_joueur(df_event['player'][index], 'players_22.csv')) != 0 :
                    

                    team = df_event['team'][index]
                    player_id = df_event['player_id'][index]
                    for i in range(len(df_lineup[team]['player_id'])):
                        if player_id == df_lineup[team]['player_id'][i] and len(df_lineup[team]['positions'][i]) != 0:
                            if df_lineup[team]['positions'][i][0]['position'] == "Left Center Back" :
                                pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                if pied == "Left" :
                                    def_central[0] += 1
                                elif pied == "Right" :
                                    def_central[1] += 1
                            elif df_lineup[team]['positions'][i][0]['position'] == "Right Center Back" :
                                pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                if pied == "Left" :
                                    def_central[2] += 1
                                elif pied == "Right" :
                                    def_central[3] += 1
                            elif df_lineup[team]['positions'][i][0]['position'] == "Left Back" :
                                pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                if pied == "Left" :
                                    back_aile[0] += 1
                                elif pied == "Right" :
                                    back_aile[1] += 1
                            elif df_lineup[team]['positions'][i][0]['position'] == "Right Back" :
                                pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                if pied == "Left" :
                                    back_aile[2] += 1
                                elif pied == "Right" :
                                    back_aile[3] += 1


                    if data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == "Left" :
                        pref_foot[0] += 1
                    else : 
                        pref_foot[1] += 1

                    if data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                        skills[0] += 1
                    elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                        skills[1] += 1
                    elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                        skills[2] += 1
                    elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                        skills[3] += 1
                    else :
                        skills[4] += 1

                    player_IMC = calcul_imc(data_players['weight_kg'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]], data_players['height_cm'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]])

                    if player_IMC <= 20 :
                        IMC[0] += 1
                    elif 20 < player_IMC <= 21 :
                        IMC[1] += 1
                    elif 21 < player_IMC <= 22 :
                        IMC[2] += 1
                    elif 22 < player_IMC <= 23 :
                        IMC[3] += 1
                    elif 23 < player_IMC <= 24 :
                        IMC[4] += 1
                    elif 24 < player_IMC <= 25 :
                        IMC[5] += 1
                    else :
                        IMC[6] += 1

                    if data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                        weak_foot[0] += 1
                    elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                        weak_foot[1] += 1
                    elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                        weak_foot[2] += 1
                    elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                        weak_foot[3] += 1
                    else :
                        weak_foot[4] += 1

                    player_passing = data_players['passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if data_players['club_position'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] != "GK" :
                        if player_passing <= 50:
                            passing[0] += 1
                        elif 50 < player_passing <= 60:
                            passing[1] += 1
                        elif 60 < player_passing <= 70:
                            passing[2] += 1
                        elif 70 < player_passing <= 80:
                            passing[3] += 1
                        elif 80 < player_passing <= 90:
                            passing[4] += 1
                        else : 
                            passing[5] += 1

                    player_mentality_pos = data_players['mentality_positioning'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if player_mentality_pos <= 50:
                        mentality_pos[0] += 1
                    elif 50 < player_mentality_pos <= 60:
                        mentality_pos[1] += 1
                    elif 60 < player_mentality_pos <= 70:
                        mentality_pos[2] += 1
                    elif 70 < player_mentality_pos <= 80:
                        mentality_pos[3] += 1
                    elif 80 < player_mentality_pos <= 90:
                        mentality_pos[4] += 1
                    else : 
                        mentality_pos[5] += 1

                    player_mentality_vis = data_players['mentality_vision'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if player_mentality_vis <= 50:
                        mentality_vis[0] += 1
                    elif 50 < player_mentality_vis <= 60:
                        mentality_vis[1] += 1
                    elif 60 < player_mentality_vis <= 70:
                        mentality_vis[2] += 1
                    elif 70 < player_mentality_vis <= 80:
                        mentality_vis[3] += 1
                    elif 80 < player_mentality_vis <= 90:
                        mentality_vis[4] += 1
                    else : 
                        mentality_vis[5] += 1

                    player_long_passing = data_players['skill_long_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if player_long_passing <= 50:
                        long_passing[0] += 1
                    elif 50 < player_long_passing <= 60:
                        long_passing[1] += 1
                    elif 60 < player_long_passing <= 70:
                        long_passing[2] += 1
                    elif 70 < player_long_passing <= 80:
                        long_passing[3] += 1
                    elif 80 < player_long_passing <= 90:
                        long_passing[4] += 1
                    else : 
                        long_passing[5] += 1
                    
                    player_pace = data_players['pace'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if player_pace <= 50:
                        pace[0] += 1
                    elif 50 < player_pace <= 60:
                        pace[1] += 1
                    elif 60 < player_pace <= 70:
                        pace[2] += 1
                    elif 70 < player_pace <= 80:
                        pace[3] += 1
                    elif 80 < player_pace <= 90:
                        pace[4] += 1
                    else : 
                        pace[5] += 1

                    player_ball_control = data_players['skill_ball_control'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_ball_control <= 50:
                        ball_control[0] += 1
                    elif 50 < player_ball_control <= 60:
                        ball_control[1] += 1
                    elif 60 < player_ball_control <= 70:
                        ball_control[2] += 1
                    elif 70 < player_ball_control <= 80:
                        ball_control[3] += 1
                    elif 80 < player_ball_control <= 90:
                        ball_control[4] += 1
                    else : 
                        ball_control[5] += 1

                    player_short_passing = data_players['attacking_short_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_short_passing <= 50:
                        short_passing[0] += 1
                    elif 50 < player_short_passing <= 60:
                        short_passing[1] += 1
                    elif 60 < player_short_passing <= 70:
                        short_passing[2] += 1
                    elif 70 < player_short_passing <= 80:
                        short_passing[3] += 1
                    elif 80 < player_short_passing <= 90:
                        short_passing[4] += 1
                    else : 
                        short_passing[5] += 1

                    player_league = data_players['league_name'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_league == "French Ligue 1":
                        league[0] += 1
                    elif player_league == "Spain Primera Division":
                        league[1] += 1
                    elif player_league == "German 1. Bundesliga":
                        league[2] += 1
                    elif player_league == "Italian Serie A":
                        league[3] += 1
                    elif player_league == "English Premier League":
                        league[4] += 1

                    player_work_rate = data_players['work_rate'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_work_rate == "High/High":
                        work_rate[0] += 1
                    elif player_work_rate == "High/Medium":
                        work_rate[1] += 1
                    elif player_work_rate == "High/Low":
                        work_rate[2] += 1
                    elif player_work_rate == "Medium/High":
                        work_rate[3] += 1
                    elif player_work_rate == "Medium/Medium":
                        work_rate[4] += 1
                    elif player_work_rate == "Medium/Low":
                        work_rate[5] += 1
                    elif player_work_rate == "Low/High":
                        work_rate[6] += 1
                    elif player_work_rate == "Low/Medium":
                        work_rate[7] += 1
                    elif player_work_rate == "Low/Low": 
                        work_rate[8] += 1

                    player_physic = data_players['physic'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_physic <= 50:
                        physic[0] += 1
                    elif 50 < player_physic <= 60:
                        physic[1] += 1
                    elif 60 < player_physic <= 70:
                        physic[2] += 1
                    elif 70 < player_physic <= 80:
                        physic[3] += 1
                    elif 80 < player_physic <= 90:
                        physic[4] += 1
                    else : 
                        physic[5] += 1

                    player_accel = data_players['movement_acceleration'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_accel <= 50:
                        accel[0] += 1
                    elif 50 < player_accel <= 60:
                        accel[1] += 1
                    elif 60 < player_accel <= 70:
                        accel[2] += 1
                    elif 70 < player_accel <= 80:
                        accel[3] += 1
                    elif 80 < player_accel <= 90:
                        accel[4] += 1
                    else : 
                        accel[5] += 1

                    player_strength = data_players['power_strength'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_strength <= 50:
                        strength[0] += 1
                    elif 50 < player_strength <= 60:
                        strength[1] += 1
                    elif 60 < player_strength <= 70:
                        strength[2] += 1
                    elif 70 < player_strength <= 80:
                        strength[3] += 1
                    elif 80 < player_strength <= 90:
                        strength[4] += 1
                    else : 
                        strength[5] += 1

                    player_agility = data_players['movement_agility'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_agility <= 50:
                        agility[0] += 1
                    elif 50 < player_agility <= 60:
                        agility[1] += 1
                    elif 60 < player_agility <= 70:
                        agility[2] += 1
                    elif 70 < player_agility <= 80:
                        agility[3] += 1
                    elif 80 < player_agility <= 90:
                        agility[4] += 1
                    else : 
                        agility[5] += 1

                    player_aggression = data_players['mentality_aggression'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_aggression <= 50:
                        aggression[0] += 1
                    elif 50 < player_aggression <= 60:
                        aggression[1] += 1
                    elif 60 < player_aggression <= 70:
                        aggression[2] += 1
                    elif 70 < player_aggression <= 80:
                        aggression[3] += 1
                    elif 80 < player_aggression <= 90:
                        aggression[4] += 1
                    else : 
                        aggression[5] += 1
                
                if df_event['under_pressure'][index] == True: 
                    nb_press += 1
                else : 
                    nb_nonPress += 1
                
                if df_event['type'][index] == "Pass":
                    if df_event['pass_height'][index] == "Ground Pass":
                        array_hauteur_pass[0] += 1
                    if df_event['pass_height'][index] == "Low Pass":
                        array_hauteur_pass[1] += 1
                    if df_event['pass_height'][index] == "High Pass":
                        array_hauteur_pass[2] += 1
                    pass_length += df_event['pass_length'][index]
                    nb_pass += 1
                    duration += df_event['duration'][index]
                    liste_attente.append([df_event['location'][index][0], df_event['location'][index][1], df_event['pass_end_location'][index][0], df_event['pass_end_location'][index][1], df_event['type'][index], numero_relance])
                if df_event['type'][index] == "Carry":
                    nb_carry += 1
                    dist_carry += distance_entre_points(df_event['location'][index][0], df_event['location'][index][1], df_event['carry_end_location'][index][0], df_event['carry_end_location'][index][1])   
                    duration += df_event['duration'][index]
                    liste_attente.append([df_event['location'][index][0], df_event['location'][index][1], df_event['carry_end_location'][index][0], df_event['carry_end_location'][index][1], df_event['type'][index], numero_relance])
                if df_event['type'][index_suiv] == "Ball Receipt*" and df_event['ball_receipt_outcome'][index_suiv] != "Incomplete":
                    if df_event['pass_end_location'][index][0] > 60 :
                        for i in range(len(liste_attente)):
                            liste_reussi.append(liste_attente[i])

                        packing = packing_one_event(df_event['id'][index], index, match_id)
                        if packing == 0 or packing == 1 :
                            moyenne_duration[1] += duration
                            compteur_relance[1] += 1
                            moyenne_type_passes[1][0] += array_hauteur_pass[0]
                            moyenne_type_passes[1][1] += array_hauteur_pass[1]
                            moyenne_type_passes[1][2] += array_hauteur_pass[2]
                            moyenne_nb_passes[1] += nb_pass
                            moyenne_long_par_la_passe[1] += pass_length
                            moyenne_nb_carry[1] += nb_carry
                            moyenne_long_par_le_carry[1] += dist_carry
                            moyenne_pied_fort[1][0] += pref_foot[0]
                            moyenne_pied_fort[1][1] += pref_foot[1]
                            for j in range(len(weak_foot)):
                                moyenne_pied_faible[1][j] += weak_foot[j]
                            for j in range(len(skills)):
                                moyenne_skills[1][j] += skills[j]
                            for j in range(len(IMC)):
                                moyenne_IMC[1][j] += IMC[j]
                            for j in range(len(passing)):
                                moyenne_passing[1][j] += passing[j]
                            for j in range(len(mentality_pos)):
                                moyenne_mentality_pos[1][j] += mentality_pos[j]
                            for j in range(len(mentality_vis)):
                                moyenne_mentality_vis[1][j] += mentality_vis[j]
                            for j in range(len(long_passing)):
                                moyenne_long_passing[1][j] += long_passing[j]
                            for j in range(len(pace)):
                                moyenne_pace[1][j] += pace[j]
                            for j in range(len(ball_control)):
                                moyenne_ball_control[1][j] += ball_control[j]
                            for j in range(len(short_passing)):
                                moyenne_short_passing[1][j] += short_passing[j]
                            for j in range(len(league)):
                                moyenne_league[1][j] += league[j]
                            for j in range(len(work_rate)):
                                moyenne_work_rate[1][j] += work_rate[j]
                            for j in range(len(physic)):
                                moyenne_physic[1][j] += physic[j]
                            for j in range(len(accel)):
                                moyenne_accel[1][j] += accel[j]
                            for j in range(len(strength)):
                                moyenne_strength[1][j] += strength[j]
                            for j in range(len(agility)):
                                moyenne_agility[1][j] += agility[j]
                            for j in range(len(aggression)):
                                moyenne_aggression[1][j] += aggression[j]
                            for j in range(len(def_central)):
                                center_back[1][j] += def_central[j]
                            for j in range(len(back_aile)):
                                back[1][j] += back_aile[j]
                            
                            if df_event['team'][index] == team[0]:
                                dico_compo[compo[0]][1] += 1
                            else : 
                                dico_compo[compo[1]][1] += 1
                        if packing == 2 or packing == 3 :
                            moyenne_duration[2] += duration
                            compteur_relance[2] += 1
                            moyenne_type_passes[2][0] += array_hauteur_pass[0]
                            moyenne_type_passes[2][1] += array_hauteur_pass[1]
                            moyenne_type_passes[2][2] += array_hauteur_pass[2]
                            moyenne_nb_passes[2] += nb_pass
                            moyenne_long_par_la_passe[2] += pass_length
                            moyenne_nb_carry[2] += nb_carry
                            moyenne_long_par_le_carry[2] += dist_carry
                            moyenne_pied_fort[2][0] += pref_foot[0]
                            moyenne_pied_fort[2][1] += pref_foot[1]
                            for j in range(len(weak_foot)):
                                moyenne_pied_faible[2][j] += weak_foot[j]
                            for j in range(len(skills)):
                                moyenne_skills[2][j] += skills[j]
                            for j in range(len(IMC)):
                                moyenne_IMC[2][j] += IMC[j]
                            for j in range(len(passing)):
                                moyenne_passing[2][j] += passing[j]
                            for j in range(len(mentality_pos)):
                                moyenne_mentality_pos[2][j] += mentality_pos[j]
                            for j in range(len(mentality_vis)):
                                moyenne_mentality_vis[2][j] += mentality_vis[j]
                            for j in range(len(long_passing)):
                                moyenne_long_passing[2][j] += long_passing[j]
                            for j in range(len(pace)):
                                moyenne_pace[2][j] += pace[j]
                            for j in range(len(ball_control)):
                                moyenne_ball_control[2][j] += ball_control[j]
                            for j in range(len(short_passing)):
                                moyenne_short_passing[2][j] += short_passing[j]
                            for j in range(len(league)):
                                moyenne_league[2][j] += league[j]
                            for j in range(len(work_rate)):
                                moyenne_work_rate[2][j] += work_rate[j]
                            for j in range(len(physic)):
                                moyenne_physic[2][j] += physic[j]
                            for j in range(len(accel)):
                                moyenne_accel[2][j] += accel[j]
                            for j in range(len(strength)):
                                moyenne_strength[2][j] += strength[j]
                            for j in range(len(agility)):
                                moyenne_agility[2][j] += agility[j]
                            for j in range(len(aggression)):
                                moyenne_aggression[2][j] += aggression[j]
                            for j in range(len(def_central)):
                                center_back[2][j] += def_central[j]
                            for j in range(len(back_aile)):
                                back[2][j] += back_aile[j]
                            
                            if df_event['team'][index] == team[0]:
                                dico_compo[compo[0]][2] += 1
                            else : 
                                dico_compo[compo[1]][2] += 1
                        if packing == 4 or packing == 5 :
                            moyenne_duration[3] += duration
                            compteur_relance[3] += 1
                            moyenne_type_passes[3][0] += array_hauteur_pass[0]
                            moyenne_type_passes[3][1] += array_hauteur_pass[1]
                            moyenne_type_passes[3][2] += array_hauteur_pass[2]
                            moyenne_nb_passes[3] += nb_pass
                            moyenne_long_par_la_passe[3] += pass_length
                            moyenne_nb_carry[3] += nb_carry
                            moyenne_long_par_le_carry[3] += dist_carry
                            moyenne_pied_fort[3][0] += pref_foot[0]
                            moyenne_pied_fort[3][1] += pref_foot[1]
                            for j in range(len(weak_foot)):
                                moyenne_pied_faible[3][j] += weak_foot[j]
                            for j in range(len(skills)):
                                moyenne_skills[3][j] += skills[j]
                            for j in range(len(IMC)):
                                moyenne_IMC[3][j] += IMC[j]
                            for j in range(len(passing)):
                                moyenne_passing[3][j] += passing[j]
                            for j in range(len(mentality_pos)):
                                moyenne_mentality_pos[3][j] += mentality_pos[j]
                            for j in range(len(mentality_vis)):
                                moyenne_mentality_vis[3][j] += mentality_vis[j]
                            for j in range(len(long_passing)):
                                moyenne_long_passing[3][j] += long_passing[j]
                            for j in range(len(pace)):
                                moyenne_pace[3][j] += pace[j]
                            for j in range(len(ball_control)):
                                moyenne_ball_control[3][j] += ball_control[j]
                            for j in range(len(short_passing)):
                                moyenne_short_passing[3][j] += short_passing[j]
                            for j in range(len(league)):
                                moyenne_league[3][j] += league[j]
                            for j in range(len(work_rate)):
                                moyenne_work_rate[3][j] += work_rate[j]
                            for j in range(len(physic)):
                                moyenne_physic[3][j] += physic[j]
                            for j in range(len(accel)):
                                moyenne_accel[3][j] += accel[j]
                            for j in range(len(strength)):
                                moyenne_strength[3][j] += strength[j]
                            for j in range(len(agility)):
                                moyenne_agility[3][j] += agility[j]
                            for j in range(len(aggression)):
                                moyenne_aggression[3][j] += aggression[j]
                            for j in range(len(def_central)):
                                center_back[3][j] += def_central[j]
                            for j in range(len(back_aile)):
                                back[3][j] += back_aile[j]
                            
                            if df_event['team'][index] == team[0]:
                                dico_compo[compo[0]][3] += 1
                            else : 
                                dico_compo[compo[1]][3] += 1
                        if packing == 6 or packing == 7 :
                            moyenne_duration[4] += duration
                            compteur_relance[4] += 1
                            moyenne_type_passes[4][0] += array_hauteur_pass[0]
                            moyenne_type_passes[4][1] += array_hauteur_pass[1]
                            moyenne_type_passes[4][2] += array_hauteur_pass[2]
                            moyenne_nb_passes[4] += nb_pass
                            moyenne_long_par_la_passe[4] += pass_length
                            moyenne_nb_carry[4] += nb_carry
                            moyenne_long_par_le_carry[4] += dist_carry
                            moyenne_pied_fort[4][0] += pref_foot[0]
                            moyenne_pied_fort[4][1] += pref_foot[1]
                            for j in range(len(weak_foot)):
                                moyenne_pied_faible[4][j] += weak_foot[j]
                            for j in range(len(skills)):
                                moyenne_skills[4][j] += skills[j]
                            for j in range(len(IMC)):
                                moyenne_IMC[4][j] += IMC[j]
                            for j in range(len(passing)):
                                moyenne_passing[4][j] += passing[j]
                            for j in range(len(mentality_pos)):
                                moyenne_mentality_pos[4][j] += mentality_pos[j]
                            for j in range(len(mentality_vis)):
                                moyenne_mentality_vis[4][j] += mentality_vis[j]
                            for j in range(len(long_passing)):
                                moyenne_long_passing[4][j] += long_passing[j]
                            for j in range(len(pace)):
                                moyenne_pace[4][j] += pace[j]
                            for j in range(len(ball_control)):
                                moyenne_ball_control[4][j] += ball_control[j]
                            for j in range(len(short_passing)):
                                moyenne_short_passing[4][j] += short_passing[j]
                            for j in range(len(league)):
                                moyenne_league[4][j] += league[j]
                            for j in range(len(work_rate)):
                                moyenne_work_rate[4][j] += work_rate[j]
                            for j in range(len(physic)):
                                moyenne_physic[4][j] += physic[j]
                            for j in range(len(accel)):
                                moyenne_accel[4][j] += accel[j]
                            for j in range(len(strength)):
                                moyenne_strength[4][j] += strength[j]
                            for j in range(len(agility)):
                                moyenne_agility[4][j] += agility[j]
                            for j in range(len(aggression)):
                                moyenne_aggression[4][j] += aggression[j]
                            for j in range(len(def_central)):
                                center_back[4][j] += def_central[j]
                            for j in range(len(back_aile)):
                                back[4][j] += back_aile[j]
                            
                            if df_event['team'][index] == team[0]:
                                dico_compo[compo[0]][4] += 1
                            else : 
                                dico_compo[compo[1]][4] += 1
                        if packing == 8 or packing == 9 :
                            moyenne_duration[5] += duration
                            compteur_relance[5] += 1
                            moyenne_type_passes[5][0] += array_hauteur_pass[0]
                            moyenne_type_passes[5][1] += array_hauteur_pass[1]
                            moyenne_type_passes[5][2] += array_hauteur_pass[2]
                            moyenne_nb_passes[5] += nb_pass
                            moyenne_long_par_la_passe[5] += pass_length
                            moyenne_nb_carry[5] += nb_carry
                            moyenne_long_par_le_carry[5] += dist_carry
                            moyenne_pied_fort[5][0] += pref_foot[0]
                            moyenne_pied_fort[5][1] += pref_foot[1]
                            for j in range(len(weak_foot)):
                                moyenne_pied_faible[5][j] += weak_foot[j]
                            for j in range(len(skills)):
                                moyenne_skills[5][j] += skills[j]
                            for j in range(len(IMC)):
                                moyenne_IMC[5][j] += IMC[j]
                            for j in range(len(passing)):
                                moyenne_passing[5][j] += passing[j]
                            for j in range(len(mentality_pos)):
                                moyenne_mentality_pos[5][j] += mentality_pos[j]
                            for j in range(len(mentality_vis)):
                                moyenne_mentality_vis[5][j] += mentality_vis[j]
                            for j in range(len(long_passing)):
                                moyenne_long_passing[5][j] += long_passing[j]
                            for j in range(len(pace)):
                                moyenne_pace[5][j] += pace[j]
                            for j in range(len(ball_control)):
                                moyenne_ball_control[5][j] += ball_control[j]
                            for j in range(len(short_passing)):
                                moyenne_short_passing[5][j] += short_passing[j]
                            for j in range(len(league)):
                                moyenne_league[5][j] += league[j]
                            for j in range(len(work_rate)):
                                moyenne_work_rate[5][j] += work_rate[j]
                            for j in range(len(physic)):
                                moyenne_physic[5][j] += physic[j]
                            for j in range(len(accel)):
                                moyenne_accel[5][j] += accel[j]
                            for j in range(len(strength)):
                                moyenne_strength[5][j] += strength[j]
                            for j in range(len(agility)):
                                moyenne_agility[5][j] += agility[j]
                            for j in range(len(aggression)):
                                moyenne_aggression[5][j] += aggression[j]
                            for j in range(len(def_central)):
                                center_back[5][j] += def_central[j]
                            for j in range(len(back_aile)):
                                back[5][j] += back_aile[j]
                            
                            if df_event['team'][index] == team[0]:
                                dico_compo[compo[0]][5] += 1
                            else : 
                                dico_compo[compo[1]][5] += 1
                        if packing == 10 :
                            moyenne_duration[6] += duration
                            compteur_relance[6] += 1
                            moyenne_type_passes[6][0] += array_hauteur_pass[0]
                            moyenne_type_passes[6][1] += array_hauteur_pass[1]
                            moyenne_type_passes[6][2] += array_hauteur_pass[2]
                            moyenne_nb_passes[6] += nb_pass
                            moyenne_long_par_la_passe[6] += pass_length
                            moyenne_nb_carry[6] += nb_carry
                            moyenne_long_par_le_carry[6] += dist_carry
                            moyenne_pied_fort[6][0] += pref_foot[0]
                            moyenne_pied_fort[6][1] += pref_foot[1]
                            for j in range(len(weak_foot)):
                                moyenne_pied_faible[6][j] += weak_foot[j]
                            for j in range(len(skills)):
                                moyenne_skills[6][j] += skills[j]
                            for j in range(len(IMC)):
                                moyenne_IMC[6][j] += IMC[j]
                            for j in range(len(passing)):
                                moyenne_passing[6][j] += passing[j]
                            for j in range(len(mentality_pos)):
                                moyenne_mentality_pos[6][j] += mentality_pos[j]
                            for j in range(len(mentality_vis)):
                                moyenne_mentality_vis[6][j] += mentality_vis[j]
                            for j in range(len(long_passing)):
                                moyenne_long_passing[6][j] += long_passing[j]
                            for j in range(len(pace)):
                                moyenne_pace[6][j] += pace[j]
                            for j in range(len(ball_control)):
                                moyenne_ball_control[6][j] += ball_control[j]
                            for j in range(len(short_passing)):
                                moyenne_short_passing[6][j] += short_passing[j]
                            for j in range(len(league)):
                                moyenne_league[6][j] += league[j]
                            for j in range(len(work_rate)):
                                moyenne_work_rate[6][j] += work_rate[j]
                            for j in range(len(physic)):
                                moyenne_physic[6][j] += physic[j]
                            for j in range(len(accel)):
                                moyenne_accel[6][j] += accel[j]
                            for j in range(len(strength)):
                                moyenne_strength[6][j] += strength[j]
                            for j in range(len(agility)):
                                moyenne_agility[6][j] += agility[j]
                            for j in range(len(aggression)):
                                moyenne_aggression[6][j] += aggression[j]
                            for j in range(len(def_central)):
                                center_back[6][j] += def_central[j]
                            for j in range(len(back_aile)):
                                back[6][j] += back_aile[j]
                            
                            if df_event['team'][index] == team[0]:
                                dico_compo[compo[0]][6] += 1
                            else : 
                                dico_compo[compo[1]][6] += 1
                
                else :
                    if df_event['type'][index_suiv] == "Pass" :
                        if df_event['pass_height'][index_suiv] == "Ground Pass":
                            array_hauteur_pass[0] += 1
                        if df_event['pass_height'][index_suiv] == "Low Pass":
                            array_hauteur_pass[1] += 1
                        if df_event['pass_height'][index_suiv] == "High Pass":
                            array_hauteur_pass[2] += 1
                        pass_length += df_event['pass_length'][index_suiv]
                        nb_pass += 1
                        duration += df_event['duration'][index_suiv]
                        liste_attente.append([df_event['location'][index_suiv][0], df_event['location'][index_suiv][1], df_event['pass_end_location'][index_suiv][0], df_event['pass_end_location'][index_suiv][1], df_event['type'][index_suiv], numero_relance])
                    if df_event['type'][index_suiv] == "Carry" :
                        nb_carry += 1
                        dist_carry += distance_entre_points(df_event['location'][index_suiv][0], df_event['location'][index_suiv][1], df_event['carry_end_location'][index_suiv][0], df_event['carry_end_location'][index_suiv][1])
                        duration += df_event['duration'][index_suiv]
                        liste_attente.append([df_event['location'][index_suiv][0], df_event['location'][index_suiv][1], df_event['carry_end_location'][index_suiv][0], df_event['carry_end_location'][index_suiv][1], df_event['type'][index_suiv], numero_relance])
                    for i in range(len(liste_attente)):
                        liste_echec.append(liste_attente[i])
                    compteur_relance[0] += 1
                    moyenne_duration[0] += duration 
                    moyenne_type_passes[0][0] += array_hauteur_pass[0]
                    moyenne_type_passes[0][1] += array_hauteur_pass[1]
                    moyenne_type_passes[0][2] += array_hauteur_pass[2]
                    moyenne_nb_passes[0] += nb_pass
                    moyenne_long_par_la_passe[0] += pass_length
                    moyenne_nb_carry[0] += nb_carry
                    moyenne_long_par_le_carry[0] += dist_carry
                    moyenne_pied_fort[0][0] += pref_foot[0]
                    moyenne_pied_fort[0][1] += pref_foot[1]
                    for j in range(len(weak_foot)):
                        moyenne_pied_faible[0][j] += weak_foot[j]
                    for j in range(len(skills)):
                        moyenne_skills[0][j] += skills[j]
                    for j in range(len(IMC)):
                        moyenne_IMC[0][j] += IMC[j]
                    for j in range(len(passing)):
                        moyenne_passing[0][j] += passing[j]
                    for j in range(len(mentality_pos)):
                        moyenne_mentality_pos[0][j] += mentality_pos[j]
                    for j in range(len(mentality_vis)):
                        moyenne_mentality_vis[0][j] += mentality_vis[j]
                    for j in range(len(long_passing)):
                        moyenne_long_passing[0][j] += long_passing[j]
                    for j in range(len(pace)):
                        moyenne_pace[0][j] += pace[j]
                    for j in range(len(ball_control)):
                        moyenne_ball_control[0][j] += ball_control[j]
                    for j in range(len(short_passing)):
                        moyenne_short_passing[0][j] += short_passing[j]
                    for j in range(len(league)):
                        moyenne_league[0][j] += league[j]
                    for j in range(len(work_rate)):
                        moyenne_work_rate[0][j] += work_rate[j]
                    for j in range(len(physic)):
                        moyenne_physic[0][j] += physic[j]
                    for j in range(len(accel)):
                        moyenne_accel[0][j] += accel[j]
                    for j in range(len(strength)):
                        moyenne_strength[0][j] += strength[j]
                    for j in range(len(agility)):
                        moyenne_agility[0][j] += agility[j]
                    for j in range(len(aggression)):
                        moyenne_aggression[0][j] += aggression[j]
                    for j in range(len(def_central)):
                        center_back[0][j] += def_central[j]
                    for j in range(len(back_aile)):
                        back[0][j] += back_aile[j]
                    
                    if df_event['team'][index] == team[0]:
                        dico_compo[compo[0]][0] += 1
                    else : 
                        dico_compo[compo[1]][0] += 1
                    
                numero_relance += 1
                
            if index_final != 0 :
                end_time = datetime.strptime(df_event['timestamp'][index_suiv], "%H:%M:%S.%f")
                
                time_difference = end_time - start_time
                time_difference_sec = time_difference.total_seconds()
                duration = time_difference_sec
                if df_event['location'][index_suiv][0] > 60 :
                    for i in range(len(liste_attente)):
                        liste_reussi.append(liste_attente[i])
                    packing = packing_relance(df_event['id'][first_index], df_event['id'][index_final], first_index, index_final, match_id)
                    if packing == 0 or packing == 1 :
                        moyenne_duration[1] += duration
                        compteur_relance[1] += 1
                        moyenne_type_passes[1][0] += array_hauteur_pass[0]
                        moyenne_type_passes[1][1] += array_hauteur_pass[1]
                        moyenne_type_passes[1][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[1] += nb_pass
                        moyenne_long_par_la_passe[1] += pass_length
                        moyenne_nb_carry[1] += nb_carry
                        moyenne_long_par_le_carry[1] += dist_carry
                        moyenne_pied_fort[1][0] += pref_foot[0]
                        moyenne_pied_fort[1][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[1][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[1][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[1][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[1][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[1][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[1][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[1][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[1][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[1][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[1][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[1][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[1][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[1][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[1][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[1][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[1][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[1][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[1][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[1][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][1] += 1
                        else : 
                            dico_compo[compo[1]][1] += 1
                    if packing == 2 or packing == 3 :
                        moyenne_duration[2] += duration
                        compteur_relance[2] += 1
                        moyenne_type_passes[2][0] += array_hauteur_pass[0]
                        moyenne_type_passes[2][1] += array_hauteur_pass[1]
                        moyenne_type_passes[2][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[2] += nb_pass
                        moyenne_long_par_la_passe[2] += pass_length
                        moyenne_nb_carry[2] += nb_carry
                        moyenne_long_par_le_carry[2] += dist_carry
                        moyenne_pied_fort[2][0] += pref_foot[0]
                        moyenne_pied_fort[2][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[2][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[2][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[2][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[2][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[2][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[2][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[2][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[2][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[2][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[2][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[2][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[2][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[2][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[2][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[2][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[2][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[2][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[2][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[2][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][2] += 1
                        else : 
                            dico_compo[compo[1]][2] += 1
                    if packing == 4 or packing == 5 :
                        moyenne_duration[3] += duration
                        compteur_relance[3] += 1
                        moyenne_type_passes[3][0] += array_hauteur_pass[0]
                        moyenne_type_passes[3][1] += array_hauteur_pass[1]
                        moyenne_type_passes[3][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[3] += nb_pass
                        moyenne_long_par_la_passe[3] += pass_length
                        moyenne_nb_carry[3] += nb_carry
                        moyenne_long_par_le_carry[3] += dist_carry
                        moyenne_pied_fort[3][0] += pref_foot[0]
                        moyenne_pied_fort[3][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[3][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[3][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[3][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[3][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[3][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[3][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[3][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[3][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[3][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[3][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[3][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[3][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[3][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[3][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[3][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[3][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[3][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[3][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[3][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][3] += 1
                        else : 
                            dico_compo[compo[1]][3] += 1
                    if packing == 6 or packing == 7 :
                        moyenne_duration[4] += duration
                        compteur_relance[4] += 1
                        moyenne_type_passes[4][0] += array_hauteur_pass[0]
                        moyenne_type_passes[4][1] += array_hauteur_pass[1]
                        moyenne_type_passes[4][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[4] += nb_pass
                        moyenne_long_par_la_passe[4] += pass_length
                        moyenne_nb_carry[4] += nb_carry
                        moyenne_long_par_le_carry[4] += dist_carry
                        moyenne_pied_fort[4][0] += pref_foot[0]
                        moyenne_pied_fort[4][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[4][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[4][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[4][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[4][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[4][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[4][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[4][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[4][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[4][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[4][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[4][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[4][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[4][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[4][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[4][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[4][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[4][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[4][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[4][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][4] += 1
                        else : 
                            dico_compo[compo[1]][4] += 1
                    if packing == 8 or packing == 9 :
                        moyenne_duration[5] += duration
                        compteur_relance[5] += 1
                        moyenne_type_passes[5][0] += array_hauteur_pass[0]
                        moyenne_type_passes[5][1] += array_hauteur_pass[1]
                        moyenne_type_passes[5][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[5] += nb_pass
                        moyenne_long_par_la_passe[5] += pass_length
                        moyenne_nb_carry[5] += nb_carry
                        moyenne_long_par_le_carry[5] += dist_carry
                        moyenne_pied_fort[5][0] += pref_foot[0]
                        moyenne_pied_fort[5][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[5][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[5][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[5][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[5][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[5][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[5][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[5][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[5][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[5][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[5][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[5][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[5][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[5][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[5][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[5][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[5][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[5][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[5][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[5][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][5] += 1
                        else : 
                            dico_compo[compo[1]][5] += 1
                    if packing == 10 :
                        moyenne_duration[6] += duration
                        compteur_relance[6] += 1
                        moyenne_type_passes[6][0] += array_hauteur_pass[0]
                        moyenne_type_passes[6][1] += array_hauteur_pass[1]
                        moyenne_type_passes[6][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[6] += nb_pass
                        moyenne_long_par_la_passe[6] += pass_length
                        moyenne_nb_carry[6] += nb_carry
                        moyenne_long_par_le_carry[6] += dist_carry
                        moyenne_pied_fort[6][0] += pref_foot[0]
                        moyenne_pied_fort[6][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[6][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[6][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[6][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[6][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[6][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[6][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[6][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[6][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[6][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[6][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[6][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[6][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[6][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[6][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[6][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[6][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[6][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[6][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[6][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][6] += 1
                        else : 
                            dico_compo[compo[1]][6] += 1

                else : 
                    for i in range(len(liste_attente)):
                        liste_echec.append(liste_attente[i])
                    compteur_relance[0] += 1
                    moyenne_duration[0] += duration
                    moyenne_type_passes[0][0] += array_hauteur_pass[0]
                    moyenne_type_passes[0][1] += array_hauteur_pass[1]
                    moyenne_type_passes[0][2] += array_hauteur_pass[2]
                    moyenne_nb_passes[0] += nb_pass
                    moyenne_long_par_la_passe[0] += pass_length
                    moyenne_nb_carry[0] += nb_carry
                    moyenne_long_par_le_carry[0] += dist_carry
                    moyenne_pied_fort[0][0] += pref_foot[0]
                    moyenne_pied_fort[0][1] += pref_foot[1]
                    for j in range(len(weak_foot)):
                        moyenne_pied_faible[0][j] += weak_foot[j]
                    for j in range(len(skills)):
                        moyenne_skills[0][j] += skills[j]
                    for j in range(len(IMC)):
                        moyenne_IMC[0][j] += IMC[j]
                    for j in range(len(passing)):
                        moyenne_passing[0][j] += passing[j]
                    for j in range(len(mentality_pos)):
                        moyenne_mentality_pos[0][j] += mentality_pos[j]
                    for j in range(len(mentality_vis)):
                        moyenne_mentality_vis[0][j] += mentality_vis[j]
                    for j in range(len(long_passing)):
                        moyenne_long_passing[0][j] += long_passing[j]
                    for j in range(len(pace)):
                        moyenne_pace[0][j] += pace[j]
                    for j in range(len(ball_control)):
                        moyenne_ball_control[0][j] += ball_control[j]
                    for j in range(len(short_passing)):
                        moyenne_short_passing[0][j] += short_passing[j]
                    for j in range(len(league)):
                        moyenne_league[0][j] += league[j]
                    for j in range(len(work_rate)):
                        moyenne_work_rate[0][j] += work_rate[j]
                    for j in range(len(physic)):
                        moyenne_physic[0][j] += physic[j]
                    for j in range(len(accel)):
                        moyenne_accel[0][j] += accel[j]
                    for j in range(len(strength)):
                        moyenne_strength[0][j] += strength[j]
                    for j in range(len(agility)):
                        moyenne_agility[0][j] += agility[j]
                    for j in range(len(aggression)):
                        moyenne_aggression[0][j] += aggression[j]
                    for j in range(len(def_central)):
                        center_back[0][j] += def_central[j]
                    for j in range(len(back_aile)):
                        back[0][j] += back_aile[j]
                    
                    if df_event['team'][index] == team[0]:
                        dico_compo[compo[0]][0] += 1
                    else : 
                        dico_compo[compo[1]][0] += 1
                
                numero_relance += 1
    
    for i in range(len(moyenne_duration)):
        if compteur_relance[i] != 0 :
            moyenne_duration[i] /= compteur_relance[i]
            moyenne_nb_passes[i] /= compteur_relance[i]
            moyenne_long_par_la_passe[i] /= compteur_relance[i]
            moyenne_nb_carry[i] /= compteur_relance[i]
            moyenne_long_par_le_carry[i] /= compteur_relance[i]
            
    return moyenne_duration, moyenne_type_passes, moyenne_nb_passes, moyenne_long_par_la_passe, moyenne_nb_carry, moyenne_long_par_le_carry, moyenne_pied_fort, moyenne_pied_faible, moyenne_skills, moyenne_IMC, moyenne_passing, moyenne_mentality_pos, moyenne_mentality_vis, moyenne_long_passing, moyenne_pace, moyenne_ball_control, moyenne_short_passing, moyenne_league, moyenne_work_rate, moyenne_physic, moyenne_accel, moyenne_strength, moyenne_agility, moyenne_aggression

for i in WC2022_games :
    mean_duration, mean_type_passes, mean_nb_passes, mean_long_par_la_passe, mean_nb_carry, mean_long_par_le_carry, mean_pied_fort, mean_pied_faible, mean_skills, mean_IMC, mean_passing, mean_mentality_pos, mean_mentality_vis, mean_long_passing, mean_pace, mean_ball_control, mean_short_passing, mean_league, mean_work_rate, mean_physic, mean_accel, mean_strength, mean_agility, mean_aggression = relance_rentrée(i)
    
    for j in range(len(mean_duration)):
        if mean_duration[j] != 0:
            compteur_duration[j] += 1
    for j in range(len(mean_pied_fort)):
        if mean_pied_fort[j][0] != 0 or mean_pied_fort[j][1] != 0 :
            compteur_pied_fort[j] += 1

    for j in range(len(duration)):
        duration[j] += mean_duration[j]
        nb_passes[j] += mean_nb_passes[j]
        long_par_la_passe[j] += mean_long_par_la_passe[j]
        nb_carry[j] += mean_nb_carry[j]
        long_par_le_carry[j] += mean_long_par_le_carry[j]
        for k in range(len(type_passes[0])):
            type_passes[j][k] += mean_type_passes[j][k]
        for k in range(len(pied_fort[0])):
            pied_fort[j][k] += mean_pied_fort[j][k]
        for k in range(len(pied_faible[0])):
            pied_faible[j][k] += mean_pied_faible[j][k]
        for k in range(len(skills[0])):
            skills[j][k] += mean_skills[j][k]
        for k in range(len(IMC[0])):
            IMC[j][k] += mean_IMC[j][k]
        for k in range(len(passing[0])):
            passing[j][k] += mean_passing[j][k]
        for k in range(len(mentality_pos[0])):
            mentality_pos[j][k] += mean_mentality_pos[j][k]
        for k in range(len(mentality_vis[0])):
            mentality_vis[j][k] += mean_mentality_vis[j][k]
        for k in range(len(long_passing[0])):
            long_passing[j][k] += mean_long_passing[j][k]
        for k in range(len(pace[0])):
            pace[j][k] += mean_pace[j][k]
        for k in range(len(ball_control[0])):
            ball_control[j][k] += mean_ball_control[j][k]
        for k in range(len(short_passing[0])):
            short_passing[j][k] += mean_short_passing[j][k]
        for k in range(len(league[0])):
            league[j][k] += mean_league[j][k]
        for k in range(len(work_rate[0])):
            work_rate[j][k] += mean_work_rate[j][k]
        for k in range(len(physic[0])):
            physic[j][k] += mean_physic[j][k]
        for k in range(len(accel[0])):
            accel[j][k] += mean_accel[j][k]
        for k in range(len(strength[0])):
            strength[j][k] += mean_strength[j][k]
        for k in range(len(agility[0])):
            agility[j][k] += mean_agility[j][k]
        for k in range(len(aggression[0])):
            aggression[j][k] += mean_aggression[j][k]


### FREE KICK ###
liste_reussi = []
liste_echec = []


""" On part d'un coup franc et on avance jusqu'à la fin de la phase de relance """
def relance_coup_franc(match_id):
    df_event = sb.events(match_id)
    df_lineup = sb.lineups(match_id)
    Type = df_event['type']
    types_exclus = ["Dispossessed", "Error", "Miscontrol", "Block", "Clearance", "Interception", "Pressure", "Foul Committed"]
    numero_relance = 1

    # Variable comptant le nombre de relances ayant un packing -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    compteur_relance = [0, 0, 0, 0, 0, 0, 0]

    # Variable donnant le nombre moyen de passes d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_nb_passes = [0, 0, 0, 0, 0, 0, 0]

    # Variable donnant la longueur moyenne parcourue par la passe d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_long_par_la_passe = [0, 0, 0, 0, 0, 0, 0]

    # Variable donnant le nombre moyen de chaque type de passes pour les relances ayant un packing -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_type_passes = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]] # [0] : Ground, [1] : Low, [2] : High
    
    # variable donnant la duree moyenne d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_duration = [0, 0, 0, 0, 0, 0, 0]
    
    # Variable donnant le nombre moyen de carries d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_nb_carry = [0, 0, 0, 0, 0, 0, 0]

    # Variable donnant la longueur moyenne parcourue par le carry d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_long_par_le_carry = [0, 0, 0, 0, 0, 0, 0]

    # Variable donnant la repartition du nombre de touches moyen par un droitier / gaucher d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_pied_fort = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]] # [0] : gaucher, [1] : droitier
    
    # Variable donnant la repartition (de 1/5 à 5/5) de la qualité de pied faible des joueurs d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_pied_faible = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]] # [0] : 1/5, [1] : 2/5, [2] : 3/5, [3] : 4/5, [4] : 5/5
    
    # Variable donnant la repartition (de 1/5 à 5/5) des skills des joueurs d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_skills = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]] # [0] : 1/5, [1] : 2/5, [2] : 3/5, [3] : 4/5, [4] : 5/5
    
    # Variable donnant la répartition (de 19-20 à 25-26) de l'IMC des joueurs d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_IMC = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]] # [0] : 19-20, [1] : 20-21, [2] : 21-22, [3] : 22-23, [4] : 23-24, [5] : 24-25, [6] : 25-26
    
    # Variable donnant la qualité de passe des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_passing = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la qualité de mentality positioning des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_mentality_pos = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la qualité de mentality positioning des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_mentality_vis = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la qualité de long passing des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_long_passing = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la qualité de vitesse des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_pace = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la qualité de controle des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_ball_control = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la qualité de passe courte des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_short_passing = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la repartition des championnats des joueurs d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_league = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]] # [0] : Ligue 1, [1] : Liga, [2] : Bundesliga, [3] : Serie A, [4] : Premier League

    # Variable donnant le work-rate des joueurs d'une relance -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_work_rate = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]] # Le premier est l'apport off et le deuxieme l'apport def : [0] : High/High, [1] : High/Medium, [2] : High/Low, [3] : Medium/High, [4] : Medium/Medium, [5] : Medium/Low, [6] : Low/High, [7] : Low/Medium, [8] : Low/Low

    # Variable donnant la qualité physique des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_physic = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90

    # Variable donnant la qualité d'acceleration des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_accel = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90
    
    # Variable donnant la force des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_strength = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90
    
    # Variable donnant l'agilité' des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_agility = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90
    
    # Variable donnant l'agressivité des joueurs participant aux relances -5; 0-1; 2-3; 4-5; 6-7; 8-9 et 10
    moyenne_aggression = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] # [0] : <50, [1] : 50-60, [2] : 60-70, [3] : 70-80, [4] : 80-90, [5] : >90
    
    index_suiv = 0
    nb_nonPress = 0
    nb_press = 0
    first_index = 0

    compo = []
    team = []

    for index, chaine in enumerate(Type):
        if chaine == "Starting XI" : 
            compo.append(df_event['tactics'][index]['formation'])
            team.append(df_event['team'][index])
        if chaine == "Pass" and df_event.loc[index, 'pass_type'] == "Free Kick" and df_event['location'][index][0] <= 30:
            array_hauteur_pass = [0,0,0]
            nb_pass = 0
            pass_length = 0
            dist_carry = 0
            nb_carry = 0
            duration = 0
            nb_press = 0
            nb_nonPress = 0
            first_index = index
            liste_attente = []

            pref_foot = [0, 0] # index 0 : gaucher, index 1 : droitier

            skills = [0, 0, 0, 0, 0] # 1/5, 2/5, 3/5, 4/5, 5/5

            weak_foot = [0, 0, 0, 0, 0] # 1/5, 2/5, 3/5, 4/5, 5/5

            IMC = [0, 0, 0, 0, 0, 0, 0] # 19-20, 20-21, 21-22, 22-23, 23-24, 24-25, 25-26

            passing = [0, 0, 0, 0, 0, 0]

            mentality_pos = [0, 0, 0, 0, 0, 0]

            mentality_vis = [0, 0, 0, 0, 0, 0]

            long_passing = [0, 0, 0, 0, 0, 0]

            pace = [0, 0, 0, 0, 0, 0]

            ball_control = [0, 0, 0, 0, 0, 0]

            short_passing = [0, 0, 0, 0, 0, 0]

            league = [0, 0, 0, 0, 0]

            work_rate = [0, 0, 0, 0, 0, 0, 0, 0, 0]

            physic = [0, 0, 0, 0, 0, 0]

            accel = [0, 0, 0, 0, 0, 0]

            strength = [0, 0, 0, 0, 0, 0]

            agility = [0, 0, 0, 0, 0, 0]

            aggression = [0, 0, 0, 0, 0, 0]

            def_central = [0, 0, 0, 0]

            back_aile = [0, 0, 0, 0]

            start_time = datetime.strptime(df_event['timestamp'][index], "%H:%M:%S.%f")
            time_suiv, index_suiv, periode_suiv = timestamp_suiv(index, match_id)
            

            if len(trouver_joueur(df_event['player'][index], 'players_22.csv')) != 0 :
                

                team = df_event['team'][index]
                player_id = df_event['player_id'][index]
                for i in range(len(df_lineup[team]['player_id'])):
                    if player_id == df_lineup[team]['player_id'][i] and len(df_lineup[team]['positions'][i]) != 0:
                        if df_lineup[team]['positions'][i][0]['position'] == "Left Center Back" :
                            pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                            if pied == "Left" :
                                def_central[0] += 1
                            elif pied == "Right" :
                                def_central[1] += 1
                        elif df_lineup[team]['positions'][i][0]['position'] == "Right Center Back" :
                            pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                            if pied == "Left" :
                                def_central[2] += 1
                            elif pied == "Right" :
                                def_central[3] += 1
                        elif df_lineup[team]['positions'][i][0]['position'] == "Left Back" :
                            pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                            if pied == "Left" :
                                back_aile[0] += 1
                            elif pied == "Right" :
                                back_aile[1] += 1
                        elif df_lineup[team]['positions'][i][0]['position'] == "Right Back" :
                            pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                            if pied == "Left" :
                                back_aile[2] += 1
                            elif pied == "Right" :
                                back_aile[3] += 1


                if data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == "Left" :
                    pref_foot[0] += 1
                else : 
                    pref_foot[1] += 1

                if data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                    skills[0] += 1
                elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                    skills[1] += 1
                elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                    skills[2] += 1
                elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                    skills[3] += 1
                else :
                    skills[4] += 1

                player_IMC = calcul_imc(data_players['weight_kg'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]], data_players['height_cm'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]])

                if player_IMC <= 20 :
                    IMC[0] += 1
                elif 20 < player_IMC <= 21 :
                    IMC[1] += 1
                elif 21 < player_IMC <= 22 :
                    IMC[2] += 1
                elif 22 < player_IMC <= 23 :
                    IMC[3] += 1
                elif 23 < player_IMC <= 24 :
                    IMC[4] += 1
                elif 24 < player_IMC <= 25 :
                    IMC[5] += 1
                else :
                    IMC[6] += 1

                if data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                    weak_foot[0] += 1
                elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                    weak_foot[1] += 1
                elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                    weak_foot[2] += 1
                elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                    weak_foot[3] += 1
                else :
                    weak_foot[4] += 1

                player_passing = data_players['passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                if data_players['club_position'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] != "GK" :
                    if player_passing <= 50:
                        passing[0] += 1
                    elif 50 < player_passing <= 60:
                        passing[1] += 1
                    elif 60 < player_passing <= 70:
                        passing[2] += 1
                    elif 70 < player_passing <= 80:
                        passing[3] += 1
                    elif 80 < player_passing <= 90:
                        passing[4] += 1
                    else : 
                        passing[5] += 1
                    
                player_mentality_pos = data_players['mentality_positioning'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                if player_mentality_pos <= 50:
                    mentality_pos[0] += 1
                elif 50 < player_mentality_pos <= 60:
                    mentality_pos[1] += 1
                elif 60 < player_mentality_pos <= 70:
                    mentality_pos[2] += 1
                elif 70 < player_mentality_pos <= 80:
                    mentality_pos[3] += 1
                elif 80 < player_mentality_pos <= 90:
                    mentality_pos[4] += 1
                else : 
                    mentality_pos[5] += 1

                player_mentality_vis = data_players['mentality_vision'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                if player_mentality_vis <= 50:
                    mentality_vis[0] += 1
                elif 50 < player_mentality_vis <= 60:
                    mentality_vis[1] += 1
                elif 60 < player_mentality_vis <= 70:
                    mentality_vis[2] += 1
                elif 70 < player_mentality_vis <= 80:
                    mentality_vis[3] += 1
                elif 80 < player_mentality_vis <= 90:
                    mentality_vis[4] += 1
                else : 
                    mentality_vis[5] += 1

                player_long_passing = data_players['skill_long_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                if player_long_passing <= 50:
                    long_passing[0] += 1
                elif 50 < player_long_passing <= 60:
                    long_passing[1] += 1
                elif 60 < player_long_passing <= 70:
                    long_passing[2] += 1
                elif 70 < player_long_passing <= 80:
                    long_passing[3] += 1
                elif 80 < player_long_passing <= 90:
                    long_passing[4] += 1
                else : 
                    long_passing[5] += 1
                
                player_pace = data_players['pace'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                if player_pace <= 50:
                    pace[0] += 1
                elif 50 < player_pace <= 60:
                    pace[1] += 1
                elif 60 < player_pace <= 70:
                    pace[2] += 1
                elif 70 < player_pace <= 80:
                    pace[3] += 1
                elif 80 < player_pace <= 90:
                    pace[4] += 1
                else : 
                    pace[5] += 1

                player_ball_control = data_players['skill_ball_control'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_ball_control <= 50:
                    ball_control[0] += 1
                elif 50 < player_ball_control <= 60:
                    ball_control[1] += 1
                elif 60 < player_ball_control <= 70:
                    ball_control[2] += 1
                elif 70 < player_ball_control <= 80:
                    ball_control[3] += 1
                elif 80 < player_ball_control <= 90:
                    ball_control[4] += 1
                else : 
                    ball_control[5] += 1

                player_short_passing = data_players['attacking_short_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_short_passing <= 50:
                    short_passing[0] += 1
                elif 50 < player_short_passing <= 60:
                    short_passing[1] += 1
                elif 60 < player_short_passing <= 70:
                    short_passing[2] += 1
                elif 70 < player_short_passing <= 80:
                    short_passing[3] += 1
                elif 80 < player_short_passing <= 90:
                    short_passing[4] += 1
                else : 
                    short_passing[5] += 1
                
                player_league = data_players['league_name'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_league == "French Ligue 1":
                    league[0] += 1
                elif player_league == "Spain Primera Division":
                    league[1] += 1
                elif player_league == "German 1. Bundesliga":
                    league[2] += 1
                elif player_league == "Italian Serie A":
                    league[3] += 1
                elif player_league == "English Premier League":
                    league[4] += 1

                player_work_rate = data_players['work_rate'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_work_rate == "High/High":
                    work_rate[0] += 1
                elif player_work_rate == "High/Medium":
                    work_rate[1] += 1
                elif player_work_rate == "High/Low":
                    work_rate[2] += 1
                elif player_work_rate == "Medium/High":
                    work_rate[3] += 1
                elif player_work_rate == "Medium/Medium":
                    work_rate[4] += 1
                elif player_work_rate == "Medium/Low":
                    work_rate[5] += 1
                elif player_work_rate == "Low/High":
                    work_rate[6] += 1
                elif player_work_rate == "Low/Medium":
                    work_rate[7] += 1
                elif player_work_rate == "Low/Low": 
                    work_rate[8] += 1

                player_physic = data_players['physic'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_physic <= 50:
                    physic[0] += 1
                elif 50 < player_physic <= 60:
                    physic[1] += 1
                elif 60 < player_physic <= 70:
                    physic[2] += 1
                elif 70 < player_physic <= 80:
                    physic[3] += 1
                elif 80 < player_physic <= 90:
                    physic[4] += 1
                else : 
                    physic[5] += 1

                player_accel = data_players['movement_acceleration'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_accel <= 50:
                    accel[0] += 1
                elif 50 < player_accel <= 60:
                    accel[1] += 1
                elif 60 < player_accel <= 70:
                    accel[2] += 1
                elif 70 < player_accel <= 80:
                    accel[3] += 1
                elif 80 < player_accel <= 90:
                    accel[4] += 1
                else : 
                    accel[5] += 1

                player_strength = data_players['power_strength'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_strength <= 50:
                    strength[0] += 1
                elif 50 < player_strength <= 60:
                    strength[1] += 1
                elif 60 < player_strength <= 70:
                    strength[2] += 1
                elif 70 < player_strength <= 80:
                    strength[3] += 1
                elif 80 < player_strength <= 90:
                    strength[4] += 1
                else : 
                    strength[5] += 1

                player_agility = data_players['movement_agility'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_agility <= 50:
                    agility[0] += 1
                elif 50 < player_agility <= 60:
                    agility[1] += 1
                elif 60 < player_agility <= 70:
                    agility[2] += 1
                elif 70 < player_agility <= 80:
                    agility[3] += 1
                elif 80 < player_agility <= 90:
                    agility[4] += 1
                else : 
                    agility[5] += 1

                player_aggression = data_players['mentality_aggression'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                if player_aggression <= 50:
                    aggression[0] += 1
                elif 50 < player_aggression <= 60:
                    aggression[1] += 1
                elif 60 < player_aggression <= 70:
                    aggression[2] += 1
                elif 70 < player_aggression <= 80:
                    aggression[3] += 1
                elif 80 < player_aggression <= 90:
                    aggression[4] += 1
                else : 
                    aggression[5] += 1
        
            index_final = 0
            liste_attente.append([df_event['location'][index][0], df_event['location'][index][1], df_event['pass_end_location'][index][0], df_event['pass_end_location'][index][1], df_event['type'][index], numero_relance])
            
            while (df_event['location'][index_suiv][0] < 60 and df_event.loc[index_suiv, 'type'] not in types_exclus and df_event['pass_outcome'][index_suiv] != "Incomplete" and df_event['pass_outcome'][index_suiv] != "Out" and df_event['pass_outcome'][index_suiv] != "Pass Offside" and df_event['pass_outcome'][index_suiv] != "Unknown" and df_event['dribble_outcome'][index_suiv] != "Incomplete"):
                index = index_suiv
                
                if len(trouver_joueur(df_event['player'][index], 'players_22.csv')) != 0 :
                    

                    team = df_event['team'][index]
                    player_id = df_event['player_id'][index]
                    for i in range(len(df_lineup[team]['player_id'])):
                        if player_id == df_lineup[team]['player_id'][i] and len(df_lineup[team]['positions'][i]) != 0:
                            if df_lineup[team]['positions'][i][0]['position'] == "Left Center Back" :
                                pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                if pied == "Left" :
                                    def_central[0] += 1
                                elif pied == "Right" :
                                    def_central[1] += 1
                            elif df_lineup[team]['positions'][i][0]['position'] == "Right Center Back" :
                                pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                if pied == "Left" :
                                    def_central[2] += 1
                                elif pied == "Right" :
                                    def_central[3] += 1
                            elif df_lineup[team]['positions'][i][0]['position'] == "Left Back" :
                                pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                if pied == "Left" :
                                    back_aile[0] += 1
                                elif pied == "Right" :
                                    back_aile[1] += 1
                            elif df_lineup[team]['positions'][i][0]['position'] == "Right Back" :
                                pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                if pied == "Left" :
                                    back_aile[2] += 1
                                elif pied == "Right" :
                                    back_aile[3] += 1


                    if data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == "Left" :
                        pref_foot[0] += 1
                    else : 
                        pref_foot[1] += 1

                    if data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                        skills[0] += 1
                    elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                        skills[1] += 1
                    elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                        skills[2] += 1
                    elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                        skills[3] += 1
                    else :
                        skills[4] += 1

                    player_IMC = calcul_imc(data_players['weight_kg'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]], data_players['height_cm'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]])
                    if player_IMC <= 20 :
                        IMC[0] += 1
                    elif 20 < player_IMC <= 21 :
                        IMC[1] += 1
                    elif 21 < player_IMC <= 22 :
                        IMC[2] += 1
                    elif 22 < player_IMC <= 23 :
                        IMC[3] += 1
                    elif 23 < player_IMC <= 24 :
                        IMC[4] += 1
                    elif 24 < player_IMC <= 25 :
                        IMC[5] += 1
                    else :
                        IMC[6] += 1
                    
                    if data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                        weak_foot[0] += 1
                    elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                        weak_foot[1] += 1
                    elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                        weak_foot[2] += 1                        
                    elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                        weak_foot[3] += 1
                    else :
                        weak_foot[4] += 1

                    player_passing = data_players['passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if data_players['club_position'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] != "GK" :
                        if player_passing <= 50:
                            passing[0] += 1
                        elif 50 < player_passing <= 60:
                            passing[1] += 1
                        elif 60 < player_passing <= 70:
                            passing[2] += 1
                        elif 70 < player_passing <= 80:
                            passing[3] += 1
                        elif 80 < player_passing <= 90:
                            passing[4] += 1
                        else : 
                            passing[5] += 1

                    player_mentality_pos = data_players['mentality_positioning'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if player_mentality_pos <= 50:
                        mentality_pos[0] += 1
                    elif 50 < player_mentality_pos <= 60:
                        mentality_pos[1] += 1
                    elif 60 < player_mentality_pos <= 70:
                        mentality_pos[2] += 1
                    elif 70 < player_mentality_pos <= 80:
                        mentality_pos[3] += 1
                    elif 80 < player_mentality_pos <= 90:
                        mentality_pos[4] += 1
                    else : 
                        mentality_pos[5] += 1

                    player_mentality_vis = data_players['mentality_vision'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if player_mentality_vis <= 50:
                        mentality_vis[0] += 1
                    elif 50 < player_mentality_vis <= 60:
                        mentality_vis[1] += 1
                    elif 60 < player_mentality_vis <= 70:
                        mentality_vis[2] += 1
                    elif 70 < player_mentality_vis <= 80:
                        mentality_vis[3] += 1
                    elif 80 < player_mentality_vis <= 90:
                        mentality_vis[4] += 1
                    else : 
                        mentality_vis[5] += 1

                    player_long_passing = data_players['skill_long_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if player_long_passing <= 50:
                        long_passing[0] += 1
                    elif 50 < player_long_passing <= 60:
                        long_passing[1] += 1
                    elif 60 < player_long_passing <= 70:
                        long_passing[2] += 1
                    elif 70 < player_long_passing <= 80:
                        long_passing[3] += 1
                    elif 80 < player_long_passing <= 90:
                        long_passing[4] += 1
                    else : 
                        long_passing[5] += 1
                    
                    player_pace = data_players['pace'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if player_pace <= 50:
                        pace[0] += 1
                    elif 50 < player_pace <= 60:
                        pace[1] += 1
                    elif 60 < player_pace <= 70:
                        pace[2] += 1
                    elif 70 < player_pace <= 80:
                        pace[3] += 1
                    elif 80 < player_pace <= 90:
                        pace[4] += 1
                    else : 
                        pace[5] += 1

                    player_ball_control = data_players['skill_ball_control'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_ball_control <= 50:
                        ball_control[0] += 1
                    elif 50 < player_ball_control <= 60:
                        ball_control[1] += 1
                    elif 60 < player_ball_control <= 70:
                        ball_control[2] += 1
                    elif 70 < player_ball_control <= 80:
                        ball_control[3] += 1
                    elif 80 < player_ball_control <= 90:
                        ball_control[4] += 1
                    else : 
                        ball_control[5] += 1

                    player_short_passing = data_players['attacking_short_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_short_passing <= 50:
                        short_passing[0] += 1
                    elif 50 < player_short_passing <= 60:
                        short_passing[1] += 1
                    elif 60 < player_short_passing <= 70:
                        short_passing[2] += 1
                    elif 70 < player_short_passing <= 80:
                        short_passing[3] += 1
                    elif 80 < player_short_passing <= 90:
                        short_passing[4] += 1
                    else : 
                        short_passing[5] += 1

                    player_league = data_players['league_name'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_league == "French Ligue 1":
                        league[0] += 1
                    elif player_league == "Spain Primera Division":
                        league[1] += 1
                    elif player_league == "German 1. Bundesliga":
                        league[2] += 1
                    elif player_league == "Italian Serie A":
                        league[3] += 1
                    elif player_league == "English Premier League":
                        league[4] += 1

                    player_work_rate = data_players['work_rate'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_work_rate == "High/High":
                        work_rate[0] += 1
                    elif player_work_rate == "High/Medium":
                        work_rate[1] += 1
                    elif player_work_rate == "High/Low":
                        work_rate[2] += 1
                    elif player_work_rate == "Medium/High":
                        work_rate[3] += 1
                    elif player_work_rate == "Medium/Medium":
                        work_rate[4] += 1
                    elif player_work_rate == "Medium/Low":
                        work_rate[5] += 1
                    elif player_work_rate == "Low/High":
                        work_rate[6] += 1
                    elif player_work_rate == "Low/Medium":
                        work_rate[7] += 1
                    elif player_work_rate == "Low/Low": 
                        work_rate[8] += 1

                    player_physic = data_players['physic'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_physic <= 50:
                        physic[0] += 1
                    elif 50 < player_physic <= 60:
                        physic[1] += 1
                    elif 60 < player_physic <= 70:
                        physic[2] += 1
                    elif 70 < player_physic <= 80:
                        physic[3] += 1
                    elif 80 < player_physic <= 90:
                        physic[4] += 1
                    else : 
                        physic[5] += 1

                    player_accel = data_players['movement_acceleration'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_accel <= 50:
                        accel[0] += 1
                    elif 50 < player_accel <= 60:
                        accel[1] += 1
                    elif 60 < player_accel <= 70:
                        accel[2] += 1
                    elif 70 < player_accel <= 80:
                        accel[3] += 1
                    elif 80 < player_accel <= 90:
                        accel[4] += 1
                    else : 
                        accel[5] += 1

                    player_strength = data_players['power_strength'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_strength <= 50:
                        strength[0] += 1
                    elif 50 < player_strength <= 60:
                        strength[1] += 1
                    elif 60 < player_strength <= 70:
                        strength[2] += 1
                    elif 70 < player_strength <= 80:
                        strength[3] += 1
                    elif 80 < player_strength <= 90:
                        strength[4] += 1
                    else : 
                        strength[5] += 1

                    player_agility = data_players['movement_agility'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_agility <= 50:
                        agility[0] += 1
                    elif 50 < player_agility <= 60:
                        agility[1] += 1
                    elif 60 < player_agility <= 70:
                        agility[2] += 1
                    elif 70 < player_agility <= 80:
                        agility[3] += 1
                    elif 80 < player_agility <= 90:
                        agility[4] += 1
                    else : 
                        agility[5] += 1

                    player_aggression = data_players['mentality_aggression'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_aggression <= 50:
                        aggression[0] += 1
                    elif 50 < player_aggression <= 60:
                        aggression[1] += 1
                    elif 60 < player_aggression <= 70:
                        aggression[2] += 1
                    elif 70 < player_aggression <= 80:
                        aggression[3] += 1
                    elif 80 < player_aggression <= 90:
                        aggression[4] += 1
                    else : 
                        aggression[5] += 1
        
            
                if df_event['under_pressure'][index] == True: 
                    nb_press += 1
                else : 
                    nb_nonPress += 1
                    
            
                if df_event['type'][index] == "Pass":
                    if df_event['pass_height'][index] == "Ground Pass":
                        array_hauteur_pass[0] += 1
                    if df_event['pass_height'][index] == "Low Pass":
                        array_hauteur_pass[1] += 1
                    if df_event['pass_height'][index] == "High Pass":
                        array_hauteur_pass[2] += 1
                    pass_length += df_event['pass_length'][index]
                    nb_pass += 1
                    duration += df_event['duration'][index]
                    liste_attente.append([df_event['location'][index][0], df_event['location'][index][1], df_event['pass_end_location'][index][0], df_event['pass_end_location'][index][1], df_event['type'][index], numero_relance])
                if df_event['type'][index] == "Carry":
                    nb_carry += 1
                    dist_carry += distance_entre_points(df_event['location'][index][0], df_event['location'][index][1], df_event['carry_end_location'][index][0], df_event['carry_end_location'][index][1])
                    liste_attente.append([df_event['location'][index][0], df_event['location'][index][1], df_event['carry_end_location'][index][0], df_event['carry_end_location'][index][1], df_event['type'][index], numero_relance])
                    duration += df_event['duration'][index]
                time_suiv, index_suiv, periode_suiv = timestamp_suiv(index, match_id)
                if not isinstance(df_event['location'][index_suiv], list): 
                    index_suiv = index
                    break
                index_final = index_suiv

            if index_final == 0 :
                if len(trouver_joueur(df_event['player'][index], 'players_22.csv')) != 0 :
                    

                    team = df_event['team'][index]
                    player_id = df_event['player_id'][index]
                    for i in range(len(df_lineup[team]['player_id'])):
                        if player_id == df_lineup[team]['player_id'][i] and len(df_lineup[team]['positions'][i]) != 0:
                            if df_lineup[team]['positions'][i][0]['position'] == "Left Center Back" :
                                pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                if pied == "Left" :
                                    def_central[0] += 1
                                elif pied == "Right" :
                                    def_central[1] += 1
                            elif df_lineup[team]['positions'][i][0]['position'] == "Right Center Back" :
                                pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                if pied == "Left" :
                                    def_central[2] += 1
                                elif pied == "Right" :
                                    def_central[3] += 1
                            elif df_lineup[team]['positions'][i][0]['position'] == "Left Back" :
                                pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                if pied == "Left" :
                                    back_aile[0] += 1
                                elif pied == "Right" :
                                    back_aile[1] += 1
                            elif df_lineup[team]['positions'][i][0]['position'] == "Right Back" :
                                pied = data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                                if pied == "Left" :
                                    back_aile[2] += 1
                                elif pied == "Right" :
                                    back_aile[3] += 1


                    if data_players['preferred_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == "Left" :
                        pref_foot[0] += 1
                    else : 
                        pref_foot[1] += 1

                    if data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                        skills[0] += 1
                    elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                        skills[1] += 1
                    elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                        skills[2] += 1
                    elif data_players['skill_moves'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                        skills[3] += 1
                    else :
                        skills[4] += 1

                    player_IMC = calcul_imc(data_players['weight_kg'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]], data_players['height_cm'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]])

                    if player_IMC <= 20 :
                        IMC[0] += 1
                    elif 20 < player_IMC <= 21 :
                        IMC[1] += 1
                    elif 21 < player_IMC <= 22 :
                        IMC[2] += 1
                    elif 22 < player_IMC <= 23 :
                        IMC[3] += 1
                    elif 23 < player_IMC <= 24 :
                        IMC[4] += 1
                    elif 24 < player_IMC <= 25 :
                        IMC[5] += 1
                    else :
                        IMC[6] += 1

                    if data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 1 :
                        weak_foot[0] += 1
                    elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 2 :
                        weak_foot[1] += 1
                    elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 3 :
                        weak_foot[2] += 1
                    elif data_players['weak_foot'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] == 4 :
                        weak_foot[3] += 1
                    else :
                        weak_foot[4] += 1

                    player_passing = data_players['passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if data_players['club_position'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]] != "GK" :
                        if player_passing <= 50:
                            passing[0] += 1
                        elif 50 < player_passing <= 60:
                            passing[1] += 1
                        elif 60 < player_passing <= 70:
                            passing[2] += 1
                        elif 70 < player_passing <= 80:
                            passing[3] += 1
                        elif 80 < player_passing <= 90:
                            passing[4] += 1
                        else : 
                            passing[5] += 1
                    
                    player_mentality_pos = data_players['mentality_positioning'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if player_mentality_pos <= 50:
                        mentality_pos[0] += 1
                    elif 50 < player_mentality_pos <= 60:
                        mentality_pos[1] += 1
                    elif 60 < player_mentality_pos <= 70:
                        mentality_pos[2] += 1
                    elif 70 < player_mentality_pos <= 80:
                        mentality_pos[3] += 1
                    elif 80 < player_mentality_pos <= 90:
                        mentality_pos[4] += 1
                    else : 
                        mentality_pos[5] += 1

                    player_mentality_vis = data_players['mentality_vision'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if player_mentality_vis <= 50:
                        mentality_vis[0] += 1
                    elif 50 < player_mentality_vis <= 60:
                        mentality_vis[1] += 1
                    elif 60 < player_mentality_vis <= 70:
                        mentality_vis[2] += 1
                    elif 70 < player_mentality_vis <= 80:
                        mentality_vis[3] += 1
                    elif 80 < player_mentality_vis <= 90:
                        mentality_vis[4] += 1
                    else : 
                        mentality_vis[5] += 1

                    player_long_passing = data_players['skill_long_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if player_long_passing <= 50:
                        long_passing[0] += 1
                    elif 50 < player_long_passing <= 60:
                        long_passing[1] += 1
                    elif 60 < player_long_passing <= 70:
                        long_passing[2] += 1
                    elif 70 < player_long_passing <= 80:
                        long_passing[3] += 1
                    elif 80 < player_long_passing <= 90:
                        long_passing[4] += 1
                    else : 
                        long_passing[5] += 1
                    
                    player_pace = data_players['pace'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]
                
                    if player_pace <= 50:
                        pace[0] += 1
                    elif 50 < player_pace <= 60:
                        pace[1] += 1
                    elif 60 < player_pace <= 70:
                        pace[2] += 1
                    elif 70 < player_pace <= 80:
                        pace[3] += 1
                    elif 80 < player_pace <= 90:
                        pace[4] += 1
                    else : 
                        pace[5] += 1

                    player_ball_control = data_players['skill_ball_control'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_ball_control <= 50:
                        ball_control[0] += 1
                    elif 50 < player_ball_control <= 60:
                        ball_control[1] += 1
                    elif 60 < player_ball_control <= 70:
                        ball_control[2] += 1
                    elif 70 < player_ball_control <= 80:
                        ball_control[3] += 1
                    elif 80 < player_ball_control <= 90:
                        ball_control[4] += 1
                    else : 
                        ball_control[5] += 1

                    player_short_passing = data_players['attacking_short_passing'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_short_passing <= 50:
                        short_passing[0] += 1
                    elif 50 < player_short_passing <= 60:
                        short_passing[1] += 1
                    elif 60 < player_short_passing <= 70:
                        short_passing[2] += 1
                    elif 70 < player_short_passing <= 80:
                        short_passing[3] += 1
                    elif 80 < player_short_passing <= 90:
                        short_passing[4] += 1
                    else : 
                        short_passing[5] += 1

                    player_league = data_players['league_name'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_league == "French Ligue 1":
                        league[0] += 1
                    elif player_league == "Spain Primera Division":
                        league[1] += 1
                    elif player_league == "German 1. Bundesliga":
                        league[2] += 1
                    elif player_league == "Italian Serie A":
                        league[3] += 1
                    elif player_league == "English Premier League":
                        league[4] += 1

                    player_work_rate = data_players['work_rate'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_work_rate == "High/High":
                        work_rate[0] += 1
                    elif player_work_rate == "High/Medium":
                        work_rate[1] += 1
                    elif player_work_rate == "High/Low":
                        work_rate[2] += 1
                    elif player_work_rate == "Medium/High":
                        work_rate[3] += 1
                    elif player_work_rate == "Medium/Medium":
                        work_rate[4] += 1
                    elif player_work_rate == "Medium/Low":
                        work_rate[5] += 1
                    elif player_work_rate == "Low/High":
                        work_rate[6] += 1
                    elif player_work_rate == "Low/Medium":
                        work_rate[7] += 1
                    elif player_work_rate == "Low/Low": 
                        work_rate[8] += 1

                    player_physic = data_players['physic'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_physic <= 50:
                        physic[0] += 1
                    elif 50 < player_physic <= 60:
                        physic[1] += 1
                    elif 60 < player_physic <= 70:
                        physic[2] += 1
                    elif 70 < player_physic <= 80:
                        physic[3] += 1
                    elif 80 < player_physic <= 90:
                        physic[4] += 1
                    else : 
                        physic[5] += 1

                    player_accel = data_players['movement_acceleration'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_accel <= 50:
                        accel[0] += 1
                    elif 50 < player_accel <= 60:
                        accel[1] += 1
                    elif 60 < player_accel <= 70:
                        accel[2] += 1
                    elif 70 < player_accel <= 80:
                        accel[3] += 1
                    elif 80 < player_accel <= 90:
                        accel[4] += 1
                    else : 
                        accel[5] += 1

                    player_strength = data_players['power_strength'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_strength <= 50:
                        strength[0] += 1
                    elif 50 < player_strength <= 60:
                        strength[1] += 1
                    elif 60 < player_strength <= 70:
                        strength[2] += 1
                    elif 70 < player_strength <= 80:
                        strength[3] += 1
                    elif 80 < player_strength <= 90:
                        strength[4] += 1
                    else : 
                        strength[5] += 1

                    player_agility = data_players['movement_agility'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_agility <= 50:
                        agility[0] += 1
                    elif 50 < player_agility <= 60:
                        agility[1] += 1
                    elif 60 < player_agility <= 70:
                        agility[2] += 1
                    elif 70 < player_agility <= 80:
                        agility[3] += 1
                    elif 80 < player_agility <= 90:
                        agility[4] += 1
                    else : 
                        agility[5] += 1

                    player_aggression = data_players['mentality_aggression'][trouver_joueur(df_event['player'][index], 'players_22.csv')[0]]

                    if player_aggression <= 50:
                        aggression[0] += 1
                    elif 50 < player_aggression <= 60:
                        aggression[1] += 1
                    elif 60 < player_aggression <= 70:
                        aggression[2] += 1
                    elif 70 < player_aggression <= 80:
                        aggression[3] += 1
                    elif 80 < player_aggression <= 90:
                        aggression[4] += 1
                    else : 
                        aggression[5] += 1
                
                if df_event['under_pressure'][index] == True: 
                    nb_press += 1
                else : 
                    nb_nonPress += 1

                if df_event['type'][index] == "Pass":
                    if df_event['pass_height'][index] == "Ground Pass":
                        array_hauteur_pass[0] += 1
                    if df_event['pass_height'][index] == "Low Pass":
                        array_hauteur_pass[1] += 1
                    if df_event['pass_height'][index] == "High Pass":
                        array_hauteur_pass[2] += 1
                    pass_length += df_event['pass_length'][index]
                    nb_pass += 1
                    duration += df_event['duration'][index]
                    liste_attente.append([df_event['location'][index][0], df_event['location'][index][1], df_event['pass_end_location'][index][0], df_event['pass_end_location'][index][1], df_event['type'][index], numero_relance])
                if df_event['type'][index] == "Carry":
                    nb_carry += 1
                    dist_carry += distance_entre_points(df_event['location'][index][0], df_event['location'][index][1], df_event['carry_end_location'][index][0], df_event['carry_end_location'][index][1])   
                    duration += df_event['duration'][index]
                    liste_attente.append([df_event['location'][index][0], df_event['location'][index][1], df_event['carry_end_location'][index][0], df_event['carry_end_location'][index][1], df_event['type'][index], numero_relance])
                if df_event['type'][index_suiv] == "Ball Receipt*" and df_event['ball_receipt_outcome'][index_suiv] != "Incomplete":
                    if df_event['pass_end_location'][index][0] > 60 :
                        for i in range(len(liste_attente)):
                            liste_reussi.append(liste_attente[i])
                        packing = packing_one_event(df_event['id'][index], index, match_id)
                        if packing == 0 or packing == 1 :
                            moyenne_duration[1] += duration
                            compteur_relance[1] += 1
                            moyenne_type_passes[1][0] += array_hauteur_pass[0]
                            moyenne_type_passes[1][1] += array_hauteur_pass[1]
                            moyenne_type_passes[1][2] += array_hauteur_pass[2]
                            moyenne_nb_passes[1] += nb_pass
                            moyenne_long_par_la_passe[1] += pass_length
                            moyenne_nb_carry[1] += nb_carry
                            moyenne_long_par_le_carry[1] += dist_carry
                            moyenne_pied_fort[1][0] += pref_foot[0]
                            moyenne_pied_fort[1][1] += pref_foot[1]
                            for j in range(len(weak_foot)):
                                moyenne_pied_faible[1][j] += weak_foot[j]
                            for j in range(len(skills)):
                                moyenne_skills[1][j] += skills[j]
                            for j in range(len(IMC)):
                                moyenne_IMC[1][j] += IMC[j]
                            for j in range(len(passing)):
                                moyenne_passing[1][j] += passing[j]
                            for j in range(len(mentality_pos)):
                                moyenne_mentality_pos[1][j] += mentality_pos[j]
                            for j in range(len(mentality_vis)):
                                moyenne_mentality_vis[1][j] += mentality_vis[j]
                            for j in range(len(long_passing)):
                                moyenne_long_passing[1][j] += long_passing[j]
                            for j in range(len(pace)):
                                moyenne_pace[1][j] += pace[j]
                            for j in range(len(ball_control)):
                                moyenne_ball_control[1][j] += ball_control[j]
                            for j in range(len(short_passing)):
                                moyenne_short_passing[1][j] += short_passing[j]
                            for j in range(len(league)):
                                moyenne_league[1][j] += league[j]
                            for j in range(len(work_rate)):
                                moyenne_work_rate[1][j] += work_rate[j]
                            for j in range(len(physic)):
                                moyenne_physic[1][j] += physic[j]
                            for j in range(len(accel)):
                                moyenne_accel[1][j] += accel[j]
                            for j in range(len(strength)):
                                moyenne_strength[1][j] += strength[j]
                            for j in range(len(agility)):
                                moyenne_agility[1][j] += agility[j]
                            for j in range(len(aggression)):
                                moyenne_aggression[1][j] += aggression[j]
                            for j in range(len(def_central)):
                                center_back[1][j] += def_central[j]
                            for j in range(len(back_aile)):
                                back[1][j] += back_aile[j]
                            
                            if df_event['team'][index] == team[0]:
                                dico_compo[compo[0]][1] += 1
                            else : 
                                dico_compo[compo[1]][1] += 1
                        if packing == 2 or packing == 3 :
                            moyenne_duration[2] += duration
                            compteur_relance[2] += 1
                            moyenne_type_passes[2][0] += array_hauteur_pass[0]
                            moyenne_type_passes[2][1] += array_hauteur_pass[1]
                            moyenne_type_passes[2][2] += array_hauteur_pass[2]
                            moyenne_nb_passes[2] += nb_pass
                            moyenne_long_par_la_passe[2] += pass_length
                            moyenne_nb_carry[2] += nb_carry
                            moyenne_long_par_le_carry[2] += dist_carry
                            moyenne_pied_fort[2][0] += pref_foot[0]
                            moyenne_pied_fort[2][1] += pref_foot[1]
                            for j in range(len(weak_foot)):
                                moyenne_pied_faible[2][j] += weak_foot[j]
                            for j in range(len(skills)):
                                moyenne_skills[2][j] += skills[j]
                            for j in range(len(IMC)):
                                moyenne_IMC[2][j] += IMC[j]
                            for j in range(len(passing)):
                                moyenne_passing[2][j] += passing[j]
                            for j in range(len(mentality_pos)):
                                moyenne_mentality_pos[2][j] += mentality_pos[j]
                            for j in range(len(mentality_vis)):
                                moyenne_mentality_vis[2][j] += mentality_vis[j]
                            for j in range(len(long_passing)):
                                moyenne_long_passing[2][j] += long_passing[j]
                            for j in range(len(pace)):
                                moyenne_pace[2][j] += pace[j]
                            for j in range(len(ball_control)):
                                moyenne_ball_control[2][j] += ball_control[j]
                            for j in range(len(short_passing)):
                                moyenne_short_passing[2][j] += short_passing[j]
                            for j in range(len(league)):
                                moyenne_league[2][j] += league[j]
                            for j in range(len(work_rate)):
                                moyenne_work_rate[2][j] += work_rate[j]
                            for j in range(len(physic)):
                                moyenne_physic[2][j] += physic[j]
                            for j in range(len(accel)):
                                moyenne_accel[2][j] += accel[j]
                            for j in range(len(strength)):
                                moyenne_strength[2][j] += strength[j]
                            for j in range(len(agility)):
                                moyenne_agility[2][j] += agility[j]
                            for j in range(len(aggression)):
                                moyenne_aggression[2][j] += aggression[j]
                            for j in range(len(def_central)):
                                center_back[2][j] += def_central[j]
                            for j in range(len(back_aile)):
                                back[2][j] += back_aile[j]
                            
                            if df_event['team'][index] == team[0]:
                                dico_compo[compo[0]][2] += 1
                            else : 
                                dico_compo[compo[1]][2] += 1
                        if packing == 4 or packing == 5 :
                            moyenne_duration[3] += duration
                            compteur_relance[3] += 1
                            moyenne_type_passes[3][0] += array_hauteur_pass[0]
                            moyenne_type_passes[3][1] += array_hauteur_pass[1]
                            moyenne_type_passes[3][2] += array_hauteur_pass[2]
                            moyenne_nb_passes[3] += nb_pass
                            moyenne_long_par_la_passe[3] += pass_length
                            moyenne_nb_carry[3] += nb_carry
                            moyenne_long_par_le_carry[3] += dist_carry
                            moyenne_pied_fort[3][0] += pref_foot[0]
                            moyenne_pied_fort[3][1] += pref_foot[1]
                            for j in range(len(weak_foot)):
                                moyenne_pied_faible[3][j] += weak_foot[j]
                            for j in range(len(skills)):
                                moyenne_skills[3][j] += skills[j]
                            for j in range(len(IMC)):
                                moyenne_IMC[3][j] += IMC[j]
                            for j in range(len(passing)):
                                moyenne_passing[3][j] += passing[j]
                            for j in range(len(mentality_pos)):
                                moyenne_mentality_pos[3][j] += mentality_pos[j]
                            for j in range(len(mentality_vis)):
                                moyenne_mentality_vis[3][j] += mentality_vis[j]
                            for j in range(len(long_passing)):
                                moyenne_long_passing[3][j] += long_passing[j]
                            for j in range(len(pace)):
                                moyenne_pace[3][j] += pace[j]
                            for j in range(len(ball_control)):
                                moyenne_ball_control[3][j] += ball_control[j]
                            for j in range(len(short_passing)):
                                moyenne_short_passing[3][j] += short_passing[j]
                            for j in range(len(league)):
                                moyenne_league[3][j] += league[j]
                            for j in range(len(work_rate)):
                                moyenne_work_rate[3][j] += work_rate[j]
                            for j in range(len(physic)):
                                moyenne_physic[3][j] += physic[j]
                            for j in range(len(accel)):
                                moyenne_accel[3][j] += accel[j]
                            for j in range(len(strength)):
                                moyenne_strength[3][j] += strength[j]
                            for j in range(len(agility)):
                                moyenne_agility[3][j] += agility[j]
                            for j in range(len(aggression)):
                                moyenne_aggression[3][j] += aggression[j]
                            for j in range(len(def_central)):
                                center_back[3][j] += def_central[j]
                            for j in range(len(back_aile)):
                                back[3][j] += back_aile[j]
                            
                            if df_event['team'][index] == team[0]:
                                dico_compo[compo[0]][3] += 1
                            else : 
                                dico_compo[compo[1]][3] += 1
                        if packing == 6 or packing == 7 :
                            moyenne_duration[4] += duration
                            compteur_relance[4] += 1
                            moyenne_type_passes[4][0] += array_hauteur_pass[0]
                            moyenne_type_passes[4][1] += array_hauteur_pass[1]
                            moyenne_type_passes[4][2] += array_hauteur_pass[2]
                            moyenne_nb_passes[4] += nb_pass
                            moyenne_long_par_la_passe[4] += pass_length
                            moyenne_nb_carry[4] += nb_carry
                            moyenne_long_par_le_carry[4] += dist_carry
                            moyenne_pied_fort[4][0] += pref_foot[0]
                            moyenne_pied_fort[4][1] += pref_foot[1]
                            for j in range(len(weak_foot)):
                                moyenne_pied_faible[4][j] += weak_foot[j]
                            for j in range(len(skills)):
                                moyenne_skills[4][j] += skills[j]
                            for j in range(len(IMC)):
                                moyenne_IMC[4][j] += IMC[j]
                            for j in range(len(passing)):
                                moyenne_passing[4][j] += passing[j]
                            for j in range(len(mentality_pos)):
                                moyenne_mentality_pos[4][j] += mentality_pos[j]
                            for j in range(len(mentality_vis)):
                                moyenne_mentality_vis[4][j] += mentality_vis[j]
                            for j in range(len(long_passing)):
                                moyenne_long_passing[4][j] += long_passing[j]
                            for j in range(len(pace)):
                                moyenne_pace[4][j] += pace[j]
                            for j in range(len(ball_control)):
                                moyenne_ball_control[4][j] += ball_control[j]
                            for j in range(len(short_passing)):
                                moyenne_short_passing[4][j] += short_passing[j]
                            for j in range(len(league)):
                                moyenne_league[4][j] += league[j]
                            for j in range(len(work_rate)):
                                moyenne_work_rate[4][j] += work_rate[j]
                            for j in range(len(physic)):
                                moyenne_physic[4][j] += physic[j]
                            for j in range(len(accel)):
                                moyenne_accel[4][j] += accel[j]
                            for j in range(len(strength)):
                                moyenne_strength[4][j] += strength[j]
                            for j in range(len(agility)):
                                moyenne_agility[4][j] += agility[j]
                            for j in range(len(aggression)):
                                moyenne_aggression[4][j] += aggression[j]
                            for j in range(len(def_central)):
                                center_back[4][j] += def_central[j]
                            for j in range(len(back_aile)):
                                back[4][j] += back_aile[j]
                            
                            if df_event['team'][index] == team[0]:
                                dico_compo[compo[0]][4] += 1
                            else : 
                                dico_compo[compo[1]][4] += 1
                        if packing == 8 or packing == 9 :
                            moyenne_duration[5] += duration
                            compteur_relance[5] += 1
                            moyenne_type_passes[5][0] += array_hauteur_pass[0]
                            moyenne_type_passes[5][1] += array_hauteur_pass[1]
                            moyenne_type_passes[5][2] += array_hauteur_pass[2]
                            moyenne_nb_passes[5] += nb_pass
                            moyenne_long_par_la_passe[5] += pass_length
                            moyenne_nb_carry[5] += nb_carry
                            moyenne_long_par_le_carry[5] += dist_carry
                            moyenne_pied_fort[5][0] += pref_foot[0]
                            moyenne_pied_fort[5][1] += pref_foot[1]
                            for j in range(len(weak_foot)):
                                moyenne_pied_faible[5][j] += weak_foot[j]
                            for j in range(len(skills)):
                                moyenne_skills[5][j] += skills[j]
                            for j in range(len(IMC)):
                                moyenne_IMC[5][j] += IMC[j]
                            for j in range(len(passing)):
                                moyenne_passing[5][j] += passing[j]
                            for j in range(len(mentality_pos)):
                                moyenne_mentality_pos[5][j] += mentality_pos[j]
                            for j in range(len(mentality_vis)):
                                moyenne_mentality_vis[5][j] += mentality_vis[j]
                            for j in range(len(long_passing)):
                                moyenne_long_passing[5][j] += long_passing[j]
                            for j in range(len(pace)):
                                moyenne_pace[5][j] += pace[j]
                            for j in range(len(ball_control)):
                                moyenne_ball_control[5][j] += ball_control[j]
                            for j in range(len(short_passing)):
                                moyenne_short_passing[5][j] += short_passing[j]
                            for j in range(len(league)):
                                moyenne_league[5][j] += league[j]
                            for j in range(len(work_rate)):
                                moyenne_work_rate[5][j] += work_rate[j]
                            for j in range(len(physic)):
                                moyenne_physic[5][j] += physic[j]
                            for j in range(len(accel)):
                                moyenne_accel[5][j] += accel[j]
                            for j in range(len(strength)):
                                moyenne_strength[5][j] += strength[j]
                            for j in range(len(agility)):
                                moyenne_agility[5][j] += agility[j]
                            for j in range(len(aggression)):
                                moyenne_aggression[5][j] += aggression[j]
                            for j in range(len(def_central)):
                                center_back[5][j] += def_central[j]
                            for j in range(len(back_aile)):
                                back[5][j] += back_aile[j]
                            
                            if df_event['team'][index] == team[0]:
                                dico_compo[compo[0]][5] += 1
                            else : 
                                dico_compo[compo[1]][5] += 1
                        if packing == 10 :
                            moyenne_duration[6] += duration
                            compteur_relance[6] += 1
                            moyenne_type_passes[6][0] += array_hauteur_pass[0]
                            moyenne_type_passes[6][1] += array_hauteur_pass[1]
                            moyenne_type_passes[6][2] += array_hauteur_pass[2]
                            moyenne_nb_passes[6] += nb_pass
                            moyenne_long_par_la_passe[6] += pass_length
                            moyenne_nb_carry[6] += nb_carry
                            moyenne_long_par_le_carry[6] += dist_carry
                            moyenne_pied_fort[6][0] += pref_foot[0]
                            moyenne_pied_fort[6][1] += pref_foot[1]
                            for j in range(len(weak_foot)):
                                moyenne_pied_faible[6][j] += weak_foot[j]
                            for j in range(len(skills)):
                                moyenne_skills[6][j] += skills[j]
                            for j in range(len(IMC)):
                                moyenne_IMC[6][j] += IMC[j]
                            for j in range(len(passing)):
                                moyenne_passing[6][j] += passing[j]
                            for j in range(len(mentality_pos)):
                                moyenne_mentality_pos[6][j] += mentality_pos[j]
                            for j in range(len(mentality_vis)):
                                moyenne_mentality_vis[6][j] += mentality_vis[j]
                            for j in range(len(long_passing)):
                                moyenne_long_passing[6][j] += long_passing[j]
                            for j in range(len(pace)):
                                moyenne_pace[6][j] += pace[j]
                            for j in range(len(ball_control)):
                                moyenne_ball_control[6][j] += ball_control[j]
                            for j in range(len(short_passing)):
                                moyenne_short_passing[6][j] += short_passing[j]
                            for j in range(len(league)):
                                moyenne_league[6][j] += league[j]
                            for j in range(len(work_rate)):
                                moyenne_work_rate[6][j] += work_rate[j]
                            for j in range(len(physic)):
                                moyenne_physic[6][j] += physic[j]
                            for j in range(len(accel)):
                                moyenne_accel[6][j] += accel[j]
                            for j in range(len(strength)):
                                moyenne_strength[6][j] += strength[j]
                            for j in range(len(agility)):
                                moyenne_agility[6][j] += agility[j]
                            for j in range(len(aggression)):
                                moyenne_aggression[6][j] += aggression[j]
                            for j in range(len(def_central)):
                                center_back[6][j] += def_central[j]
                            for j in range(len(back_aile)):
                                back[6][j] += back_aile[j]
                            
                            if df_event['team'][index] == team[0]:
                                dico_compo[compo[0]][6] += 1
                            else : 
                                dico_compo[compo[1]][6] += 1
                        
                else :
                    if df_event['type'][index_suiv] == "Pass" :
                        if df_event['pass_height'][index_suiv] == "Ground Pass":
                            array_hauteur_pass[0] += 1
                        if df_event['pass_height'][index_suiv] == "Low Pass":
                            array_hauteur_pass[1] += 1
                        if df_event['pass_height'][index_suiv] == "High Pass":
                            array_hauteur_pass[2] += 1
                        pass_length += df_event['pass_length'][index_suiv]
                        nb_pass += 1
                        duration += df_event['duration'][index_suiv]
                        liste_attente.append([df_event['location'][index_suiv][0], df_event['location'][index_suiv][1], df_event['pass_end_location'][index_suiv][0], df_event['pass_end_location'][index_suiv][1], df_event['type'][index_suiv], numero_relance])
                    if df_event['type'][index_suiv] == "Carry" :
                        nb_carry += 1
                        dist_carry += distance_entre_points(df_event['location'][index_suiv][0], df_event['location'][index_suiv][1], df_event['carry_end_location'][index_suiv][0], df_event['carry_end_location'][index_suiv][1])
                        duration += df_event['duration'][index_suiv]
                        liste_attente.append([df_event['location'][index_suiv][0], df_event['location'][index_suiv][1], df_event['carry_end_location'][index_suiv][0], df_event['carry_end_location'][index_suiv][1], df_event['type'][index_suiv], numero_relance])
                    for i in range(len(liste_attente)):
                        liste_echec.append(liste_attente[i])
                    compteur_relance[0] += 1
                    moyenne_duration[0] += duration 
                    moyenne_type_passes[0][0] += array_hauteur_pass[0]
                    moyenne_type_passes[0][1] += array_hauteur_pass[1]
                    moyenne_type_passes[0][2] += array_hauteur_pass[2]
                    moyenne_nb_passes[0] += nb_pass
                    moyenne_long_par_la_passe[0] += pass_length
                    moyenne_nb_carry[0] += nb_carry
                    moyenne_long_par_le_carry[0] += dist_carry
                    moyenne_pied_fort[0][0] += pref_foot[0]
                    moyenne_pied_fort[0][1] += pref_foot[1]
                    for j in range(len(weak_foot)):
                        moyenne_pied_faible[0][j] += weak_foot[j]
                    for j in range(len(skills)):
                        moyenne_skills[0][j] += skills[j]
                    for j in range(len(IMC)):
                        moyenne_IMC[0][j] += IMC[j]
                    for j in range(len(passing)):
                        moyenne_passing[0][j] += passing[j]
                    for j in range(len(mentality_pos)):
                        moyenne_mentality_pos[0][j] += mentality_pos[j]
                    for j in range(len(mentality_vis)):
                        moyenne_mentality_vis[0][j] += mentality_vis[j]
                    for j in range(len(long_passing)):
                        moyenne_long_passing[0][j] += long_passing[j]
                    for j in range(len(pace)):
                        moyenne_pace[0][j] += pace[j]
                    for j in range(len(ball_control)):
                        moyenne_ball_control[0][j] += ball_control[j]
                    for j in range(len(short_passing)):
                        moyenne_short_passing[0][j] += short_passing[j]
                    for j in range(len(league)):
                        moyenne_league[0][j] += league[j]
                    for j in range(len(work_rate)):
                        moyenne_work_rate[0][j] += work_rate[j]
                    for j in range(len(physic)):
                        moyenne_physic[0][j] += physic[j]
                    for j in range(len(accel)):
                        moyenne_accel[0][j] += accel[j]
                    for j in range(len(strength)):
                        moyenne_strength[0][j] += strength[j]
                    for j in range(len(agility)):
                        moyenne_agility[0][j] += agility[j]
                    for j in range(len(aggression)):
                        moyenne_aggression[0][j] += aggression[j]
                    for j in range(len(def_central)):
                        center_back[0][j] += def_central[j]
                    for j in range(len(back_aile)):
                        back[0][j] += back_aile[j]
                    
                    if df_event['team'][index] == team[0]:
                        dico_compo[compo[0]][0] += 1
                    else : 
                        dico_compo[compo[1]][0] += 1
                    
                numero_relance += 1

            if index_final != 0 :
                end_time = datetime.strptime(df_event['timestamp'][index_suiv], "%H:%M:%S.%f")
                time_difference = end_time - start_time
                time_difference_sec = time_difference.total_seconds()
                duration = time_difference_sec
                if df_event['location'][index_suiv][0] > 60 :
                    for i in range(len(liste_attente)):
                        liste_reussi.append(liste_attente[i])
                    packing = packing_relance(df_event['id'][first_index], df_event['id'][index_final], first_index, index_final, match_id)
                    if packing == 0 or packing == 1 :
                        moyenne_duration[1] += duration
                        compteur_relance[1] += 1
                        moyenne_type_passes[1][0] += array_hauteur_pass[0]
                        moyenne_type_passes[1][1] += array_hauteur_pass[1]
                        moyenne_type_passes[1][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[1] += nb_pass
                        moyenne_long_par_la_passe[1] += pass_length
                        moyenne_nb_carry[1] += nb_carry
                        moyenne_long_par_le_carry[1] += dist_carry
                        moyenne_pied_fort[1][0] += pref_foot[0]
                        moyenne_pied_fort[1][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[1][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[1][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[1][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[1][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[1][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[1][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[1][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[1][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[1][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[1][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[1][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[1][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[1][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[1][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[1][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[1][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[1][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[1][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[1][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][1] += 1
                        else : 
                            dico_compo[compo[1]][1] += 1
                    if packing == 2 or packing == 3 :
                        moyenne_duration[2] += duration
                        compteur_relance[2] += 1
                        moyenne_type_passes[2][0] += array_hauteur_pass[0]
                        moyenne_type_passes[2][1] += array_hauteur_pass[1]
                        moyenne_type_passes[2][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[2] += nb_pass
                        moyenne_long_par_la_passe[2] += pass_length
                        moyenne_nb_carry[2] += nb_carry
                        moyenne_long_par_le_carry[2] += dist_carry
                        moyenne_pied_fort[2][0] += pref_foot[0]
                        moyenne_pied_fort[2][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[2][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[2][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[2][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[2][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[2][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[2][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[2][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[2][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[2][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[2][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[2][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[2][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[2][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[2][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[2][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[2][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[2][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[2][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[2][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][2] += 1
                        else : 
                            dico_compo[compo[1]][2] += 1
                    if packing == 4 or packing == 5 :
                        moyenne_duration[3] += duration
                        compteur_relance[3] += 1
                        moyenne_type_passes[3][0] += array_hauteur_pass[0]
                        moyenne_type_passes[3][1] += array_hauteur_pass[1]
                        moyenne_type_passes[3][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[3] += nb_pass
                        moyenne_long_par_la_passe[3] += pass_length
                        moyenne_nb_carry[3] += nb_carry
                        moyenne_long_par_le_carry[3] += dist_carry
                        moyenne_pied_fort[3][0] += pref_foot[0]
                        moyenne_pied_fort[3][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[3][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[3][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[3][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[3][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[3][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[3][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[3][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[3][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[3][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[3][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[3][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[3][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[3][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[3][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[3][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[3][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[3][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[3][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[3][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][3] += 1
                        else : 
                            dico_compo[compo[1]][3] += 1
                    if packing == 6 or packing == 7 :
                        moyenne_duration[4] += duration
                        compteur_relance[4] += 1
                        moyenne_type_passes[4][0] += array_hauteur_pass[0]
                        moyenne_type_passes[4][1] += array_hauteur_pass[1]
                        moyenne_type_passes[4][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[4] += nb_pass
                        moyenne_long_par_la_passe[4] += pass_length
                        moyenne_nb_carry[4] += nb_carry
                        moyenne_long_par_le_carry[4] += dist_carry
                        moyenne_pied_fort[4][0] += pref_foot[0]
                        moyenne_pied_fort[4][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[4][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[4][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[4][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[4][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[4][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[4][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[4][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[4][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[4][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[4][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[4][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[4][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[4][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[4][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[4][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[4][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[4][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[4][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[4][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][4] += 1
                        else : 
                            dico_compo[compo[1]][4] += 1
                    if packing == 8 or packing == 9 :
                        moyenne_duration[5] += duration
                        compteur_relance[5] += 1
                        moyenne_type_passes[5][0] += array_hauteur_pass[0]
                        moyenne_type_passes[5][1] += array_hauteur_pass[1]
                        moyenne_type_passes[5][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[5] += nb_pass
                        moyenne_long_par_la_passe[5] += pass_length
                        moyenne_nb_carry[5] += nb_carry
                        moyenne_long_par_le_carry[5] += dist_carry
                        moyenne_pied_fort[5][0] += pref_foot[0]
                        moyenne_pied_fort[5][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[5][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[5][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[5][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[5][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[5][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[5][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[5][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[5][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[5][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[5][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[5][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[5][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[5][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[5][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[5][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[5][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[5][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[5][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[5][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][5] += 1
                        else : 
                            dico_compo[compo[1]][5] += 1
                    if packing == 10 :
                        moyenne_duration[6] += duration
                        compteur_relance[6] += 1
                        moyenne_type_passes[6][0] += array_hauteur_pass[0]
                        moyenne_type_passes[6][1] += array_hauteur_pass[1]
                        moyenne_type_passes[6][2] += array_hauteur_pass[2]
                        moyenne_nb_passes[6] += nb_pass
                        moyenne_long_par_la_passe[6] += pass_length
                        moyenne_nb_carry[6] += nb_carry
                        moyenne_long_par_le_carry[6] += dist_carry
                        moyenne_pied_fort[6][0] += pref_foot[0]
                        moyenne_pied_fort[6][1] += pref_foot[1]
                        for j in range(len(weak_foot)):
                            moyenne_pied_faible[6][j] += weak_foot[j]
                        for j in range(len(skills)):
                            moyenne_skills[6][j] += skills[j]
                        for j in range(len(IMC)):
                            moyenne_IMC[6][j] += IMC[j]
                        for j in range(len(passing)):
                            moyenne_passing[6][j] += passing[j]
                        for j in range(len(mentality_pos)):
                            moyenne_mentality_pos[6][j] += mentality_pos[j]
                        for j in range(len(mentality_vis)):
                            moyenne_mentality_vis[6][j] += mentality_vis[j]
                        for j in range(len(long_passing)):
                            moyenne_long_passing[6][j] += long_passing[j]
                        for j in range(len(pace)):
                            moyenne_pace[6][j] += pace[j]
                        for j in range(len(ball_control)):
                            moyenne_ball_control[6][j] += ball_control[j]
                        for j in range(len(short_passing)):
                            moyenne_short_passing[6][j] += short_passing[j]
                        for j in range(len(league)):
                            moyenne_league[6][j] += league[j]
                        for j in range(len(work_rate)):
                            moyenne_work_rate[6][j] += work_rate[j]
                        for j in range(len(physic)):
                            moyenne_physic[6][j] += physic[j]
                        for j in range(len(accel)):
                            moyenne_accel[6][j] += accel[j]
                        for j in range(len(strength)):
                            moyenne_strength[6][j] += strength[j]
                        for j in range(len(agility)):
                            moyenne_agility[6][j] += agility[j]
                        for j in range(len(aggression)):
                            moyenne_aggression[6][j] += aggression[j]
                        for j in range(len(def_central)):
                            center_back[6][j] += def_central[j]
                        for j in range(len(back_aile)):
                            back[6][j] += back_aile[j]
                        
                        if df_event['team'][index] == team[0]:
                            dico_compo[compo[0]][6] += 1
                        else : 
                            dico_compo[compo[1]][6] += 1

                else : 
                    for i in range(len(liste_attente)):
                        liste_echec.append(liste_attente[i])
                    compteur_relance[0] += 1
                    moyenne_duration[0] += duration
                    moyenne_type_passes[0][0] += array_hauteur_pass[0]
                    moyenne_type_passes[0][1] += array_hauteur_pass[1]
                    moyenne_type_passes[0][2] += array_hauteur_pass[2]
                    moyenne_nb_passes[0] += nb_pass
                    moyenne_long_par_la_passe[0] += pass_length
                    moyenne_nb_carry[0] += nb_carry
                    moyenne_long_par_le_carry[0] += dist_carry
                    moyenne_pied_fort[0][0] += pref_foot[0]
                    moyenne_pied_fort[0][1] += pref_foot[1]
                    for j in range(len(weak_foot)):
                        moyenne_pied_faible[0][j] += weak_foot[j]
                    for j in range(len(skills)):
                        moyenne_skills[0][j] += skills[j]
                    for j in range(len(IMC)):
                        moyenne_IMC[0][j] += IMC[j]
                    for j in range(len(passing)):
                        moyenne_passing[0][j] += passing[j]
                    for j in range(len(mentality_pos)):
                        moyenne_mentality_pos[0][j] += mentality_pos[j]
                    for j in range(len(mentality_vis)):
                        moyenne_mentality_vis[0][j] += mentality_vis[j]
                    for j in range(len(long_passing)):
                        moyenne_long_passing[0][j] += long_passing[j]
                    for j in range(len(pace)):
                        moyenne_pace[0][j] += pace[j]
                    for j in range(len(ball_control)):
                        moyenne_ball_control[0][j] += ball_control[j]
                    for j in range(len(short_passing)):
                        moyenne_short_passing[0][j] += short_passing[j]
                    for j in range(len(league)):
                        moyenne_league[0][j] += league[j]
                    for j in range(len(work_rate)):
                        moyenne_work_rate[0][j] += work_rate[j]
                    for j in range(len(physic)):
                        moyenne_physic[0][j] += physic[j]
                    for j in range(len(accel)):
                        moyenne_accel[0][j] += accel[j]
                    for j in range(len(strength)):
                        moyenne_strength[0][j] += strength[j]
                    for j in range(len(agility)):
                        moyenne_agility[0][j] += agility[j]
                    for j in range(len(aggression)):
                        moyenne_aggression[0][j] += aggression[j]
                    for j in range(len(def_central)):
                        center_back[0][j] += def_central[j]
                    for j in range(len(back_aile)):
                        back[0][j] += back_aile[j]
                    
                    if df_event['team'][index] == team[0]:
                        dico_compo[compo[0]][0] += 1
                    else : 
                        dico_compo[compo[1]][0] += 1
                
                numero_relance += 1

    for i in range(len(moyenne_duration)):
        if compteur_relance[i] != 0 :
            moyenne_duration[i] /= compteur_relance[i]
            moyenne_nb_passes[i] /= compteur_relance[i]
            moyenne_long_par_la_passe[i] /= compteur_relance[i]
            moyenne_nb_carry[i] /= compteur_relance[i]
            moyenne_long_par_le_carry[i] /= compteur_relance[i]
           
    return moyenne_duration, moyenne_type_passes, moyenne_nb_passes, moyenne_long_par_la_passe, moyenne_nb_carry, moyenne_long_par_le_carry, moyenne_pied_fort, moyenne_pied_faible, moyenne_skills, moyenne_IMC, moyenne_passing, moyenne_mentality_pos, moyenne_mentality_vis, moyenne_long_passing, moyenne_pace, moyenne_ball_control, moyenne_short_passing, moyenne_league, moyenne_work_rate, moyenne_physic, moyenne_accel, moyenne_strength, moyenne_agility, moyenne_aggression
    
for i in WC2022_games :
    mean_duration, mean_type_passes, mean_nb_passes, mean_long_par_la_passe, mean_nb_carry, mean_long_par_le_carry, mean_pied_fort, mean_pied_faible, mean_skills, mean_IMC, mean_passing, mean_mentality_pos, mean_mentality_vis, mean_long_passing, mean_pace, mean_ball_control, mean_short_passing, mean_league, mean_work_rate, mean_physic, mean_accel, mean_strength, mean_agility, mean_aggression = relance_coup_franc(i)
    
    for j in range(len(mean_duration)):
        if mean_duration[j] != 0:
            compteur_duration[j] += 1
    for j in range(len(mean_pied_fort)):
        if mean_pied_fort[j][0] != 0 or mean_pied_fort[j][1] != 0 :
            compteur_pied_fort[j] += 1

    for j in range(len(duration)):
        duration[j] += mean_duration[j]
        nb_passes[j] += mean_nb_passes[j]
        long_par_la_passe[j] += mean_long_par_la_passe[j]
        nb_carry[j] += mean_nb_carry[j]
        long_par_le_carry[j] += mean_long_par_le_carry[j]
        for k in range(len(type_passes[0])):
            type_passes[j][k] += mean_type_passes[j][k]
        for k in range(len(pied_fort[0])):
            pied_fort[j][k] += mean_pied_fort[j][k]
        for k in range(len(pied_faible[0])):
            pied_faible[j][k] += mean_pied_faible[j][k]
        for k in range(len(skills[0])):
            skills[j][k] += mean_skills[j][k]
        for k in range(len(IMC[0])):
            IMC[j][k] += mean_IMC[j][k]
        for k in range(len(passing[0])):
            passing[j][k] += mean_passing[j][k]
        for k in range(len(mentality_pos[0])):
            mentality_pos[j][k] += mean_mentality_pos[j][k]
        for k in range(len(mentality_vis[0])):
            mentality_vis[j][k] += mean_mentality_vis[j][k]
        for k in range(len(long_passing[0])):
            long_passing[j][k] += mean_long_passing[j][k]
        for k in range(len(pace[0])):
            pace[j][k] += mean_pace[j][k]
        for k in range(len(ball_control[0])):
            ball_control[j][k] += mean_ball_control[j][k]
        for k in range(len(short_passing[0])):
            short_passing[j][k] += mean_short_passing[j][k]
        for k in range(len(league[0])):
            league[j][k] += mean_league[j][k]
        for k in range(len(work_rate[0])):
            work_rate[j][k] += mean_work_rate[j][k]
        for k in range(len(physic[0])):
            physic[j][k] += mean_physic[j][k]
        for k in range(len(accel[0])):
            accel[j][k] += mean_accel[j][k]
        for k in range(len(strength[0])):
            strength[j][k] += mean_strength[j][k]
        for k in range(len(agility[0])):
            agility[j][k] += mean_agility[j][k]
        for k in range(len(aggression[0])):
            aggression[j][k] += mean_aggression[j][k]


for i in range(len(duration)) :
    if compteur_duration[i] != 0 :
        duration[i] /= compteur_duration[i]
        nb_passes[i] /= compteur_duration[i]
        long_par_la_passe[i] /= compteur_duration[i]
        nb_carry[i] /= compteur_duration[i]
        long_par_le_carry[i] /= compteur_duration[i]
        
print("Resultat Duree = ", duration)
print("Resultat Type de passes = ", type_passes)
print("Resultat Nombre de passes = ", nb_passes)
print("Resultat Longueur par la passe = ", long_par_la_passe)
print("Resultat Nombre de carries = ", nb_carry)
print("Resultat Longueur par le carry = ", long_par_le_carry)
print("Resultat Pied fort = ", pied_fort)
print("Resultat Pied faible = ", pied_faible)
print("Resultat Skills = ", skills)
print("Resultat IMC = ", IMC)
print("Resultat Passing = ", passing)
print("Resultat Mentality positioning = ", mentality_pos)
print("Resultat mentality vision = ", mentality_vis)
print("Resultat Long passing = ", long_passing)
print("Resultat Pace = ", pace)
print("Resultat Ball control = ", ball_control)
print("Resultat Short passing = ", short_passing)
print("Resultat League = ", league)
print("Resultat Work rate = ", work_rate)
print("Resultat Physic = ", physic)
print("Resultat Accel = ", accel)
print("Resultat Strength = ", strength)
print("Resultat Agility = ", agility)
print("Resultat Aggression = ", aggression)
print("Resultat def central = ", center_back)
print("Resultat back d'aile = ", back)

print("Compo final = ", dico_compo)