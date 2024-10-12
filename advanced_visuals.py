from ift6758.data import load_events_dataframe
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from matplotlib.colors import BoundaryNorm
import pandas as pd

# The main function for advanced visuals is 'get_adv_visuals(year)' where year is an int in between 2016 and 2020

rink_width = 550
rink_height = 467
rink_x_min, rink_x_max = 0, 100
rink_y_min, rink_y_max = -42.5, 42.5
rink_image_path = "./figures/nhl_rink_cropped.png"


# Utils
def how_many_non_zero_elem(matrix):
    '''
    Takes a 2D matrix of floats for input and return the number of positive elements higher than 0 and a list of their
    coordinates and value.
    '''
    num = 0
    list_coord = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            elem = matrix[i][j]
            if elem != 0 and elem > 0.0000000000001:
                tuple = (i, j, float(elem))
                list_coord.append(tuple)
                num += 1
    return num, list_coord


def transform_coordinates(x, y):
    '''
    Takes 2 floats for input and returns their values converted for them to fit correctly on the half nhl rink image.
    '''
    transformed_x = ((x - rink_x_min) / (rink_x_max - rink_x_min)) * rink_width
    transformed_y = rink_height - ((y - rink_y_min) / (rink_y_max - rink_y_min) * rink_height)
    return transformed_x, transformed_y


def get_teams_dict(teams):
    '''
    Takes a list of teams (strings) as input to return a dictionnary with teams as keys and an int index for values.
    '''
    teams_dict = {}
    for i in range(0, len(teams)):
        teams_dict[teams[i]] = i
    return teams_dict


def put_right_format(array):
    '''
    Takes a 1D array (here it is the number of games per team) of length n and converts it to a 3D array of shape
    (n, 468, 551) so its shape fits the shape of the 'rates' variable.
    '''
    final_matrix = []
    for elem in array:
        new_elem = np.array([[elem] * (550 + 1)] * (467 + 1))
        final_matrix.append(new_elem)
    return np.array(final_matrix)


# Main function
def get_adv_visuals(year: int):

    assert year in [2016, 2017, 2018, 2019, 2020]
    if year == 2016:
        teams = ['Capitals', 'Penguins', 'Blue Jackets', 'Canadiens',
                 'Senators', 'Bruins', 'Rangers', 'Maple Leafs', 'Islanders',
                 'Lightning', 'Flyers', 'Hurricanes', 'Panthers',
                 'Red Wings', 'Sabres', 'Devils', 'Blackhawks', 'Wild',
                 'Blues', 'Ducks', 'Oilers', 'Sharks', 'Flames',
                 'Predators', 'Jets', 'Kings', 'Stars', 'Coyotes',
                 'Canucks', 'Avalanche']
        num_teams = len(teams)
    else:
        teams = ['Capitals', 'Penguins', 'Blue Jackets', 'Canadiens',
                 'Senators', 'Bruins', 'Rangers', 'Maple Leafs', 'Islanders',
                 'Lightning', 'Flyers', 'Hurricanes', 'Panthers',
                 'Red Wings', 'Sabres', 'Devils', 'Blackhawks', 'Wild',
                 'Blues', 'Ducks', 'Oilers', 'Sharks', 'Flames',
                 'Predators', 'Jets', 'Kings', 'Stars', 'Coyotes',
                 'Canucks', 'Avalanche', 'Golden Knights']
        num_teams = len(teams)

    teams_dict = get_teams_dict(teams)

    # Load dataframe
    df = load_events_dataframe(year)

    # We only keep shots made in offensive zone ('O') or offensive side of neutral zone ('O'). That means, every shot
    # made on the same half-ice as the goal the shot is on.
    # df = df.query('x_coord*goal_x_coord >= 0')  # this query should work but it reduces the amount of shots taken
    # into account
    df = df.query('zone_code == \'O\' or (zone_code == \'N\' and x_coord*goal_x_coord >= 0)')

    # Calculation of shot per spot per hour per team
    rates = [[[0.0] * (550 + 1)] * (467 + 1)] * (num_teams + 1)
    rates = np.array(rates)
    last_gid = 0
    number_of_games = [0] * (num_teams + 1)
    print(f"Length of dataframe : {len(df)}")

    # For every event (shot-on-goal or goal)
    for i in range(len(df)):
        plus_one_game = False
        point = df.iloc[i]
        gid = point["game_id"]
        x = float(point["x_coord"])
        y = float(point["y_coord"])

        # Skip iteration if x_coord or y_coord is NaN
        if pd.isna(x) or pd.isna(y):
            continue

        if gid != last_gid:  # New game
            number_of_games[-1] += 1
            last_gid = gid
            plus_one_game = True

        home_id = point["home_team_id"]
        away_id = point["away_team_id"]
        team_id = point["event_owner_team_id"]

        if plus_one_game:
            home_index = teams_dict[point["home_team_name"]]
            number_of_games[home_index] += 1
            away_index = teams_dict[point["away_team_name"]]
            number_of_games[away_index] += 1

        if team_id == home_id:
            team = point["home_team_name"]
        else:
            team = point["away_team_name"]
        team_index = teams_dict[team]

        if x < 0:
            x = -x
            y = -y  # Needed for good symetry

        x_trans, y_trans = transform_coordinates(x, y)
        x = round(x_trans)
        y = round(y_trans)
        rates[-1][y][x] += 1
        rates[team_index][y][x] += 1

    assert np.sum(number_of_games) - (number_of_games[-1] * 3) == 0
    print(f'Total number of games is {number_of_games[-1]}')

    number_of_games[-1] = number_of_games[-1] * 2  # Times two because two teams are playing each game
    number_of_games = put_right_format(number_of_games)  # Change format to divide rates by number_of_games
    rates = np.divide(rates, number_of_games)

    # Now let's take the difference of each team's shot rates by the average one
    final_rates = np.subtract(rates[0:num_teams], rates[-1])

    # Finally, let's plot these final_rates
    # First things first, let's smooth the data using a gaussian filter and find the absolute extremum of our data
    # to build a uniform colorbar for every team
    smoothed_rates = []
    extremum = 0
    for i in range(len(final_rates)):
        rates = final_rates[i]
        smoothed_rates.append(gaussian_filter(rates, sigma=12))
        potential_extremum = np.max(np.abs(smoothed_rates))
        if potential_extremum > extremum:
            extremum = potential_extremum

    # Now build every plot and save
    for i in range(len(final_rates)):
        team = list(teams_dict.keys())[i]
        smoothed_rate = smoothed_rates[i]
        filepath = './figures/' + str(year) + '_' + str(team) + '.png'
        x = np.linspace(0, 551, 551)
        y = np.linspace(0, 468, 468)
        X, Y = np.meshgrid(x, y)

        levels = np.linspace(-extremum, extremum, 8)
        cmap = plt.get_cmap('coolwarm')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        img = np.asarray(Image.open(rink_image_path))
        plt.imshow(img)
        visu = plt.contourf(X, Y, smoothed_rate, levels=levels, cmap=cmap, norm=norm, alpha=0.8)
        cbar = plt.colorbar(visu, ticks=levels)
        cbar.ax.set_ylabel('Shot Rate Difference per Hour')
        tick_locs = (levels[:-1] + levels[1:]) / 2
        cbar.set_ticks(tick_locs)
        plt.title('Shot rate difference for the ' + str(team) + '\nwith the mean for the league in ' + str(year))
        plt.savefig(filepath)
        plt.clf()
    return


get_adv_visuals(2020)