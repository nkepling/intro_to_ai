## Group #7
## Members: Mohammad Khan, Nathan Keplinger, Raj Chopra, Theo Hodges
## How to run: 

from collections import defaultdict
import random
#import networkx as nx
import pandas as pd
import time

class Node:
    def __init__(self):
        self.name = ''
        # self.children stores adjacent nodes via the following interfact (Node, Length, Edge Preference) <- that's a tuple
        self.children = []

## This function performs a search for round trip road trips starting from a given location. It reads location and edge information from specified files, constructs a graph, and searches for all possible round trips that fit within a specified maximum time constraint.
"""
    Parameters:
    startLoc (str): The starting location for the road trip.
    LocFile (str): The file path to a CSV file containing location data.
    EdgeFile (str): The file path to a CSV file containing edge (road) data between locations.
    maxTime (int/float): The maximum time allowed for the road trip in hours.
    x_mph (int/float): The average speed in miles per hour for the road trip.
    resultFile (str): The file path where the results will be saved.

    The search algorithm uses a depth-first approach, storing potential paths in a stack and evaluating them based on a heuristic value. It prompts the user at regular intervals to decide whether to continue the search.

    Returns:
    solutions (list of tuples): A list of tuples, where each tuple contains a valid round trip path and its heuristic value. Each path is a round trip that starts and ends at the startLoc, fits within the maxTime constraint, and optimizes for location and edge preferences.

    Note:
    - The function prints the search progress and the found solutions.
    - The function saves a summary of the solutions and statistics about the search to the specified result file.
"""
def RoundTripRoadTrip(startLoc, LocFile, EdgeFile, maxTime, x_mph, resultFile):
    print("Starting search...")

    locations_df, edges_df = read_csv(LocFile, EdgeFile)

    G, locationPrefs = make_graph(locations_df, edges_df)

    # Initalize the frontier and solution arrays
    solutions = []
    stack = []
    #stack objects look like (LocationName, Path, Heauristic Value)
    stack.append((startLoc, [startLoc], 0))
    startTime = time.time()


    while True:
        if len(solutions) > 0:
            pauseStart = time.time()
            print("Found " + str(len(solutions)) + " solutions ")
            yn = input("Would you like to keep searching? (y/n) ")
            pauseEnd = time.time()
            startTime += pauseEnd - pauseStart
            if yn == 'n':
                endTime = time.time()
                break


        while stack:
            location, path, h = stack.pop()

            #check for time constraint here
            if time_estimate(path, x_mph,locationPrefs,G) > maxTime:
                continue

            #check if location == start location and path is a round trip
            if location == startLoc and len(path) > 1:
                solutions.append((path, h))
                if len(solutions) < 3:
                    continue
                elif len(solutions) == 3 or len(solutions) % 6 == 0: # ask user to continue or not every 6 solutions and when first 3 are found
                    break



            next = []
            for neighbor, distance, edge_pref in G[location]:
                location_utility = locationPrefs[neighbor] if neighbor not in path else 0
                edge_utility = edge_pref if not check_duplicate_edge(path, location, neighbor) else 0

                next.append((neighbor, path + [neighbor], h + location_utility + edge_utility))
            next.sort(key=lambda x: x[2])

            stack.extend(next)


    print('Solutions: ')
    totalTime = endTime - startTime
    summary_stats(solutions,totalTime)
    print_solutions(solutions,resultFile)



    return solutions

 """
    Checks if an edge, defined by a start and next location, already exists in a given path.

    This function is used to ensure that a new edge (road) being considered for addition to a road trip path is not a duplicate of an edge already in the path. This is important for avoiding revisiting the same road between two specific locations.

    Parameters:
    path (list): The current path of the road trip as a list of locations.
    start (str): The starting location of the edge to check.
    next (str): The next location (destination) of the edge to check.

    Returns:
    bool: Returns True if the edge (start to next) is already present in the path, False otherwise.

    The function iterates through the path and compares each consecutive pair of locations with the start and next locations. If a match is found, it indicates that the edge is a duplicate.
"""
def check_duplicate_edge(path, start, next):
    for i in range(len(path) - 1):
        if path[i] == start and path[i+1] == next:
            return True
    return False


"""
    Constructs a graph representation of locations and edges for a road trip, 
    along with preference scores for each location and edge.

    This function takes dataframes containing information about locations and 
    edges and converts them into a graph structure. It assigns preference scores 
    to each location and edge based on the data provided. This graph is used to 
    facilitate the search for optimal road trip paths.

    Parameters:
    locations_df (DataFrame): A Pandas DataFrame containing information about 
                              various locations. Expected to have columns for 
                              'Location Label' and 'Preference'.
    edges_df (DataFrame): A Pandas DataFrame containing information about edges 
                          (roads) between locations. Expected to have columns 
                          for 'locationA', 'locationB', 'actualDistance', and 
                          'Preference'.

    The function initializes a graph (G) and a dictionary for location preferences 
    (locationPrefs). It then iterates through the dataframes to populate these 
    structures with the relevant data.

    Returns:
    tuple: A tuple containing two elements:
           1. G (defaultdict of list): A graph represented as an adjacency list, 
              where each key is a location, and its value is a list of tuples 
              (neighbor location, distance, edge preference).
           2. locationPrefs (defaultdict of float): A dictionary mapping each 
              location to its preference score.

    Note:
    - The graph is undirected; hence, for each edge in edges_df, entries are made 
      for both directions.
    - Preference scores are used to evaluate the desirability of visiting a 
      location or traveling an edge during the road trip.
"""
def make_graph(locations_df, edges_df):
    G = defaultdict(list)
    locationPrefs = defaultdict(float)

    # Assigning preference value to each location
    location_preference_assignments(0, 1, locations_df)

    # Assigning preference value to each edge
    edge_preference_assignments(0, 0.1, edges_df)

    for _, row in locations_df.iterrows():
        locationPrefs[row['Location Label']] = row['Preference']

    for _, row in edges_df.iterrows():
        G[row['locationA']].append((row['locationB'], row['actualDistance'], row['Preference']))
        G[row['locationB']].append((row['locationA'], row['actualDistance'], row['Preference']))

    return G, locationPrefs


"""
    Calculates the total preference score for a given road trip path.

    This function computes the overall utility or preference score of a road trip 
    by summing up the preference scores of each location visited and the preference 
    scores of each edge (road segment) traversed in the trip. The preference scores 
    for locations and edges are provided as input in the form of a dictionary and 
    graph respectively.

    Parameters:
    roadtrip (list): A list of location names (strings) representing the order of 
                     locations visited in the road trip.
    locationPrefs (dict): A dictionary where keys are location names and values are 
                          their corresponding preference scores (floats).
    G (dict): A graph represented as a dictionary of dictionaries, where each key is 
              a location, and its value is another dictionary representing the edges 
              from this location to other locations. Each edge has an associated 
              preference score.

    The function iterates through each location in the road trip list, adding the 
    preference score of each location. It then iterates through the pairs of 
    consecutive locations in the road trip, adding the preference score of the edge 
    that connects these locations.

    Returns:
    int/float: The total preference score (utility) of the road trip, summing both 
               location and edge preferences.

    Note:
    - The function assumes that the road trip list provided is a valid path in the 
      graph G.
    - Edge preferences are only added for direct connections between consecutive 
      locations in the road trip list.
"""
def total_preference(roadtrip, locationPrefs, G):
    util = 0
    for i in range(len(roadtrip)):
        util += locationPrefs[roadtrip[i]]

    for i in range(len(roadtrip) - 1):
        for neighbor, distance, edge_pref in G[roadtrip[i]]:
            if neighbor == roadtrip[i+1]:
                util += edge_pref
    
    return util

# computes the time required by a road trip in terms of its constituent edges and locations
def time_estimate(roadtrip, x_mph, locationPrefs,G):
    time = time_at_location(locationPrefs[roadtrip[0]])
    for i in range(len(roadtrip) - 1):
        time += compute_travel_time(roadtrip[i], roadtrip[i+1], x_mph,G)
        time += time_at_location(locationPrefs[roadtrip[i+1]])

    return time

# this function c
def compute_travel_time(start,end,x_mph,G):
    for neighbor, distance, edge_pref in G[start]:
        if neighbor == end:
            return (float(distance) / x_mph) + time_at_location(edge_pref)



# time at a location as a functino of the location's preference
def time_at_location(preference):
    return float(preference)*10 #arbitrary scaling factor


# assigns random values between a=0 and b=0.1 inclusive using a uniform distribution to each
# edge independently. Adds the preference value to a dataframe containing all the edges.
def edge_preference_assignments(a, b, edges_df):

    # Checking if the range is valid
    if not (0 <= a <= b <= 0.1):
        raise ValueError("Invalid range for 'a' and 'b'. Should be between 0 and 0.1")

    for index, row in edges_df.iterrows():
        edges_df.at[index, 'Preference'] = random.uniform(a, b)

    return

# assigns random values between a=0 and b=1 inclusive using a uniform distribution to each location independently.
# Adds the preference value to a dataframe containing all the locations.
def location_preference_assignments(a, b, locations_df):

    # Check if the range is valid
    if not (0 <= a <= b <= 1):
        raise ValueError("Invalid range for 'a' and 'b'. Should be between 0 and 1")

    for index, row in locations_df.iterrows():
        locations_df.at[index, 'Preference'] = random.uniform(a, b)

    return

#reads the CSVs, constructs the graph (requires callign edge/loc pref assignments) and returns it
def read_csv(locFile, edgeFile):

    # Read the locations and edges CSV files
    locations_df = pd.read_csv(locFile)
    edges_df = pd.read_csv(edgeFile)

    return locations_df, edges_df

def print_solutions(solutions,resultFile):

    d = {"path":[],
         "pref":[]}

    for p in solutions:
        string_path = "-".join(p[0])
        d["path"].append(string_path)
        d["pref"].append(p[1])

    df = pd.DataFrame(d)

    df.to_csv(resultFile,index=False)
    print("Data Saved to CSV")

def summary_stats(solutions,total_time):

    max_pref = 0
    min_pref = 99999999
    sum_pref = 0

    for p in solutions:
        pref = p[1]
        if p[1] >= max_pref:
            max_pref = p[1]

        if p[1] <= min_pref:
            min_pref = p[1]

        sum_pref+=p[1]

    print("Average search time per solution: ", total_time/(len(solutions)))
    print("Max TotalTripPreference", max_pref)
    print("Min TotalTripPreference", min_pref)
    print("Average TotalTripPrefrence", sum_pref/(len(solutions)) )

    return {"Max":max_pref,"Min":min_pref,"Average_Time":total_time/len(solutions),"Ave_Pref":sum_pref/(len(solutions))}


def main():

    startLoc = "ColumbusOH"
    LocFile = 'locations.csv'
    EdgeFile = 'edges.csv'
    maxTime = 200
    x_mph = 10
    resultFile = 'result.csv'

    RoundTripRoadTrip(startLoc, LocFile, EdgeFile, maxTime, x_mph, resultFile)










if __name__ == "__main__":
    main()
