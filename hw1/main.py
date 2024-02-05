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

def check_duplicate_edge(path, start, next):
    for i in range(len(path) - 1):
        if path[i] == start and path[i+1] == next:
            return True
    return False

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

#returns the sum of all location and all edge preferences in a road trip.
# The roadtrip argument can be any graph of valid locations and edges – it need
# not be round trip, and it need not be connected — this is because the function
# can be called on partially constructed road-trips at intermediate points in search,
# as well as being called to evaluate the utility of a fully-connected round-trip.
# You decide on the internal representation of the roadtrip argument.
def total_preference(roadtrip):
    pass


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
