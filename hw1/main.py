from collections import defaultdict
import random
import networkx as nx
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

    # construct adjacency list with loc/edge prefs
    for _, row in edges_df.iterrows(): 
        edgePref = random.uniform(0,.1)
        # location name => array of tuples (location, distance, edge preference)
        G[row['locationA']].append((row['locationB'], row['actualDistance'], edgePref))
        G[row['locationB']].append((row['locationA'], row['actualDistance'], edgePref))
        # location name => location preference
        locationPrefs[row['locationA']] = random.uniform(0,1)
        locationPrefs[row['locationB']] = random.uniform(0,1)

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
# edge independently. Note that edges have a smaller preference upper bound than locations for Program 1
def edge_preference_assignments(a, b):

    if not (0 <= a <= b <= 0.1):
        raise ValueError("Invalid range for 'a' and 'b'. Should be between 0 and 0.1")

    #Placeholder until we figure out how we want to store the prefrence. 
    edges = [("node1", "node2"), ("node2", "node3"), ("node3", "node4")]

    edge_assignments = {}

    for edge in edges:
        edge_assignments[edge] = random.uniform(a, b)

    return edge_assignments

# assigns random values between a=0 and b=1 inclusive using a uniform distribution to each location independently
def location_preference_assignments(a, b):
    # Predefined list of locations
    locations = ['Location1', 'Location2', 'Location3', ...]  # Add your locations here

    # Check if the range is valid
    if a > b:
        raise ValueError("a should be less than or equal to b")

    # Assign a random value to each location
    assignments = {location: random.uniform(a, b) for location in locations}
    
    return assignments


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