class Node: 
    def __init__(self): 
        self.name = ''
        # self.children stores adjacent nodes via the following interfact (Node, Length, Edge Preference) <- that's a tuple
        self.children = []
    
def RoundTripRoadTrip(startLoc, LocFile, EdgeFile, maxTime, x_mph, resultFile):
    pass


#returns the sum of all location and all edge preferences in a road trip. 
# The roadtrip argument can be any graph of valid locations and edges – it need 
# not be round trip, and it need not be connected — this is because the function 
# can be called on partially constructed road-trips at intermediate points in search,
# as well as being called to evaluate the utility of a fully-connected round-trip. 
# You decide on the internal representation of the roadtrip argument.
def total_preference(roadtrip):
    pass


# computes the time required by a road trip in terms of its constituent edges and locations
def time_estimate(roadtrip, x_mph):
    pass


# the greater the preference for a location, the more time spent at the location. loc can be an edge or vertex
def time_at_location(loc):
    pass


# assigns random values between a=0 and b=0.1 inclusive using a uniform distribution to each 
# edge independently. Note that edges have a smaller preference upper bound than locations for Program 1
def edge_preference_assignments(a, b):
    pass

# assigns random values between a=0 and b=1 inclusive using a uniform distribution to each location independently
def location_preference_assignments(a, b):
    pass


#reads the CSVs, constructs the graph (requires callign edge/loc pref assignments) and returns it
def read_csv(locFile, edgeFile):
    pass


def main():
    RoundTripRoadTrip('A', 'locations.csv', 'edges.csv', 100, 5, 'result.csv')


if __name__ == "__main__":
    main()