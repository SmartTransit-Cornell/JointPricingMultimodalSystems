#!/usr/bin/env python
# coding: utf-8

# In[1]:


import osmnx as ox, geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry import shape
from shapely.geometry import Point
from shapely.ops import cascaded_union
from shapely.ops import unary_union
from copy import deepcopy

import time

import fiona

import networkx as nx
ox.config(log_console=True, use_cache=True)

from scipy import spatial


# In[47]:


# select subset of routes
N = 200
G_manhattan = ox.graph_from_place('Manhattan, New York, New York, USA', network_type='drive')
node_id = list(G_manhattan.nodes())
# one day contains 24 hours, so the demand is / 24
total_hours = 24

all_routes = [line.split(' ') for line in open('all_routes_1000_c5.txt')]
all_routes = [[int(float(i.strip())) for i in all_routes[j]] for j in range(len(all_routes))]
all_routes = all_routes[:N]

# create demand matrix
demand = [line.split('\n') for line in open('OD_matrix_feb_fhv.txt')]  # february demand
demand_dic = {}
for line in demand:
    outid = node_id[int(float(line[0].split(" ")[0]))]
    inid = node_id[int(float(line[0].split(" ")[1]))]
    flow = int(float(line[0].split(" ")[2]))
    demand_dic[(outid, inid)] = flow/total_hours 

demand_nodes = set([])
for key in demand_dic.keys():
    nd1 = key[0]
    nd2 = key[1]
    demand_nodes.add(nd1)
    demand_nodes.add(nd2)
    
# all starting and end nodes 
node_dic = {}  # key is node, item is lines
all_route_nodes = set([])
for ID in range(len(all_routes)):
    route = all_routes[ID]
    for n in route:
        all_route_nodes.add(n)
        if n not in node_dic.keys():
            node_dic[n] = [ID]
        else:
            node_dic[n].append(ID)

# create a new graph with route nodes
G_route = G_manhattan.subgraph(list(all_route_nodes))

# slow, find nearest transit stop for each demand node
neb_node_dic = {}
bus_nodes = [(G_route.nodes[n]['x'], G_route.nodes[n]['y']) 
             for n in G_route.nodes() if n in demand_nodes]
bus_index = [n for n in G_route.nodes() if n in demand_nodes]
tree = spatial.KDTree(bus_nodes)
for nd in demand_nodes:
    X = G_manhattan.nodes[nd]['x']
    Y = G_manhattan.nodes[nd]['y']
    #nbs = ox.distance.nearest_nodes(G_route, X, Y)
    # find 100 nearest lines
    line_dist, line_index = tree.query((X,Y), k=5)
    index1 = [bus_index[i] for i in line_index[:5]]
    for i in range(len(line_index)):
        if line_dist[i]*68.35>= 0.2 and len(index1) <= 5:
            index1.append(bus_index[line_index[i]])
    neb_node_dic[nd] = index1      


# In[48]:


nearestline_od_dic = pd.read_csv('nearest_od.csv', header=None)
new_df = nearestline_od_dic.set_index(0)
nearestline_od_dic = new_df.to_dict('index')


# In[49]:


import re
nb_od_dic = {}
for k, v in nearestline_od_dic.items():
    # k1: route
    k1 =int(re.search('\((.+?),', k).group(1))
    # k2: o or d point
    k2 =int(re.search(',(.+?)\)', k).group(1))
    value = list(v.values())[0]
    nb_od_dic[(k1, k2)]  = value


# In[50]:


# step 1: construct modes for each od
#import warnings
#slow
from itertools import product
from shapely.ops import nearest_points

mode_dic = {}
total_route_num = N

# mode dictionary 
# key: origin, destination
# item: routes (1 transfer is allowed)
for index in demand_dic.keys():
    # start, end are demand matrix index
    # origin, destination are transit route stops
    start = index[0]   
    end = index[1]
    modes = []
    origins = neb_node_dic[start]
    
    for o in origins[:-1]:
        rts = node_dic[o]
        # find the destination
        for r in rts:
            if r<=total_route_num:
            #line = LineString([(G_route.nodes[n]['x'], G_route.nodes[n]['y']) for n in all_routes[r]])
            #end_pt = Point(G_manhattan.nodes[end]['x'], G_manhattan.nodes[end]['y'])
                destination = nb_od_dic[(r, end)]
                if [r, (o, destination)] not in modes:
                    modes.append([r, (o, destination)])
          
    destinations = neb_node_dic[end]       
    for d in destinations[:-1]:
        rts = node_dic[d]
        # find the origin
        for r in rts:
            if r<=total_route_num:
                origin =  nb_od_dic[(r, start)]
                if [r, (origin, d)] not in modes:
                    modes.append([r, (origin, d)])
               
    # add two mod route   #1000 for MoD
    modes.append([1000, (origins[-1], end)])
    modes.append([1000, (start, destinations[-1])])
    mode_dic[(start, end)] = modes
    
class transit_route:
    def __init__(self):
        self.rt_id = 0
        self.route =[]
        self.edges = []
        self.length = 0
        self.graph = []
        
    def initialization(self, rt_id, graph, route):
        self.rt_id = rt_id
        self.graph = graph
        self.route = route
        self.edges = [i for i in zip(route, route[1:])]
        length = self.calculate_route_length(self.graph, route)
        self.length = length
    
    def calculate_route_length(self, G, route):
        length = 0
        if len(route)>0:
            for u,v in zip(route, route[1:]):
                try:
                    length += G.edges[u,v,0]['length']
                except KeyError:
                    length += 0
        else:
            length = 0
        length = length*0.000621371
        return length
    
class route_set:
    def __init__(self):
        self.routes = []
        self.graph = []
        self.route_id = []
           
    def initialization(self, route_list, graph):
        self.route_id = [i for i in range(len(route_list))]
        self.graph = graph
        for i in self.route_id:
            route = transit_route()
            route.initialization(i, self.graph, route_list[i])
            self.routes.append(route)   

      
all_route_set = route_set()
all_route_set.initialization(all_routes, G_manhattan)
  


# In[51]:


#for key, item in mode_dic.items:
# data structure
# od set   (s,t) for s, t in V
#   | 
# od_pair  for each o,d pair, contains all models
#   |
# mode_inf  origin, destination, all transit routes used for in this mode
#   |
# route       transit line

# route set is a set of transits and calculate its distance

class transit_seg:
    def __init__(self):
        self.rt_id = 0
        self.origin = 0
        self.destination = 0
        self.route =[]
        self.edges = []
        self.length = 0
        self.graph = []
    
    def calculate_route_length(self, route):
        length = 0
        G = self.graph
        if len(route)>0:
            for u,v in zip(route, route[1:]):
                try:
                    length += G.edges[u,v,0]['length']
                except KeyError:
                    length += 0
        else:
            length = 0
        length = length*0.000621371   # meters to miles
        return length
        
    def initialization(self, rt_id, route_set, od, graph):
        self.origin = od[0]
        self.destination = od[1]
        self.rt_id = rt_id
        self.graph = graph
        full_route = route_set.routes[rt_id].route
        subroute1= full_route[len(full_route)-full_route[::-1].index(self.origin):
                              full_route.index(self.destination)]
        full_route2 = list(reversed(full_route))
        subroute2 = full_route[len(full_route2)-full_route2[::-1].index(self.origin):
                              full_route2.index(self.destination)]
        if len(subroute1)>0  and len(subroute1)>len(subroute2):
            self.route = subroute1
        elif len(subroute2)>0  and len(subroute2)>len(subroute1):
            self.route = subroute2
        if len(self.route)>0:
            self.edges =  [i for i in zip(self.route, self.route[1:])]
        self.length = self.calculate_route_length(self.route)

class mode_inf:
    def __init__(self):
        self.route = []
        self.length = 0
        self.edges = []
        # type = 0 -- bus/bus+walk; type = 1 hybrid (bus+mod); 
        # type = 9 -- MoD only; type = 10 -- walking only
        self.type = 0
        self.travel_time = 0
        # in manhattan, average walking speed is 3 mph and driving is 7.5 mph, bus is 6.7 mph
        self.walk_speed = 3.0
        self.drive_speed = 7.5
        self.bus_speed = 6.7
        # origin and destination are transit stops
        self.origin = 0
        self.destination = 0
        # start and end are actual nodes
        self.start = 0
        self.end = 0
        
        self.route_ind = 0
        self.cost = 0
        self.base_value = 0
        # -------------------------friction cost --------------
        self.friction_cost = 1
    
        self.mod_cost = 2.0
        # --------------------------MoD cost --------------
        self.walking_threshold = 0.2
        self.graph = []
        
    def calculate_base_value(self, G, route):
        #  $18.6 per hour for willingness to pay
        # type 1 is hybrid, type 0 is transit
        v_hour = 18.6 
        travel_time = self.length/self.bus_speed
        cost = 0
        # FM distance
        y1 = G.nodes[self.start]['y']
        x1 = G.nodes[self.start]['x']
        y2 = G.nodes[self.origin]['y']
        x2 = G.nodes[self.origin]['x']
        #distance 1 and 2 are FM/LM distance in miles
        dist1 = ox.distance.euclidean_dist_vec(y1, x1, y2, x2)*68.35*np.sqrt(2)
        
        if dist1>self.walking_threshold:
            # average driving speed is 10 mph
            travel_time += dist1/self.drive_speed
            cost += self.friction_cost
            self.type = 1
        else:
            # average walking speed is 3 mph
            travel_time += dist1/self.walk_speed 
            
        
        # LM distance
        y1 = G.nodes[self.end]['y']
        x1 = G.nodes[self.end]['x']
        y2 = G.nodes[self.destination]['y']
        x2 = G.nodes[self.destination]['x']
        dist2 = ox.distance.euclidean_dist_vec(y1, x1, y2, x2)*68.35*np.sqrt(2)
        # add transfer cost
        
        if dist2>self.walking_threshold:
            travel_time += dist2/self.drive_speed
            cost += self.friction_cost
            self.type = 1
        else:
            # average walking speed is 3 mph
            travel_time += dist2/self.walk_speed
        
        if self.type !=1:
            # walking
            self.type = 0
        
        self.travel_time = travel_time
        self.length += dist1 
        self.length += dist2
        
        value = v_hour * (self.length/self.walk_speed - self.travel_time)
        value -= cost
        if self.type == 1:
            value += 1
        # if the value of trip is less than 0, the trip is not feasible
        value = max(0, value)
        self.value = value
        return value
    
    def calculate_MoD_value(self, G, od):
        #  $18.6 per hour for willingness to pay
        v_hour = 18.6 
        orig, dest = od
        value = 0
        try:
            route = nx.shortest_path(G, orig, dest, 'travel_time')
            self.length = self.calculate_route_length(route)
            if self.length >2*self.walking_threshold:
                self.travel_time = self.length / self.drive_speed
                value = v_hour * (self.length/self.walk_speed - self.travel_time)
                self.type = 9
            else:
                # value of walking is 0
                self.type = 10
                # walking is not included in modes
            
        except nx.NetworkXNoPath:
            route = [] 
        # if the value of trip is less than 0, the trip is not feasible
        value = max(0, value)
        # base value
        if self.type == 9:
            # $2 for mod
            value += 2
        return route, value
    
    def calculate_MoD_cost(self,route):
        #  $18.6 per hour for willingness to pay
        self.length = self.calculate_route_length(route)
        cost = self.length*self.mod_cost
        return cost
    
    def calculate_route_length(self, route):
        length = 0
        G = self.graph
        if len(route)>0:
            for u,v in zip(route, route[1:]):
                try:
                    length += G.edges[u,v,0]['length']
                except KeyError:
                    length += 0
        else:
            length = 0
        length = length*0.000621371   # meters to miles
        return length
    
    
    def calculate_FMLM_cost(self, G):
        # only using MoD incurs additional cost of $2/mi
        cost = 0
        # FM distance
        y1 = G.nodes[self.start]['y']
        x1 = G.nodes[self.start]['x']
        y2 = G.nodes[self.origin]['y']
        x2 = G.nodes[self.origin]['x']
        #distance 1 and 2 are FM/LM distance in miles
        dist1 = ox.distance.euclidean_dist_vec(y1, x1, y2, x2)*68.35*np.sqrt(2)
        
        if dist1>self.walking_threshold:
            cost += dist1*self.mod_cost
        
        # LM distance
        y1 = G.nodes[self.end]['y']
        x1 = G.nodes[self.end]['x']
        y2 = G.nodes[self.destination]['y']
        x2 = G.nodes[self.destination]['x']
        dist2 = ox.distance.euclidean_dist_vec(y1, x1, y2, x2)*68.35*np.sqrt(2)
        # add transfer cost
        
        if dist2>self.walking_threshold:
            cost += dist2*self.mod_cost
        return cost
        
    
    def update_inf(self, input_ls):
        rt_id, nd_ls, route_set, graph = input_ls
        self.graph = graph
        self.route_ind = rt_id
        self.start = nd_ls[0]
        self.end = nd_ls[-1]
        # if rt_id is bus, create a transit seg 
        if rt_id<1000:
            bus = transit_seg()
            self.origin = nd_ls[1]
            self.destination = nd_ls[2]
            bus.initialization(rt_id, route_set,(self.origin, self.destination),graph)
            self.length += bus.length
            self.route = bus.route
            self.edges = bus.edges
            self.base_value = self.calculate_base_value(self.graph, self.route)
            self.cost = self.calculate_FMLM_cost(self.graph)
            return True
        else:    # only MoD route
            self.route, self.base_value = self.calculate_MoD_value(self.graph, (self.start, self.end))
            self.cost = self.calculate_MoD_cost(self.route)
            if self.type==10:
                return False
            else:
                return True
        

class od_pair:
    def __init__(self):
        self.start = 0
        self.end = 0
        self.demand = 0
        self.all_routes = []
        self.graph = []
        self.md_inf_ls = []
        self.index_ls = []
        
    def update_od(self, input_ls):
        # md_dic, dmd_dic, nd_dic, md_stops, all_routes, graph
        self.start, self.end = input_ls[0]
        self.modes = input_ls[1]
        self.demand = input_ls[2]
        self.all_routes = input_ls[3]
        self.graph = input_ls[4]
        self.process_data()
        #self.print_modes()
        
    def process_data(self):
        md_ls = []
        for m in self.modes:
            rt_id = m[0]
            origin, destination = m[1]
            if rt_id <= total_route_num or rt_id == 1000:
                md = mode_inf()
            #print([self.start, origin, destination, self.end])
                input_ls = [rt_id, [self.start, origin, destination, self.end], 
                            self.all_routes, self.graph]
                flag1 = md.update_inf(input_ls)
                if flag1==True:
                    md_ls.append(md)
        self.md_inf_ls = md_ls
        self.index_hybrid_ls = [md_ls.index(md) for md in self.md_inf_ls if md.type == 1]
    
    def print_modes(self):
        
        print(self.modes)
        #for key, item in self.modes.items():
        #    length = [i.length for i in item] 
        #    print(str(key)+': '+ str(length))
        
    
class od_set:
    def __init__(self):
        self.ods = []
        self.graph = []
        self.hybrid_ind = {}
        
    def update_set(self, md_dic, dmd_dic, nd_dic, routes, graph):
        t0 = time.time()
        self.graph = graph
        for key in md_dic.keys():
            demand = dmd_dic[key]
            if demand>0:
                od = od_pair()
                s = key[0]
                t = key[1]
                modes = md_dic[key]
                # format of mode: route (1000 for mod), (o, d) 
                input_ls = [[s,t], modes, demand, routes, graph]
                od.update_od(input_ls)  
                self.ods.append(od)
                self.hybrid_ind[(od.start, od.end)] = od.index_hybrid_ls
        t1 = time.time()
        runtime = t1 - t0
        print('total runtime is ' + str(runtime))
        #self.print_summary()
    
    def print_summary(self):
        #print(self.hybrid_ind)
        for od in self.ods:
            print('trip is:' + str([od.start, od.end]))
            print('number of modes is ' + str(len(od.md_inf_ls)))
            print('modes are:')
            md_ls = [md.type for md in od.md_inf_ls]
            
            print(str(md_ls))
            v_ls = [md.base_value for md in od.md_inf_ls]
            print('values are:')
            print(str(v_ls))
            c_ls = [md.cost for md in od.md_inf_ls]
            print('costs are:')
            print(str(c_ls))
            print('----------------------------------')
        
OD_set = od_set()
OD_set.update_set(mode_dic, demand_dic, neb_node_dic, all_route_set, G_manhattan)


# In[52]:


# list all edges covered per od
def compute_od_eg(ODs):
    od_edge = {}
    for od in ODs.ods:
        key = (od.start, od.end)
        edges = set([])
        for md in od.md_inf_ls:
            if md.route_ind<1000:
                edges.update(md.edges)
        od_edge[key] = list(edges)
    return od_edge
OD_edge_dic = compute_od_eg(OD_set)

def compute_eg_od(odeg_dic, graph):
    new_dict = {}
    for edge in graph.edges():
        od_included = []
        for key, items in odeg_dic.items():
            if edge in items:
                od_included.append(key)
        new_dict[edge] = od_included
        s = edge[0]
        t = edge[1]
        new_dict[(t,s)] = od_included
    return new_dict
    #    print(edge)
edge_OD_dic = compute_eg_od(OD_edge_dic, G_route)  


# In[53]:


# edge_route_mode list
class edge_inf:
    def __init__(self):
        self.s = 0
        self.t = 0
        self.transit_lines = []
        self.ods = []
        self.big_M = 1000000
        
    def update(self, edge, ODs, route_set, egod_dic):
        self.s = edge[0]
        self.t = edge[1]
        time0=time.time()
        self.ods = egod_dic[edge]
        time1=time.time()
        runtime = time1 - time0
        #print('od check' + str(runtime))
        for rt in route_set.routes:
            flag = self.check_edge_in_route(rt)
            if flag == self.big_M:
                pass
            else:
                self.transit_lines.append(flag)
        time2 = time.time()
        runtime = time2 - time1
        #self.print_results()
        #print('route check' + str(runtime))
           
    def check_edge_in_od(self, OD):
        if (self.s, self.t) in edge_set or (self.t, self.s) in edge_set:
            return (OD.start, OD.end)
        else:
            return self.big_M
        
    def check_edge_in_route(self, route):
        if (self.s, self.t) in route.edges:
            return route.rt_id
        else:
            return self.big_M
    
    def print_results(self):
        print('edge is:' + str((self.s, self.t)))
        print('total ods is ' + str(len(self.ods)))
        print('total routes is ' + str(len(self.transit_lines)))
        
        
class edge_transit_set:
    def __init__(self):
        self.calculated_edges = set([])
        self.edges = []
        self.graph = []
        self.route_set = []
        
    def initialization(self, graph, route_set, OD, egod_dic):
        time0 = time.time()
        self.graph = graph
        self.route_set = route_set
        
        for eg in self.graph.edges():
            self.calculated_edges.add(eg)
            eg_inf = edge_inf()
            eg_inf.update(eg, OD, route_set, egod_dic)
            self.edges.append(eg_inf)
        time1 = time.time()
        runtime = time1 - time0
        #print('total runtime is ' + str(runtime))

           
'''
        for rt in self.route_set.routes:
            for eg in rt.edges:
                s = eg[0]
                t = eg[1]
                if eg not in self.calculated_edges:
                    self.calculated_edges.add(eg)
                    eg_inf = edge_inf()
                    eg_inf.update(eg, OD, route_set)
                    self.edges.append(eg_inf)
'''
        
et_set = edge_transit_set()
et_set.initialization(G_route, all_route_set, OD_set, edge_OD_dic)


# In[54]:


import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
# step 2: solve the problem CP
# parameters and source:
'''
data source: 
http://www.portlandfacts.com/top10bus.html
kappa is capacity of NYC mta bus: 81.7
maintenance cost is $20.65/mile 
transit fare is 0.90/trip
uber trip per mile is $2.00/mile


'''

class MIP_solver:
    def __init__(self):
        self.max_time = 10
        self.theta = np.arange(2)
        self.freq = np.arange(2)
        self.vars = []
        self.model = []
        self.OD = []
        self.routes = []
        self.et_set = []
        self.M_value = {}
        
        self.od_dic = {}
        self.line_dic = {}
        self.od_id_dic = {}
        
        # variables
        self.Z = []
        self.PHI = {}
        self.Y = {}
        
        # parameters 
        # number of displayed modes
        self.k = 10
        # split of high and low value ratio
        self.demand_ratio = 0.75
        # operating cost
        self.oper_cost = 20.65
        # vehicle capacity
        self.veh_cap = 81.7
        
    
    def construct_edge_od_dic(self):
        edge_od_dic = {}
        for eg in self.et_set.edges:
            edge = (eg.s, eg.t)
            ods = eg.ods
            edge_od_dic[edge] = ods
        self.od_dic = edge_od_dic
        
    def construct_edge_line_dic(self):
        line_dic = {}
        for eg in self.et_set.edges:
            edge = (eg.s, eg.t)
            lines = eg.transit_lines 
            line_dic[edge] = lines
        self.line_dic = line_dic
    
    def construct_od_id_dic(self):
        od_dic = {}
        for i in range(len(self.OD.ods)):
            key = (self.OD.ods[i].start, self.OD.ods[i].end)
            od_dic[key] = i
        self.od_id_dic = od_dic
        
    def initialize(self, od_matrix, routes, edge_transit_set, graph):
        self.model = gp.Model('MIP')
        self.OD = od_matrix
        self.routes = routes
        self.k = 10
        self.demand_ratio = 0.75
        self.et_set = edge_transit_set
        self.construct_edge_od_dic()
        self.construct_edge_line_dic()
        
        # Phi, Z, Y
        self.print_variables()
        
        t0 = time.time()
        # add variable for each line
        
        # cost of setting up lines
        cz_ls = [self.oper_cost*rt.length for rt in self.routes.routes]
        Z = self.model.addVars(self.routes.route_id, vtype = GRB.BINARY, name = 'Z')
        
        # add variable for each mode per od
        for od_ind in range(len(self.OD.ods)):
            OD_pair = od_matrix.ods[od_ind] 
            local_key  = (OD_pair.start, OD_pair.end)
            name = 'Phi' + str(local_key)
            self.M_value[od_ind] = [m.base_value*(1+0.5*theta) - m.cost
                                    for m in OD_pair.md_inf_ls 
                                    for theta in self.theta]
            #print(self.M_value)
            mode_ls = np.arange(len(OD_pair.md_inf_ls))
            mode_type_ls = np.arange(len(OD_pair.md_inf_ls)*len(self.theta))
            # phi(od)
            self.PHI[od_ind] = self.model.addVars(mode_type_ls, vtype = GRB.CONTINUOUS, 
                                                  lb = 0, ub = 1000, name = name)
            # y(od)
            name = 'Y'+str(od_ind)
            self.Y[od_ind] = self.model.addVars(mode_ls, vtype = GRB.BINARY, name = name)
            
        # add constraints
            # demand constraint
            for th in self.theta:
                RHS = OD_pair.demand * (th*self.demand_ratio + (1-th)*(1-self.demand_ratio))
                if len(OD_pair.md_inf_ls)>0:
                    phi_type_ind = [i for i in np.arange(th, len(self.theta)*len(OD_pair.md_inf_ls)+th,
                                                         len(self.theta)) ]
                    phi_type = [self.PHI[od_ind][j] for j in phi_type_ind]
                    LHS = np.sum(phi_type)
                    self.model.addConstr(LHS <= RHS, name = 'demand')
               
            # joint mode constraint
            for m in range(len(OD_pair.md_inf_ls)):
                phi_mode_ind = [j for j in range(m*len(self.theta), 
                                                 (1+m)*len(self.theta))]
                phi_mode = [self.PHI[od_ind][j] for j in phi_mode_ind]
                LHS = np.sum(phi_mode)
                # assuming Theta_m = Theta
                RHS = OD_pair.demand *self.Y[od_ind][m]
                self.model.addConstr(LHS <= RHS, name = 'joint')
            
            # display capacity constraint
            LHS = np.sum(self.Y[od_ind][j] for j in range(len(self.Y[od_ind])))
            self.model.addConstr(LHS <= self.k, name = 'display')
        
        
        # frequency-setting constraint
        # other constraints are for each od, and this constraint is for each transit line
        self.construct_od_id_dic()
        
        for od_ind in range(len(self.OD.ods)):
            OD_pair = od_matrix.ods[od_ind] 
          
        for rt_id in self.routes.route_id:
            RHS = self.veh_cap * Z[rt_id]
            edges = [i for i in self.routes.routes[rt_id].edges if i in graph.edges()]
            for edge in edges:
                lines = self.line_dic[edge]
                ods = self.od_dic[edge]
                LHS = 0
                for od in ods:
                    od_index = self.od_id_dic[od]
                    OD_pair = od_matrix.ods[od_index]  
                    # recall 
                    # mode_type_ls = np.arange(len(OD_pair.modes)*len(self.theta))
                    # find all modes including lines
                    md_count = 0
                    lhs_index = []
                    for md in OD_pair.md_inf_ls:
                        route_id = md.route_ind
                        if route_id<1000 and route_id == rt_id:
                            for th in self.theta:
                                lhs_index.append(len(self.theta)*md_count+th)
                        md_count += 1
                    for i in lhs_index:
                        LHS += self.PHI[od_index][i]
                self.model.addConstr(LHS <= RHS, name = 'capacity')
                
                     
        # set objective
        obj1 = 0
        for i in range(len(self.OD.ods)):
            obj1 += np.sum(self.PHI[i][j]*self.M_value[i][j] for j in range(len(self.PHI[i])))
        obj2 = np.sum(Z[i]*cz_ls[i] for i in self.routes.route_id)
        self.model.setObjective(obj1 - obj2, GRB.MAXIMIZE)
        
        t1 = time.time()
        # optimize
        self.model.optimize()  
        
        t2 = time.time()
        times = [t0,t1,t2]
        # display   
        try:
            self.print_results(times)
        except AttributeError:
            pass
        
    def print_variables(self):
        var_dim_ls = [(len(self.OD.ods), len(self.OD.ods[0].md_inf_ls)*len(self.theta)),
                      len(self.routes.routes),
                      (len(self.OD.ods), len(self.OD.ods[0].md_inf_ls))
                     ]
        print('Phi dimension is:' + str(var_dim_ls[0]))
        print('Z dimension is:' + str(var_dim_ls[1])) 
        print('Y dimension is:' + str(var_dim_ls[2]))
    
            
    def print_results(self, times):
        t0,t1,t2 = times
        global Global_Z 
        global Global_Y
        global Global_Phi
        print('preprocess time is '+str(t1-t0))
        print('solution time is ' + str(t2-t1))
        print('Obj: %g' % self.model.objVal)
        var_flag = 1
        if var_flag>0:
            for v in self.model.getVars():
                if 'Z' in  v.varName and v.x>0:
                    print('%s %g' % (v.varName, v.x))
                    #print('%s %g' % (v.varName, v.x))
        Z_dic = {}
        Y_dic = {}
        Phi_dic = {}
        for v in self.model.getVars():
            if 'Z' in  v.varName:
                Z_dic[v.index] = v.x

            if 'Y' in v.varName:
                key1 = int(v.varName[v.varName.find('Y')+1:v.varName.rfind('[')])
                key2 = int(v.varName[v.varName.find('[')+1:v.varName.rfind(']')])
                Y_dic[(key1, key2)] = v.x
                
            if 'Phi' in v.varName:
                key1 = int(v.varName[v.varName.find('(')+1:v.varName.rfind(',')])
                key2 = int(v.varName[v.varName.find(',')+1:v.varName.rfind(')')])
                key3 = int(v.varName[v.varName.find('[')+1:v.varName.rfind(']')])
                Phi_dic[(key1, key2, key3)] = v.x
            
        Global_Z = Z_dic
        Global_Y = Y_dic
        Global_Phi = Phi_dic
        
        name = 'N'+str(N)+'transfer'+str('1')+'mod'+'1'+'.mst'
        self.model.write(name)
                      
solver1 = MIP_solver() 
solver1.initialize(OD_set, all_route_set, et_set, G_route)


# In[55]:


# print ratio of hybrid and non-hybrid trips
od_key = {}
for key, item in Global_Phi.items():
    key1 = key[0]
    key2 = key[1]
    key3 = key[2]
    if (key1, key2) not in od_key.keys():
        od_key[(key1, key2)] = [key3]
    else:
        od_key[(key1, key2)].append(key3)
    # find 
    
demand_hybrid = 0
demand_MoD = 0
for key1, key2 in od_key.items():
    # hybrid modes 
    hybrid = OD_set.hybrid_ind[key1]
    for k in key2:
        key = (key1[0], key1[1], k)
        if k in hybrid:
            demand_hybrid += Global_Phi[key]
        else:
            demand_MoD += Global_Phi[key]
    #for k in key2:
        #key = (key1[0], key1[1], k)
        #if k == max(key2) or k == max(key2)-1:
        #    demand_MoD += Global_Phi[key]
        #else:
        #    demand_hybrid += Global_Phi[key]
print('ratio of hybrid:')
print(demand_hybrid / (demand_MoD+demand_hybrid))
print((demand_MoD+demand_hybrid) / sum(list(demand_dic.values())) )


# In[56]:


# dual problem
#import optimal z and y
from itertools import product
class Dual_solver:
    def __init__(self):
        self.max_time = 10
        self.theta = np.arange(2)
        self.freq = np.arange(2)
        self.vars = []
        self.model = []
        self.OD = []
        self.routes = []
        self.et_set = []
        self.M_value = {}
        
        self.od_dic = {}
        self.line_dic = {}
        self.od_id_dic = {}
        
        # variables
        self.Z = []
        self.Y = []
        self.U = {}
        self.Mu = {}
        self.Gamma = {}
        
        # parameters 
        # number of displayed modes
        self.k = 10
        # split of high and low value ratio
        self.demand_ratio = 0.75
        # operating cost
        self.oper_cost = 20.65
        # vehicle capacity
        self.veh_cap = 81.7
        
        self.price_ls = {}
        self.price_index_ls = {}
        self.total_price = 0
        
        
    def construct_edge_od_dic(self):
        edge_od_dic = {}
        for eg in self.et_set.edges:
            edge = (eg.s, eg.t)
            ods = eg.ods
            edge_od_dic[edge] = ods
        self.od_dic = edge_od_dic
        
    def construct_edge_line_dic(self):
        line_dic = {}
        for eg in et_set.edges:
            edge = (eg.s, eg.t)
            lines = eg.transit_lines
            line_dic[edge] = lines
        self.line_dic = line_dic
    
    def construct_od_id_dic(self):
        od_dic = {}
        for i in range(len(self.ODs.ods)):
            key = (OD.ods[i].start, OD.ods[i].end)
            od_dic[key] = i
        self.od_id_dic = od_dic
        
    def initialize(self, od_matrix, routes, edge_transit_set, all_routes):
        self.model = gp.Model('MIP')
        self.ODs = od_matrix
        self.routes = routes
        self.k = 10
        self.demand_ratio = 0.75
        self.et_set = edge_transit_set
        self.construct_edge_od_dic()
        self.construct_edge_line_dic()
        
        t0 = time.time()
        # add variable for each line
        cz_ls = [self.oper_cost*rt.length for rt in self.routes.routes]
        self.Z = Global_Z
        self.Y = Global_Y
        
        
        # add variable for each mode per od
        for od_ind in range(len(self.ODs.ods)):
            OD_pair = od_matrix.ods[od_ind] 
            local_key  = (OD_pair.start, OD_pair.end)
            self.M_value[od_ind] = [m.base_value*(1+0.5*theta) - m.cost
                                    for m in OD_pair.md_inf_ls 
                                    for theta in self.theta]
            mode_ls = np.arange(len(OD_pair.md_inf_ls))
            mode_type_ls = np.arange(len(OD_pair.md_inf_ls)*len(self.theta))
            
            # U(od)(theta)
            name = 'U'+str(od_ind)
            self.U[od_ind] = self.model.addVars(self.theta, vtype = GRB.CONTINUOUS, lb=0,
                                                name = name)
            name = 'Gamma'+str(od_ind) 
            self.Gamma[od_ind] = self.model.addVars(mode_ls, vtype = GRB.CONTINUOUS, lb=0,
                                                    name = name)
            
        for rt_id in self.routes.route_id:
            edges = self.routes.routes[rt_id].edges
            name = 'Mu'+str(rt_id)
            self.Mu[rt_id] = self.model.addVars(range(len(edges)), 
                                                vtype = GRB.CONTINUOUS, lb=0, name=name)
            
        # add constraints
        for od_ind in range(len(self.ODs.ods)):
            OD_pair = od_matrix.ods[od_ind] 
            for th in self.theta:
                od_th_ind = []
                LHS = self.U[od_ind][th]
                for md in range(len(OD_pair.md_inf_ls)):
                    mode = OD_pair.md_inf_ls[md]
                    RHS = self.M_value[od_ind][md*len(self.theta)+th]
                    Gamma = list(self.Gamma[od_ind].select(md))
                    LHS += Gamma[0]
                    edges = mode.edges
                    if mode.type < 9: 
                        rt_id = mode.route_ind 
                        for e_id in range(len(edges)):
                            LHS += self.Mu[rt_id][e_id]
                            od_th_ind.append((rt_id, e_id))
                         # find route, od to match mu 
                    self.model.addConstr(LHS >= RHS, name = 'Dual')
                self.price_index_ls[(od_ind, th)] = od_th_ind
            
                                                        
        # set objective
        obj1 = 0
        obj3 = 0
        for od_ind in range(len(self.ODs.ods)):
            OD_pair = od_matrix.ods[od_ind] 
            theta = 0
            obj1 += OD_pair.demand * self.demand_ratio * self.U[od_ind][theta]
            obj1 += OD_pair.demand * (1-self.demand_ratio) * self.U[od_ind][theta+1]
            for j in range(len(OD_pair.md_inf_ls)):
                try:
                    obj3 += self.Y[(od_ind, j)]*self.Gamma[od_ind][j]*OD_pair.demand
                except KeyError:
                    pass
            
        obj2 = 0 
        for rt_id in self.routes.route_id:
            edges = self.routes.routes[rt_id].edges 
            for i in range(len(edges)):
                obj2 += self.veh_cap*self.Z[rt_id]*self.Mu[rt_id][i]
            
        self.model.setObjective(obj1 + obj2 + obj3, GRB.MINIMIZE)
        
        t1 = time.time()
        # optimize
        self.model.optimize()  
        
        t2 = time.time()
        times = [t0,t1,t2]
        
        setup_cost = sum([cz_ls[i]* Global_Z[i] for i in range(len(cz_ls))])
        # display   
        try:
            self.print_results(times, od_matrix, setup_cost)
        except AttributeError:
            pass
        
        self.compute_dual_price()
        
        
    def compute_dual_price(self):
        #total prices
        obj = 0 
        for od_ind in range(len(self.ODs.ods)):
            OD_pair = self.ODs.ods[od_ind] 
            modes = OD_pair.md_inf_ls
            #print('od is %g', %od_ind)
            for m in range(len(modes)):
                #print('mode is %g', %m)
                mode = modes[m]
                cost = mode.cost
                gamma_m = Global_Gamma[(od_ind,m)]
                price = cost + gamma_m
                if mode.type < 9: 
                    rt_id = mode.route_ind 
                    edges = mode.edges
                    for e_id in range(len(edges)):
                        price += Global_Mu[(rt_id, e_id)]
                #print('price is ' + str(price))
                self.price_ls[(od_ind, m)] = price
            for th in self.theta:
                start = OD_pair.start
                end = OD_pair.end
                indices = [len(self.theta)*i for i in range(len(modes))]
                obj += sum([Global_Phi[(start, end, i)]*
                            self.price_ls[(od_ind, np.floor(i/len(self.theta)))] 
                            for i in indices])
        self.total_price = obj
        print('total_price ' + str(self.total_price))
                
                
                
            
    def print_results(self, times, od_matrix, setup_cost): 
        global Global_U 
        global Global_Mu
        global Global_Gamma
        
        t0,t1,t2 = times
        print('preprocess time is '+str(t1-t0))
        print('solution time is ' + str(t2-t1))
        # compute average price
        Mu_dic = {}
        Gamma_dic = {}
        for v in self.model.getVars():
            if 'Mu' in  v.varName:
                Mu_dic[v.index] = v.x
            if 'Gamma' in v.varName:
                Gamma_dic[v.index] = v.x
        price = sum(Mu_dic.values()) + sum(Gamma_dic.values()) 
        
        total_demand = 0
        for od_ind in range(len(self.ODs.ods)):
            OD_pair = od_matrix.ods[od_ind] 
            total_demand += OD_pair.demand
            for j in range(len(OD_pair.md_inf_ls)):
                mode = OD_pair.md_inf_ls[j]
                price += self.Y[(od_ind, j)]*mode.cost
        avg_price = price/total_demand
        print('average price is' + str(avg_price))
        print('objective is: %g' % self.model.objVal)
        total_obj = self.model.objVal+setup_cost
        print('primal objective is: %g' % total_obj)
        
        U_dic = {}
        Mu_dic = {}
        Gamma_dic = {}
        for v in self.model.getVars():
            if 'U' in  v.varName:
                key1 = int(v.varName[v.varName.find('U')+1:v.varName.rfind('[')])
                key2 = int(v.varName[v.varName.find('[')+1:v.varName.rfind(']')])
                U_dic[(key1, key2)] = v.x

            if 'Mu' in v.varName:
                key1 = int(v.varName[v.varName.find('u')+1:v.varName.rfind('[')])
                key2 = int(v.varName[v.varName.find('[')+1:v.varName.rfind(']')])
                Mu_dic[(key1, key2)] = v.x
                
            if 'Gamma' in v.varName:
                key1 = int(v.varName[v.varName.find('ma')+2:v.varName.rfind('[')])
                key2 = int(v.varName[v.varName.find('[')+1:v.varName.rfind(']')])
                Gamma_dic[(key1, key2)] = v.x
            
        Global_U = U_dic
        Global_Mu = Mu_dic
        Global_Gamma = Gamma_dic

             
solver1 = Dual_solver() 
solver1.initialize(OD_set, all_route_set, et_set, all_routes)

