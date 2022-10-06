import re
import numpy as np
from collections import deque

class Bus():

    _lenght = 2
    _vel_max = 3
    lane_change = False

    def __init__(self, lane: int, position: int) -> None:
        pass

    @staticmethod
    def initial_position(empty_cells, shape: tuple, amount: int):

        bus_lane = int(0)
        buses_cells = empty_cells[empty_cells.T[0] == bus_lane]
        buses_cells = buses_cells[::Bus._lenght]
        if buses_cells.shape[0] < amount:
            raise ValueError("No free cells for buses")
        index_position = np.sort(np.random.choice(buses_cells.T[1], size=amount, replace=False))
        buses_position = buses_cells[index_position]
        for i in range(Bus._lenght):
            empty_cells = np.delete(empty_cells, index_position+i, axis=0)
        return buses_position, empty_cells

class HumanDriveVehicle():

    _lenght = 1
    _vel_max = 3
    lane_change = True

    def __init__(self, lane: int, position: int) -> None:
        self.lane = lane
        self.position = position
        self.velosity = 0
        self.front_vehicle = None
        self.is_change = False
        self.front_adj = None
        self.behind_adj = None
        self.new_lane = None
        pass

    @staticmethod
    def initial_position(empty_cells, shape: tuple, amount: int):

        index_position = np.sort(np.random.choice(empty_cells.shape[0], size=amount, replace=False))
        hdv_position = empty_cells[index_position]
        empty_cells = np.delete(empty_cells, index_position, axis=0)
        return hdv_position, empty_cells

class Model():

    __vehicle_type = [Bus, HumanDriveVehicle]

    def __init__(self, road_parametrs: dict, vehicles: dict) -> None:
        """
        Initialization of traffic model

        Args:
            road_parametrs (dict): N_lane - numbers of lane, N_cells - lenght of road 
            vehicles (dict): <vehicle_type <class>: amount (int), ...> - dict of vehicle types and amount 
        """
        for vehicle_type in vehicles:
            if not (vehicle_type in self.__vehicle_type):
                raise TypeError(f"Unknown vehicle type ({vehicle_type})")
        
        self.n_lane = road_parametrs['N_lane']
        self.n_cells = road_parametrs['N_cells']
        empty_cells = np.indices((self.n_lane, self.n_cells), dtype=np.int32).transpose(1, 2, 0).reshape(self.n_cells*self.n_lane, 2)
        vehicle_position = {}
        # position initialization of each vehicle type
        shape = (self.n_lane, self.n_cells)
        for vehicle_type, amount in vehicles.items():
            vehicle_position[vehicle_type], empty_cells = vehicle_type.initial_position(empty_cells, shape, amount)
        self.road_model = self.__make_road_deque(vehicle_position)


    def __make_road_deque(self, veh_pos: dict):

        road_matrix = [[None for j in range(self.n_cells)] for _ in range(self.n_lane)]
        for veh_type, positions in veh_pos.items():
            for position in positions:
                lane = position[0]
                cell = position[1]
                road_matrix[lane][cell] = veh_type(lane, cell)

        road_model = []
        for lane in road_matrix:
            road_model.append(deque(set(lane).remove(None)))
        return road_model

    def step(self):

        self.__get_distance()
        #change all 'is_change' -> False
        if self.n_lane > 1:
            for index, _ in enumerate(self.road_model):
                if index == 0:
                    self.__is_line_change(index, index+1)
                elif index == self.n_lane-1:
                    self.__is_line_change(index, index-1)
                else:
                    self.__is_line_change(index, index+1)
                    self.__is_line_change(index, index-1)
        # change lane function()
        self.__get_distance()  
        #vehicle move

    def __get_distance(self):
        pass
     
    def __is_line_change(self, current_index, adjacent_index):
        lane = self.road_model[current_index]
        lane_adj = self.road_model[adjacent_index]

        if not lane:
            return None
        
        if not lane_adj:
            for vehicle in lane:
                vehicle.is_change = True
                vehicle.new_lane = adjacent_index

        #finding neighbors in the adjacent lane
        i = 0
        vehicle_adj = lane_adj[i]
        for vehicle in lane:
            if vehicle.lane_change == False or vehicle.is_change == True:
                continue
            if vehicle.position == vehicle_adj.position:
                ## ?? not necessary
                vehicle.is_change = False
                continue
            while vehicle.position > vehicle_adj.position:
                i += 1
                if i > len(vehicle_adj):
                    i = 0
                    vehicle_adj = lane_adj[i]
                    break
                vehicle_adj = lane_adj[i]
            vehicle.front_adj = vehicle_adj
            vehicle.behind_adj = lane_adj[i-1]
            if self.__check_distance(vehicle):
                vehicle.is_change = True
                vehicle.new_lane = adjacent_index

    def __check_distance(self, vehicle):
        if (vehicle.velosity + vehicle.position + vehicle._lenght > 
            vehicle.front_vehicle.velosity + vehicle.front_vehicle.position) and \
            (vehicle.front_adj.position + vehicle.front_adj.velosity > 
            vehicle.velosity + vehicle.position + vehicle._lenght) and \
            (vehicle.velosity + vehicle.position > 
            vehicle.behind_adj.velosity + vehicle.behind_adj.position + vehicle.behind_adj._lenght):
            return True
        return False
        

    def model_stabilization(self, n_step):
        pass

    def model_research(self, n_step):
        pass

    def get_data(self):
        pass

        




        
        


