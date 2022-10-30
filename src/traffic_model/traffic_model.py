import numpy as np
from collections import deque


class Bus():

    _lenght = 2
    _vel_max = 3
    lane_change = False
    _station = None
    _stop_step = 20
    _slow_prob = 0.5

    def __init__(self, lane: int, position: int) -> None:
        """Initialization object  of human drive vehicle

        Args:
            lane (int): index of lane 
            position (int): position on lane
        """        
        self.lane = lane
        self.position = position
        self.__next_postion = position
        self.velosity = 0
        self.front_vehicle = None
        # station 
        self.__index_station = self.__first_station()
        self.__is_station = False
        self.__stop_count = 0
        # statistic
        self.__total_station = 0
        # for change line rule
        self.is_change = False
        self.front_adj = None
        self.behind_adj = None
        self.new_lane = None
    
    def __first_station(self) -> None:
        for i, stat in enumerate(self._station):
            if stat >= self.position+self._lenght:
                return i
        return 0
    
    def move(self, lenght, is_research=False) -> bool:
        """Move vehicle on one time step
        Realises NaSch model of vehicle behavior

        Args:
            lenght (int): total number of lane cells

        Returns:
            bool: <True> - if vehicle's position is out of lane 
            length after step, otherwise <False>
        """
        if self.front_vehicle.position == self.position and not (self is self.front_vehicle):
            raise ValueError('The cells overlay')
        distance = (self.front_vehicle.position - self.position - self._lenght) % lenght
        station_dist = int((self._station[self.__index_station] - self.position - self._lenght) % lenght)
        if station_dist == 0:
            self.__is_station = True
        
        if self.__is_station:
            if self.__stop_count <= self._stop_step:
                self.__stop_count += 1
                self.__next_postion += 0
                self.velosity = 0
                return False
            else:
                self.__is_station = False
                self.__stop_count = 0
                self.__index_station += 1
                self.__index_station %= len(self._station)
                if is_research:
                    self.__total_station += 1

        velosity = np.min([self.velosity+1, self._vel_max, distance, station_dist])
        slow = np.random.choice([0, 1], size=1, p=[1-self._slow_prob, 
                                self._slow_prob])
        self.velosity = np.max([velosity-slow, 0])
        # check station
        self.__next_postion += self.velosity
        if self.__next_postion >= lenght:
            self.__next_postion %= lenght
            return True
        return False

    def update_position(self):
        self.position = int(self.__next_postion)

    @staticmethod
    def initial_position(empty_cells, shape: tuple, amount: int):

        bus_lane = int(0)
        buses_cells = empty_cells[empty_cells.T[0] == bus_lane]
        buses_cells = buses_cells[:-1:Bus._lenght]
      
        if buses_cells.shape[0] < amount:
            raise ValueError("No free cells for buses")
        index_position = np.sort(np.random.choice(
            buses_cells.T[1], size=amount, replace=False))
        buses_position = empty_cells[index_position]
        del_index=[]
        for i in range(Bus._lenght):
            del_index.extend(index_position+i)
        empty_cells = np.delete(empty_cells, np.array(del_index), axis=0)
        return buses_position, empty_cells

    @classmethod
    def set_station(cls, station: tuple) -> None:
        cls._station = station
    
    @property
    def station(self):
        return self.__index_station

    def get_total_stations(self):
        return self.__total_station

    def __str__(self) -> str:
        return str([self.position+i for i in range(self._lenght)])


class HumanDriveVehicle():

    _lenght = 1
    _vel_max = 3
    lane_change = True
    _slow_prob = 0.5

    def __init__(self, lane: int, position: int) -> None:
        """Initialization object  of human drive vehicle

        Args:
            lane (int): index of lane 
            position (int): position on lane
        """        
        self.lane = lane
        self.position = position
        self.__next_postion = position
        self.velosity = 0
        self.front_vehicle = None
        # for change line rule
        self.is_change = False
        self.front_adj = None
        self.behind_adj = None
        self.new_lane = None
    
    def move(self, lenght, is_research=False) -> bool:
        """Move vehicle on one time step
        Realises NaSch model of vehicle behavior

        Args:
            lenght (int): total number of lane cells

        Returns:
            bool: <True> - if vehicle's position is out of lane 
            length after step, otherwise <False>
        """
        if self.front_vehicle.position == self.position and not (self is self.front_vehicle):
            raise ValueError('The cells overlay')
        distance = (self.front_vehicle.position - self.position - self._lenght) % lenght
        velosity = np.min([self.velosity+1, self._vel_max, distance])
        slow = np.random.choice([0, 1], size=1, p=[1-self._slow_prob, 
                                self._slow_prob])
        self.velosity = np.max([velosity-slow, 0])
        self.__next_postion += self.velosity
        if self.__next_postion >= lenght:
            self.__next_postion %= lenght
            return True
        return False
    
    def update_position(self):
        self.position = int(self.__next_postion)

    def __str__(self) -> str:
        return str([self.position+i for i in range(self._lenght)])

    @staticmethod
    def initial_position(empty_cells, shape: tuple, amount: int):
        """Initializing start positions of vehicles

        Args:
            empty_cells (list): list of empty cells
            shape (tuple): shape of road (lines x lenght)
            amount (int): amount of vehicles

        Returns:
            hdv_posiyion: list of vehicle's positions
            empty_cells: remaining free positions
        """        
        index_position = np.sort(np.random.choice(empty_cells.shape[0], 
                                    size=amount, replace=False))
        hdv_position = empty_cells[index_position]
        empty_cells = np.delete(empty_cells, index_position, axis=0)
        return hdv_position, empty_cells


class Model():

    __vehicle_type = [Bus, HumanDriveVehicle] # supported vehicle types
    lane_change_prob = 0.8 

    def __init__(self, road_parametrs: dict, vehicles: dict) -> None:
        """Initialization of traffic model

        Args:
            road_parametrs (dict): N_lane - numbers of lane, 
                                    N_cells - lenght of road 
            vehicles (dict): <vehicle_type <class>: amount (int), ...> 
                                - dict of vehicle types and amount 
        """
        for vehicle_type in vehicles:
            if not (vehicle_type in self.__vehicle_type):
                raise TypeError(f"Unknown vehicle type ({vehicle_type})")
        
        self.n_lane = road_parametrs['N_lane']
        self.n_cells = road_parametrs['N_cells']
        self.result = {'rho': None,'flow': None,'change_frequency': None}
        self.x_t_diagramm = {i: None for i in range(self.n_lane)}
        empty_cells = np.indices((self.n_lane, self.n_cells), dtype=np.int32).transpose(1, 2, 0).reshape(self.n_cells*self.n_lane, 2)
        vehicle_position = {}
        # position initialization of each vehicle type
        shape = (self.n_lane, self.n_cells)
        for vehicle_type, amount in vehicles.items():
            if amount < 1:
                continue
            vehicle_position[vehicle_type], empty_cells = vehicle_type.initial_position(empty_cells, shape, amount)
        self.road_model = self.__make_road_deque(vehicle_position)

    def __make_road_deque(self, veh_pos: dict):
        """Create list of deque 
        Each deque corresponds to lane and 
        contains vehicle object with sort by position
        on lane

        Args:
            veh_pos (dict): key: vehicle class, value: list of position

        Returns:
            list of deque
        """        
        road_matrix = [[None for j in range(self.n_cells)] for _ in range(self.n_lane)]
        for veh_type, positions in veh_pos.items():
            for position in positions:
                lane = position[0]
                cell = position[1]
                road_matrix[lane][cell] = veh_type(lane, cell)

        road_model = []
        for lane in road_matrix:
            obj = [cell for cell in lane if cell is not None]
            road_model.append(deque(obj))
        self.road_model=self.add_front_vehicle(road_model)
        return road_model

    def step(self, is_research=False):
        """
        One time step of model
        """   
        num_change: int = 0
        # check for lane change   
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
            num_change = self.__change_lane()
        self.road_model = self.add_front_vehicle(self.road_model)
        #vehicle move
        for lane in self.road_model:
            if not lane:
                continue
            rot = 0
            for vehicle in lane:
                rot += vehicle.move(self.n_cells, is_research=is_research)
            for vehicle in lane:
                vehicle.update_position()
            lane.rotate(rot)
        return num_change

    def get_distance(self, veh_left, veh_right):
        """Calculates distance between vehicles (empty cells)

        Args:
            veh_left (_type_): vehicle object
            veh_right (_type_): vehicle object, which is in front of veh_left
        Returns:
            int : empty cells between vehicles
        """        
        if veh_left.position == veh_right.position:
            return 0
        return (veh_right.position - veh_left.position - veh_left._lenght) % self.n_cells
    
    def __is_line_change(self, current_index, adjacent_index):
        """Checks lane change rules condition for each vehicle

        Args:
            current_index (int): index of lane for checking
            adjacent_index (int): index of adjacent lane for rebuilding

        """        
        lane = self.road_model[current_index]
        lane_adj = self.road_model[adjacent_index]

        if not lane:
            return None
        
        if not lane_adj:
            for vehicle in lane:
                vehicle.is_change = True
                vehicle.new_lane = adjacent_index
                vehicle.front_adj = None
                vehicle.behind_adj = None
            return None

        # finding neighbors in the adjacent lane
        i = 0
        flag = True
        vehicle_adj = lane_adj[i]
        for vehicle in lane:
            if vehicle.lane_change is False or vehicle.is_change is True:
                continue
            if self.is_intersection(vehicle, vehicle_adj, self.n_cells):
                vehicle.is_change = False
                continue

            while vehicle.position >= vehicle_adj.position and flag:
                i += 1
                if i == len(lane_adj):
                    i = 0
                    vehicle_adj = lane_adj[i]
                    flag = False
                    break
                vehicle_adj = lane_adj[i]
            if self.is_intersection(vehicle, vehicle_adj, self.n_cells):
                vehicle.is_change = False
                continue
            vehicle.front_adj = vehicle_adj
            vehicle.behind_adj = lane_adj[i-1]
            if self.__rule_lane_change(vehicle):
                vehicle.is_change = True
                vehicle.new_lane = adjacent_index

    def __rule_lane_change(self, vehicle):
        """Change lane rule

        Args:
            vehicle (_type_): vehicle for checking

        Returns:
            bool: True - change lane to adjacent, False - stay on lane
        """
        if vehicle.lane_change is False or vehicle.is_change is True:
            return False    
        if vehicle.position == vehicle.front_adj.position:
            return False
        if vehicle.position == vehicle.behind_adj.position:
            return False
        if (vehicle.velosity < vehicle.front_adj.velosity + self.get_distance(vehicle, vehicle.front_adj)) and \
            (vehicle.velosity >= vehicle.front_vehicle.velosity + self.get_distance(vehicle, vehicle.front_vehicle)) and \
            (vehicle.velosity > vehicle.behind_adj.velosity - self.get_distance(vehicle.behind_adj, vehicle)):
            return True
        return False

    def __change_lane(self):
        """
        Permutes transport between lanes in the model

        Return:
            int: number of vehicles that change lane 
        """
        num_change = 0  
        for i, lane in enumerate(self.road_model):
            if not lane:
                continue
            for vehicle in lane.copy():
                if not vehicle.lane_change:
                    continue
                if vehicle.is_change:
                    # avoid collision with same position    
                    for veh in self.road_model[vehicle.new_lane]:
                        if self.is_intersection(vehicle, veh, self.n_cells):
                            vehicle.is_change = False
                    if not vehicle.is_change:
                        continue
                    # random of lane change step    
                    vehicle.is_change = False
                    if np.random.rand() > self.lane_change_prob:
                        continue
                    num_change += 1
                    lane.remove(vehicle)
                    self.road_model[vehicle.new_lane].append(vehicle)

        for index, lane in enumerate(self.road_model):
            self.road_model[index] = deque(sorted(lane, key=lambda veh: veh.position))
        return num_change

    @staticmethod
    def is_intersection(vehicle_1: HumanDriveVehicle, vehicle_2: HumanDriveVehicle, n_cells):
        cells_1 = [int((vehicle_1.position+i)%n_cells) for i in range(vehicle_1._lenght)]
        cells_2 = [int((vehicle_2.position+i)%n_cells) for i in range(vehicle_2._lenght)]
        cells = cells_1 + cells_2
        if len(set(cells)) == len(cells):
            return False
        return True

    @staticmethod
    def add_front_vehicle(road):
        """Finds for each vehicle the next one 
        in the direction of move

        Args:
            road (list): list of lane (deque)
        """        
        for lane in road:
            if not lane:
                continue
            for i in range(len(lane)):
                if i == len(lane)-1:
                   lane[i].front_vehicle = lane[0]
                   continue 
                lane[i].front_vehicle = lane[i+1]
        return road

    def model_stabilization(self, n_step: int):
        """Stabilizes system

        Args:
            n_step (int): time step for stabilization
        Reccomend choose n_step = 10*n_cells
        """
        try:
            for _ in range(n_step):
                self.step()
        except Exception as e:
            self.__check_overlay(1)
            raise e

    def model_research(self, n_step: int, is_diagramm=False):
        """Method save information about cars position and velosity

        Args:
            n_step (int): time step for research
        """  
        total_change = 0
        total_velosity = np.array([0]*self.n_lane) 
        total_velosity_typed = {str(veh_type):0 for veh_type in self.__vehicle_type} 
        if is_diagramm:
            for key in self.x_t_diagramm:
                self.x_t_diagramm[key] = np.full((n_step, self.n_cells), None)
        for i in range(n_step):
            total_change += self.step(is_research=True)
            total_velosity += self.__get_sum_velosity()

            for veh_type in self.__vehicle_type:
                total_velosity_typed[str(veh_type)] += self.__get_average_velocity_typed(veh_type)

            if is_diagramm:
                self.__x_t_layer(i)

        self.result['rho'] = [len(lane)/self.n_cells for lane in self.road_model]
        self.result['flow'] = [lane_vel/ self.n_cells / n_step for lane_vel in total_velosity]
        self.result['change_frequency'] = np.sum(total_change) / self.n_cells / n_step
        self.result['station_time'] = self.station_time
        self.result['velosity_av_typed'] = {veh_type: value/n_step for veh_type, value in total_velosity_typed.items()}


    def __get_sum_velosity(self):
        """Calculate sum of vehicle 
        velocity on each lane

        Returns:
            ndarray: sum of velosity
        """
        sum_vel = np.array([0]*self.n_lane)
        for i, lane in enumerate(self.road_model):
            if not lane:
                continue
            for veh in lane:
                sum_vel[i] += veh.velosity
        return sum_vel

    def __get_average_velocity_typed(self, type_veh):
        """Calculate sum of each vehicle type 
        velocity on road
        Returns:
            ndarray: sum of velosity
        """
        sum_vel = 0
        count = 0
        for lane in self.road_model:
            if not lane:
                continue
            for veh in lane:
                if isinstance(veh, type_veh):
                    count += 1
                    sum_vel += veh.velosity
        if count == 0:
            return 0
        return sum_vel/count

    def __x_t_layer(self, step: int):
        """Add layer with information

        Args:
            step (int): time step
        """
        for index, lane in enumerate(self.road_model):
            if not lane:
                continue
            for veh in lane:
                self.x_t_diagramm[index][step][veh.position : veh.position+veh._lenght] = veh.velosity

    def get_data(self):
        return self.x_t_diagramm

    def __check_overlay(self, step):
        """
        Check collisions in model
        Use for debuging
        """
        amount = 0
        for index, lane in enumerate(self.road_model):
            if not lane:
                continue
            for i in range(len(lane)-1):    
                if lane[i].position > lane[i+1].position:
                    print(sum([len(l) for l in self.road_model]))
                    self.echo_model()
                    raise TypeError(f'usorted on step={step} on lane={index}')
        for index, lane in enumerate(self.road_model):
            if not lane:
                continue
            amount += len(lane)
            positions = []
            for veh in lane:
                positions.append(veh.position)
            if len(set(positions)) != len(lane):
                print(sum([len(l) for l in self.road_model]))
                self.echo_model()
                raise TypeError(f'overlay on step={step} on lane={index}')

    def echo_model(self):
        """
        Print vehicles' positions
        """
        for index, lane in enumerate(self.road_model):
            print(f'lane {index}:', end=" ")
            if not lane:
                print("empty")
                continue
            for veh in lane:
                print(veh, end=", ")
            print()

    @property
    def density(self):
        if self.result['rho'] is None:
            return None
        return sum(self.result['rho']) / self.n_lane

    @property
    def flow(self):
        if self.result['flow'] is None:
            return None
        return sum(self.result['flow']) / self.n_lane

    @property
    def station_time(self):
        if self.result['rho'] is None:
            return None
        total_station = 0
        buses = 0 
        for lane in self.road_model:
            if not lane:
                continue
            for veh in lane:
                if not isinstance(veh, Bus):
                    continue
                total_station += veh.get_total_stations()
                buses += 1
        if buses:
            return total_station/buses
        return None
        