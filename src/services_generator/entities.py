"""Entities for the services generator module."""

import ast
import datetime
import numpy as np
import os
import pandas as pd
from pathlib import Path
import random
import yaml

from src.robin.supply.entities import Station, Corridor, Seat, TimeSlot, TSP, Line, RollingStock, Service
from src.robin.supply.utils import convert_tree_to_dict, set_stations_ids, get_time
from src.services_generator.utils import _get_distance
from src.robin.scraping.utils import station_to_dict, seat_to_dict, corridor_to_dict, line_to_dict, \
    rolling_stock_to_dict, time_slot_to_dict, tsp_to_dict, service_to_dict
from src.services_generator.utils import _get_end_time, _get_start_time, _to_station, _build_service

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Tuple


class ServiceGenerator:
    """
    Generate random services from a YAML config file

    Attributes:
        stations (Dict[str, Station]): Dict of stations
        corridors (Dict[str, Corridor]): Dict of corridors
        seats (Dict[str, Seat]): Dict of seats
        rolling_stock (Dict[str, RollingStock]): Dict of rolling stocks
        tsps (Dict[str, TSP]): Dict of TSPs
        lines (Dict[str, Line]): Dict of lines
        time_slots (Dict[str, TimeSlot]): Dict of time slots
        services (List[Service]): List of services generated in the current session

    Methods:
        generate() (List[Service])
    """

    def __init__(self,
                 supply_config_path: Path,
        ) -> None:
        """
        Initialize the ServiceGenerator object

        Args:
            supply_config_path (Path): Path to the config file
        """
        with open(supply_config_path, 'r') as file:
            data = yaml.load(file, Loader=yaml.CSafeLoader)

        self.stations = self._get_stations(data, key='stations')
        self.time_slots = self._get_time_slots(data, key='timeSlot')
        self.corridors = self._get_corridors(data, self.stations, key='corridor')
        self.lines = self._get_lines(data, self.corridors, key='line')
        self.seats = self._get_seats(data, key='seat')
        self.rolling_stock = self._get_rolling_stock(data, self.seats, key='rollingStock')
        self.tsps = self._get_tsps(data, self.rolling_stock, key='trainServiceProvider')
        self.services = []

    def generate(self,
                 file_name: Path,
                 path_config: Path,
                 n_services: int = 1,
                 seed: int = None
        ) -> List[Service]:
        """
        Generate a list of services

        Args:
            file_name (Path): Name of the output file
            path_config (Path): Path to the config file
            n_services (int, optional): Number of services to generate. Defaults to 1.
            seed (int, optional): Seed for the random number generator. Defaults to None.

        Returns:
            List[Service]: List of services
        """
        if seed is not None:
            self.set_seed(seed)
        self._set_config(path_config)

        services = []
        for _ in range(n_services):
            services.append(self._generate_service())

        self.services += services
        self.save_to_yaml(services, file_name)
        return services

    def save_to_yaml(self, services: List[Service], file_name: Path) -> None:
        """
        Save the data to a yaml file

        Args:
            services (List[Service]): List of Service objects
            file_name (Path): Name of the output file


        Returns:
            None
        """
        rolling_stocks = list(set([rs for tsp in self.tsps.values() for rs in tsp.rolling_stock]))

        yaml_dict = {'stations': [station_to_dict(stn) for stn in self.stations.values()],
                     'seat': [seat_to_dict(s) for s in self.seats.values()],
                     'corridor': [corridor_to_dict(corr) for corr in self.corridors.values()],
                     'line': [line_to_dict(ln) for ln in self.lines.values()],
                     'rollingStock': [rolling_stock_to_dict(rs) for rs in rolling_stocks],
                     'trainServiceProvider': [tsp_to_dict(tsp) for tsp in self.tsps.values()],
                     'timeSlot': [time_slot_to_dict(s) for s in self.time_slots.values()],
                     'service': [service_to_dict(serv) for serv in services]}

        self._write_to_yaml(file_name, yaml_dict)

    def _generate_service(self) -> Service:
        """
        Generate a random service

        Returns:
            Service: Service object
        """
        line = self._get_random_line()
        time_slot = self._get_random_time_slot()
        tsp = self._get_random_tsp()
        rs = self._get_random_rs(tsp)
        date = self._get_random_date()
        prices = self._get_random_prices(line, rs, tsp)  # prices: Dict[Tuple[str, str], Dict[Seat, float]]
        service = _build_service(date, line, time_slot, tsp, rs, prices)

        self.services.append(service)
        return service

    def _set_config(self, path_config: Path):
        """
        Set config file

        Args:
            path_config (Path): Path to config file
        """
        with open(path_config, 'r') as f:
            config = yaml.safe_load(f)
        self.config = config

    def _get_random_rs(self, tsp: TSP) -> RollingStock:
        """
        Get a random rolling stock from a TSP

        Args:
            tsp (TSP): TSP object

        Returns:
            RollingStock: Rolling stock object randomly selected from  the specified TSP
        """
        return random.choice(tsp.rolling_stock)

    def _get_random_tsp(self) -> TSP:
        """
        Get a random TSP

        Returns:
            TSP: TSP object randomly selected from the available TSPs
        """
        return random.choice(list(self.tsps.values()))

    def _get_random_date(self) -> datetime.date:
        """
        This function will return a random datetime between two datetime objects.

        Returns:
            datetime: Random datetime in range [start, end] specified in config file
        """
        start, end = self.config['services']['dates']['min'], self.config['services']['dates']['max']

        delta = end - start
        int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
        random_second = random.randrange(int_delta)
        return start + datetime.timedelta(seconds=random_second)

    def _get_random_prices(self,
                           line: Line,
                           rolling_stock: RollingStock,
                           tsp: TSP
        ) -> Dict[Tuple[str, str], Dict[Seat, float]]:
        """
        Get prices for a service for a given line, rolling stock and TSP

        Args:
            line: Line object
            rolling_stock: RollingStock object
            tsp: TSP object

        Returns:
            Dict[Tuple[str, str], Dict[Seat, float]]: Prices for each pair of stations
        """
        prices = {}
        hard_types = rolling_stock.seats.keys()
        seats = list(filter(lambda s: s.hard_type in hard_types, list(self.seats.values())))
        base_price = self.config['prices']['base']
        max_price = self.config['prices']['max']
        distance_factor = self.config['prices']['distance_factor']
        seat_factor = self.config['prices']['seat_type_factor']
        tsp_factor = self.config['prices']['tsp_factor']

        for pair in line.pairs:
            origin_sta, destination_sta = line.pairs[pair]
            distance = _get_distance(line, origin_sta, destination_sta)
            # print(f'Distance between {origin_sta.name} and {destination_sta.name}: {distance} km')
            prices[pair] = {}
            for seat in seats:
                price_calc = (base_price + distance * distance_factor) * seat_factor[seat.id] * tsp_factor[tsp.id]
                prices[pair][seat] = str(round(price_calc, 2)) if price_calc < max_price else max_price

        return prices

    def _get_random_time_slot(self) -> TimeSlot:
        """
        Get a random time slot

        Returns:
            TimeSlot: Time slot object
        """
        ts_probabilities = self.config['time_slots']['probabilities']
        hour = random.choices(list(ts_probabilities.keys()), weights=list(ts_probabilities.values()))[0]
        minutes = random.randint(0, 59)
        start_time = datetime.timedelta(hours=hour, minutes=minutes)
        end_time = start_time + datetime.timedelta(minutes=10)
        time_slot_id = f'{start_time.seconds}'
        return TimeSlot(time_slot_id, start_time, end_time)

    def _get_random_line(self) -> Line:
        """
        Get random line from corridor

        Returns:
            Line: Line object
        """
        probs = self.config['lines']['probabilities'].values()
        return random.choices(list(self.lines.values()), weights=list(probs))[0]

    @staticmethod
    def _write_to_yaml(filename: Path, objects):
        """
        Write objects to yaml file

        Args:
            filename (Path): Name of the file
            objects (Dict): Dict of objects to write

        Returns:
            None
        """

        if not os.path.exists(filename):
            with open(filename, "w"):
                pass

        with open(filename, 'a') as f:
            yaml.safe_dump(objects, f, sort_keys=False, allow_unicode=True)

    @staticmethod
    def set_seed(seed: int) -> None:
        """
        Set seed for the random number generator.

        Args:
            seed (int): Seed for the random number generator.
        """
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    @staticmethod
    def _get_stations(data: Mapping[Any, Any],
                      key: str = 'stations'
                      ) -> Dict[str, Station]:
        """
        Private method to build a dict of Station objects from YAML data.

        Args:
            data (Mapping[Any, Any]): YAML data as nested dict.
            key (str): Key to access the data in the YAML file. Default: 'stations'.

        Returns:
            Dict[str, Station]: Dict of Station objects.
        """
        stations = {}
        for s in data[key]:
            assert all(k in s.keys() for k in ('id', 'name', 'short_name', 'city')), "Incomplete Station data"
            lat, lon = tuple(s.get('coordinates', {'lat': None, 'lon': None}).values())
            if not lat or not lon:
                station_id = str(s['id'])
                stations[station_id] = Station(station_id, s['name'], s['city'], s['short_name'])
            else:
                coords = (float(lat), float(lon))
                stations[str(s['id'])] = Station(str(s['id']), s['name'], s['city'], s['short_name'], coords)
        return stations

    @staticmethod
    def _get_time_slots(data: Mapping[Any, Any],
                        key: str = 'timeSlot'
                        ) -> Dict[str, TimeSlot]:
        """
        Private method to build a dict of TimeSlot objects from YAML data.

        Args:
            data (Mapping[Any, Any]): YAML data as nested dict.
            key (str): Key to access the data in the YAML file. Default: 'timeSlot'.

        Returns:
            Dict[str, TimeSlot]: Dict of TimeSlot objects.
        """
        time_slots = {}
        for time_slot in data[key]:
            assert all(k in time_slot.keys() for k in ('id', 'start', 'end')), "Incomplete TimeSlot data"
            time_slot_id = str(time_slot['id'])
            time_slots[time_slot_id] = TimeSlot(time_slot_id, get_time(time_slot['start']), get_time(time_slot['end']))
        return time_slots

    @staticmethod
    def _get_corridors(data: Mapping[Any, Any],
                       stations: (Mapping[str, Station]),
                       key: str = 'corridor'
                       ) -> Dict[str, Corridor]:
        """
        Private method to build a dict of Corridor objects from YAML data.

        Args:
            data (Mapping[Any, Any]): YAML data as nested dict.
            stations (Mapping[str, Station]): Dict of Station objects.
            key (str): Key to access the data in the YAML file. Default: 'corridor'.

        Returns:
            Dict[str, Corridor]: Dict of Corridor objects.
        """

        def to_station(tree: Dict, sta_dict: Mapping[str, Station]) -> Dict[Station, Dict]:
            """
            Recursive function to build a tree of Station objects from a tree of station IDs.

            Args:
                tree (Mapping): Tree of station IDs.
                sta_dict (Mapping[str, Station]): Dict of Station objects {station_id: Station object}

            Returns:
                Dict[Station, Dict]: Tree of Station objects.
            """
            if not tree:
                return {}
            return {sta_dict[node]: to_station(tree[node], sta_dict) for node in tree}

        corridors = {}
        for c in data[key]:
            assert all(k in c.keys() for k in ('id', 'name', 'stations')), "Incomplete Corridor data"

            tree_dictionary = convert_tree_to_dict(c['stations'])
            corr_stations_ids = set_stations_ids(tree_dictionary)
            assert all(s in stations.keys() for s in corr_stations_ids), "Station not found in Station list"

            stations_tree = to_station(deepcopy(tree_dictionary), stations)
            corridor_id = str(c['id'])
            corridors[corridor_id] = Corridor(corridor_id, c['name'], stations_tree)

        return corridors

    @staticmethod
    def _get_lines(data: Mapping[Any, Any],
                   corridors: Mapping[str, Corridor],
                   key='line'
                   ) -> Dict[str, Line]:
        """
        Private method to build a dict of Line objects from YAML data.

        Args:
            data (Mapping[Any, Any]): YAML data
            corridors (Mapping[str, Corridor]): Dict of Corridor objects.
            key (str): Key to access the data in the YAML file. Default: 'line'.

        Returns:
            Dict[str, Line]: Dict of Line objects.
        """
        lines = {}
        for ln in data[key]:
            assert all(k in ln.keys() for k in ('id', 'name', 'corridor', 'stops')), 'Incomplete Line data'

            corr_id = str(ln['corridor'])
            assert corr_id in corridors.keys(), 'Corridor not found in Corridor list'
            corr = corridors[corr_id]

            for stn in ln['stops']:
                assert all(k in stn for k in ('station', 'arrival_time', 'departure_time')), 'Incomplete Stops data'

            corr_stations_ids = [s.id for s in corr.stations.values()]
            assert all(s['station'] in corr_stations_ids for s in ln['stops']), 'Station not found in Corridor list'

            timetable = {s['station']: (float(s['arrival_time']), float(s['departure_time']))
                         for s in ln['stops']}
            line_id = str(ln['id'])
            lines[line_id] = Line(line_id, ln['name'], corr, timetable)

        return lines

    @staticmethod
    def _get_seats(data: Mapping[Any, Any],
                   key: str = 'seat'
                   ) -> Dict[str, Seat]:
        """
        Private method to build a dict of Seat objects from YAML data.

        Args:
            data (Mapping[Any, Any]): YAML data.
            key (str): Key to access the data in the YAML file. Default: 'seat'.

        Returns:
            Dict[str, Seat]: Dict of Seat objects.
        """
        seats = {}
        for s in data[key]:
            assert all(k in s.keys() for k in ('id', 'name', 'hard_type', 'soft_type')), 'Incomplete Seat data'
            seat_id = str(s['id'])
            seats[seat_id] = Seat(seat_id, s['name'], s['hard_type'], s['soft_type'])
        return seats

    @staticmethod
    def _get_rolling_stock(data: Mapping[Any, Any],
                           seats: Mapping[str, Seat],
                           key: str = 'rollingStock'
                           ) -> Dict[str, RollingStock]:
        """
        Private method to build a dict of RollingStock objects from YAML data.

        Args:
            data (Mapping[Any, Any]): YAML data.
            seats (Mapping[str, Seat]): Dict of Seat objects.
            key (str): Key to access the data in the YAML file. Default: 'rollingStock'.

        Returns:
            Dict[str, RollingStock]: Dict of RollingStock objects.
        """
        rolling_stocks = {}
        for rolling_stock in data[key]:
            assert all(k in rolling_stock.keys() for k in ('id', 'name', 'seats')), 'Incomplete RollingStock data'

            for seat in rolling_stock['seats']:
                assert all(key in seat for key in ('hard_type', 'quantity')), 'Incomplete seats data for RS'

            assert all(seat['hard_type'] in [seat.hard_type for seat in seats.values()] for seat in
                       rolling_stock['seats']), 'Invalid hard_type for RS'

            rolling_stock_seats = {int(seat['hard_type']): int(seat['quantity']) for seat in rolling_stock['seats']}
            rolling_stock_id = str(rolling_stock['id'])
            rolling_stocks[rolling_stock_id] = RollingStock(rolling_stock_id,
                                                            rolling_stock['name'],
                                                            rolling_stock_seats)

        return rolling_stocks

    @staticmethod
    def _get_tsps(data: Mapping[Any, Any],
                  rolling_stock: Mapping[str, RollingStock],
                  key: str = 'trainServiceProvider'
        ) -> Dict[str, TSP]:
        """
        Private method to build a dict of TSP objects from YAML data.

        Args:
            data (Mapping[Any, Any])): YAML data
            rolling_stock (Mapping[str, RollingStock]): Dict of RollingStock objects.
            key (str): Key to access the data in the YAML file. Default: 'trainServiceProvider'.

        Returns:
            Dict[str, TSP]: Dict of TSP objects.
        """
        tsps = {}
        for tsp in data[key]:
            assert all(k in tsp.keys() for k in ('id', 'name', 'rolling_stock')), 'Incomplete TSP data'
            assert all(str(i) in rolling_stock.keys() for i in tsp['rolling_stock']), 'Unknown RollingStock ID'
            tsp_id = str(tsp['id'])
            tsps[tsp_id] = TSP(tsp_id, tsp['name'], [rolling_stock[str(rs_id)] for rs_id in tsp['rolling_stock']])
        return tsps
