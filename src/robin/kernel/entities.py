"""Entities for the kernel module."""

import numpy as np
import pandas as pd
import random
import os

from ..demand.entities import Demand, Passenger
from ..supply.entities import Service, Supply
from .utils import get_constrain_value

from pathlib import Path
from typing import List, Union


class Kernel:
    """
    The kernel class integrates the supply and demand modules.

    Attributes:
        supply (Supply): Supply object.
        demand (Demand): Demand object.
    """
    
    def __init__(self, path_config_supply: Path, path_config_demand: Path, seed: Union[int, None] = None) -> None:
        """
        Initialize a kernel object.

        Args:
            path_config_supply (Path): Path to the supply configuration file.
            path_config_demand (Path): Path to the demand configuration file.
            seed (int, optional): Seed for the random number generator. Defaults to None.
        """
        if seed is not None:
            self.set_seed(seed)
        self.supply = Supply.from_yaml(path_config_supply)
        self.demand = Demand.from_yaml(path_config_demand)

    def _to_csv(self, passengers: List[Passenger], output_path: Path = Path('output.csv')) -> None:
        """
        Save passengers data to csv file.

        Args:
            passengers (List[Passenger]): List of passengers.
            output_path (Path, optional): Path to the output csv file. Defaults to 'output.csv'.
        """
        column_names = [
            'id', 'user_pattern', 'departure_station', 'arrival_station',
            'arrival_day', 'arrival_time', 'purchase_day', 'service', 'service_departure_time',
            'service_arrival_time', 'seat', 'price', 'utility', 'best_service', 'best_seat', 'best_utility'
        ]

        data = []
        for passenger in passengers:
            data.append([
                passenger.id,
                passenger.user_pattern,
                passenger.market.departure_station,
                passenger.market.arrival_station,
                passenger.arrival_day,
                passenger.arrival_time,
                passenger.purchase_day,
                passenger.service,
                passenger.service_departure_time,
                passenger.service_arrival_time,
                passenger.seat,
                passenger.ticket_price,
                passenger.utility,
                passenger.best_service,
                passenger.best_seat,
                passenger.best_utility
            ])
        df = pd.DataFrame(data=data, columns=column_names)
        df.to_csv(output_path, index=False)

        utilities_df = pd.DataFrame()
        for passenger in passengers:
            passenger_data = passenger.utility_history
            passenger_data['id'] = passenger.id
            passenger_data['user_pattern'] = passenger.user_pattern
            passenger_row = pd.DataFrame([passenger_data])
            utilities_df = pd.concat([utilities_df, passenger_row], ignore_index=True)

        utilities_df.to_csv(output_path.parent / 'utilities.csv', index=False)

    def _to_json(self, output_path: Path, inference_trace: dict) -> None:
        """
        Save inference trace to json file.

        Args:
            output_path (Path): Path to the output json file.
            inference_trace (dict): Inference trace.
        """
        import json
        with open(output_path, 'w') as f:
            json.dump(inference_trace, f, indent=4)

    def simulate(
            self,
            output_path: Union[Path, None] = None,
            departure_time_hard_restriction: bool = False,
            save_trace: bool = False
        ) -> List[Service]:
        """
        Simulate the demand-supply interaction.

        The passengers will maximize the utility for each service and seat,  according to
        its origin-destination and date, buying a ticket only if the utility is positive.

        Args:
            output_path (Path, optional): Path to the output csv file. Defaults to None.
            departure_time_hard_restriction (bool, optional): If True, the passenger will not
                be assigned to a service with a departure time that is not valid. Defaults to True.
            save_trace (bool, optional): If True, the inference trace will be saved. Defaults to False.

        Returns:
            List[Service]: List of services with updated tickets.
        """
        inference_trace = {}
        trace = None

        # Generate passengers demand
        passengers = self.demand.generate_passengers()
        
        for passenger in passengers:
            origin = passenger.market.departure_station
            destination = passenger.market.arrival_station

            # Filter services
            max_origin_diff = get_constrain_value(passenger, 'origin')
            max_destination_diff = get_constrain_value(passenger, 'destination')
            max_date_diff = get_constrain_value(passenger, 'date')

            services = self.supply.filter_services_fuzzy(
                origin=origin,
                destination=destination,
                origin_coords=passenger.market.departure_station_coords,
                destination_coords=passenger.market.arrival_station_coords,
                max_origin_diff=max_origin_diff,
                max_destination_diff=max_destination_diff,
                max_date_diff=max_date_diff
            )

            # Calculate utility for each service and seat
            service_arg_max = None
            seat_arg_max = None
            seat_utility = 0
            ticket_price = 0
            service_arg_max_global = 0
            seat_arg_max_global = 0
            seat_utility_global = 0

            passenger_row = {}
            for service in services:
                for seat in service.prices.get((origin, destination), {}).keys():
                    # Calculate utility
                    inference_result = passenger.get_fuzzy_utility(
                        seat=seat,
                        service=service,
                        departure_time_hard_restriction=departure_time_hard_restriction
                    )
                    utility = inference_result['result']
                    passenger.utility_history[(service.id, seat.id)] = {'service': service,
                                                                        'seat': seat,
                                                                        'price': service.prices[(origin, destination)][seat],
                                                                        'utility': utility}

                    # Update global utility
                    if utility > seat_utility_global:
                        service_arg_max_global = service
                        seat_arg_max_global = seat
                        seat_utility_global = utility

                    # Check if seat is available
                    anticipation = passenger.purchase_day
                    if not service.tickets_available(origin, destination, seat, anticipation):
                        continue

                    # Update service with max utility
                    if utility > seat_utility:
                        service_arg_max = service
                        seat_arg_max = seat
                        seat_utility = utility
                        ticket_price = service.prices[(origin, destination)][seat]
                        trace = inference_result

            if passenger.early_stop:
                for i, key in enumerate(passenger.utility_history.keys()):
                    if passenger.utility_history[key]['utility'] >= passenger.user_pattern.utility_threshold:
                        service_arg_max = passenger.utility_history[key]['service']
                        seat_arg_max = passenger.utility_history[key]['seat']
                        ticket_price = passenger.utility_history[key]['price']
                        seat_utility = passenger.utility_history[key]['utility']

            # Buy ticket if utility is greater than threshold
            if seat_utility > passenger.user_pattern.utility_threshold:
                assert service_arg_max is not None
                assert seat_arg_max is not None

                ticket_bought = service_arg_max.buy_ticket(
                    origin=passenger.market.departure_station,
                    destination=passenger.market.arrival_station,
                    seat=seat_arg_max,
                    anticipation=passenger.purchase_day
                )
                if ticket_bought:
                    passenger.service = service_arg_max.id
                    passenger.service_departure_time = service_arg_max.service_departure_time
                    passenger.service_arrival_time = service_arg_max.service_arrival_time
                    passenger.seat = seat_arg_max.name
                    passenger.ticket_price = ticket_price
                    passenger.utility = seat_utility
                    if save_trace:
                        trace['user_pattern'] = passenger.user_pattern.name
                        inference_trace[passenger.id] = trace

            # Even if passenger doesn't buy ticket, save best service found (if utility is positive)
            if seat_utility_global > 0:
                passenger.best_service = service_arg_max_global.id
                passenger.best_seat = seat_arg_max_global.name
                passenger.best_utility = seat_utility_global

        # Save passengers data to csv file
        if output_path is not None:
            self._to_csv(passengers, output_path)
            if save_trace:
                self._to_json(output_path.parent / 'inference_trace.json', inference_trace)

        return self.supply.services

    def set_seed(self, seed: int) -> None:
        """
        Set seed for the random number generator.

        Args:
            seed (int): Seed for the random number generator.
        """
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
