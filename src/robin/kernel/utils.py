from ..demand.entities import Passenger
from ..decision_model.propositions import PDC


def get_constrain_value(passenger: Passenger,
                        variable_name: str,
                        ) -> float:
    """
    Get the maximum value of a variable in the user pattern rules.

    Args:
        passenger: Passenger object.
        variable_name: Name of the variable.

    Returns:
        The maximum value of the variable in the user pattern.
    """
    max_value = 1.0
    for rule in passenger.user_pattern.behaviour_rules:
        # TODO: Recursive
        for proposition in rule.antecedent.proposiciones:
            if type(proposition) is PDC:
                for prop in proposition.proposiciones:
                    if prop.variable.name == variable_name:
                        max_value = max(prop.term.values)

            elif proposition.variable.name == variable_name:
                max_value = max(proposition.term.values)

    return max_value
