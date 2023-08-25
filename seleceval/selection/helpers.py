"""
Helper functions for selection algorithms
"""
import concurrent
from typing import List, Tuple, Union

from flwr.common import Code, EvaluateRes, GetPropertiesRes, GetPropertiesIns
from flwr.server.client_proxy import ClientProxy


def get_client_properties(
    client: ClientProxy, property_ins: GetPropertiesIns, timeout: int
):
    """
    Get the properties of a client
    :param client: The client proxy (for ray)
    :param property_ins: Config for getting properties (not used)
    :param timeout: Timeout for getting properties
    :return: The client proxy and the properties
    """
    res = client.get_properties(property_ins, timeout)
    return client, res


def _handle_finished_future_after_properties_get(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, GetPropertiesRes]],
    failures: List[Union[Tuple[ClientProxy, GetPropertiesRes], BaseException]],
) -> None:
    """
    Convert finished future into either a result or a failure
    :param future: List of client futures that completed (Since it happens asynchronously)
    :param results: List of results
    :param failures: List of failures
    :return: None
    """

    # Check for exceptions
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Store result
    result: Tuple[ClientProxy, GetPropertiesRes] = future.result()
    _, res = result

    # Validate result status code and add to results if successful
    if res.status.code == Code.OK:
        results.append(result)
        return
    # Append failures if not successful
    failures.append(result)


def _handle_finished_future_after_evaluate(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
) -> None:
    """
    Convert finished future into either a result or a failure
    :param future: List of client futures that completed (Since it happens asynchronously)
    :param results: List of results
    :param failures: List of failures
    :return: None
    """
    # Check for exceptions
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Store result
    result: Tuple[ClientProxy, EvaluateRes] = future.result()
    _, res = result

    # Validate result status code and add to results if successful
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Append failures if not successful
    failures.append(result)
