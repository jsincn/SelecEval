import concurrent
from typing import List, Tuple, Union

from flwr.common import Code, EvaluateRes, GetPropertiesRes, GetPropertiesIns
from flwr.server.client_proxy import ClientProxy

from ..client.client import Client


def get_client_properties(client: ClientProxy, property_ins: GetPropertiesIns, timeout: int):
    res = client.get_properties(property_ins, timeout)
    return client, res


def _handle_finished_future_after_properties_get(
        future: concurrent.futures.Future,  # type: ignore
        results: List[Tuple[ClientProxy, GetPropertiesRes]],
        failures: List[Union[Tuple[ClientProxy, GetPropertiesRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""

    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, GetPropertiesRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


def _handle_finished_future_after_evaluate(
        future: concurrent.futures.Future,  # type: ignore
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, EvaluateRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)
