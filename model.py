import numpy as np
import psyneulink as pnl


def get_model(learning_rate=.3, n_time_steps=60):
    """get a model, described in Montague, Dayan, and Sejnowski (1996)

    Parameters
    ----------
    n_time_steps : int
        number of time steps per trial
    learning_rate : float
        learning rate, default to 1e-3

    Returns
    -------
    pnl.composition, list
        the model
    """
    # Create Processing Components
    sample_mechanism = pnl.TransferMechanism(
        default_variable=np.zeros(n_time_steps),
        name=pnl.SAMPLE
    )
    action_func = pnl.Linear(slope=1.0, intercept=0.01)
    action_selection = pnl.TransferMechanism(
        default_variable=np.zeros(n_time_steps),
        function=action_func,
        name='Action Selection'
    )
    sample_to_action_selection = pnl.MappingProjection(
        sender=sample_mechanism,
        receiver=action_selection,
        matrix=np.zeros((n_time_steps, n_time_steps))
    )
    # Create Composition
    comp = pnl.Composition()
    # Add Processing Components to the Composition
    pathway = [sample_mechanism, sample_to_action_selection, action_selection]
    # Add Learning Components to the Composition
    learning_related_components = comp.add_td_learning_pathway(
        pathway, learning_rate=learning_rate
    )
    # Unpack Relevant Learning Components
    prediction_error_mechanism = learning_related_components[
        pnl.COMPARATOR_MECHANISM]
    target_mechanism = learning_related_components[pnl.TARGET_MECHANISM]
    # Create Log
    prediction_error_mechanism.log.set_log_conditions(pnl.VALUE)
    nodes = [sample_mechanism, prediction_error_mechanism, target_mechanism]
    return comp, nodes
