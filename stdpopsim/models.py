"""
Common infrastructure for specifying demographic models.
"""
import sys
import inspect

import msprime
import numpy as np
import copy

# Defaults taken from np.allclose
DEFAULT_ATOL = 1e-05
DEFAULT_RTOL = 1e-08


class UnequalModelsError(Exception):
    """
    Exception raised models by verify_equal to indicate that models are
    not sufficiently close.
    """


def population_configurations_equal(
        pop_configs1, pop_configs2, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
    """
    Returns True if the specified lists of msprime PopulationConfiguration
    objects are equal to the specified tolerances.

    See the :func:`.verify_population_configurations_equal` function for
    details on the assumptions made about the objects.
    """
    try:
        verify_population_configurations_equal(
            pop_configs1, pop_configs2, rtol=rtol, atol=atol)
        return True
    except UnequalModelsError:
        return False


def verify_population_configurations_equal(
        pop_configs1, pop_configs2, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
    """
    Checks if the specified lists of msprime PopulationConfiguration
    objects are equal to the specified tolerances and raises an UnequalModelsError
    otherwise.

    We make some assumptions here to ensure that the models we specify
    are well-defined: (1) The sample size is not set for PopulationConfigurations
    (2) the initial_size is defined. If these assumptions are violated a
    ValueError is raised.
    """
    for pc1, pc2 in zip(pop_configs1, pop_configs2):
        if pc1.sample_size is not None or pc2.sample_size is not None:
            raise ValueError(
                "Models defined in stdpopsim must not use the 'sample_size' "
                "PopulationConfiguration option")
        if pc1.initial_size is None or pc2.initial_size is None:
            raise ValueError(
                "Models defined in stdpopsim must set the initial_size")
    if len(pop_configs1) != len(pop_configs2):
        raise UnequalModelsError("Different numbers of populations")
    initial_size1 = np.array([pc.initial_size for pc in pop_configs1])
    initial_size2 = np.array([pc.initial_size for pc in pop_configs2])
    if not np.allclose(initial_size1, initial_size2, rtol=rtol, atol=atol):
        raise UnequalModelsError("Initial sizes differ")
    growth_rate1 = np.array([pc.growth_rate for pc in pop_configs1])
    growth_rate2 = np.array([pc.growth_rate for pc in pop_configs2])
    if not np.allclose(growth_rate1, growth_rate2, rtol=rtol, atol=atol):
        raise UnequalModelsError("Growth rates differ")


def demographic_events_equal(
        events1, events2, num_populations, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
    """
    Returns True if the specified list of msprime DemographicEvent objects are equal
    to the specified tolerances.
    """
    try:
        verify_demographic_events_equal(
            events1, events2, num_populations, rtol=rtol, atol=atol)
        return True
    except UnequalModelsError:
        return False


def verify_demographic_events_equal(
        events1, events2, num_populations, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
    """
    Checks if the specified list of msprime DemographicEvent objects are equal
    to the specified tolerances and raises a UnequalModelsError otherwise.
    """
    # Get the low-level dictionary representations of the events.
    dicts1 = [event.get_ll_representation(num_populations) for event in events1]
    dicts2 = [event.get_ll_representation(num_populations) for event in events2]
    if len(dicts1) != len(dicts2):
        raise UnequalModelsError("Different numbers of demographic events")
    for d1, d2 in zip(dicts1, dicts2):
        if set(d1.keys()) != set(d2.keys()):
            raise UnequalModelsError("Different types of demographic events")
        for key in d1.keys():
            value1 = d1[key]
            value2 = d2[key]
            if isinstance(value1, float):
                if not np.isclose(value1, value2, rtol=rtol, atol=atol):
                    raise UnequalModelsError("Event {} mismatch: {} != {}".format(
                        key, value1, value2))
            else:
                if value1 != value2:
                    raise UnequalModelsError("Event {} mismatch: {} != {}".format(
                        key, value1, value2))


class Model(object):
    """
    Class representing a simulation model that can be run in msprime.

    .. todo:: Document this class.
    """
    def __init__(self):
        self.population_configurations = []
        self.demographic_events = []
        # Defaults to a single population
        self.migration_matrix = [[0]]
        #self.ddb = msprime.DemographyDebugger(**self.asdict())
        #self.epochs = self.ddb.epochs

    def debug(self, out_file=sys.stdout):
        # Use the demography debugger to print out the demographic history
        # that we have just described.
        dd = msprime.DemographyDebugger(
            population_configurations=self.population_configurations,
            migration_matrix=self.migration_matrix,
            demographic_events=self.demographic_events)
        dd.print_history(out_file)

    def asdict(self):
        return {
            "population_configurations": self.population_configurations,
            "migration_matrix": self.migration_matrix,
            "demographic_events": self.demographic_events}


# ###############################################################
# UNDER CONSTRUCTION - GROUND TRUTH N_t
# ###############################################################
        
    def num_pops(self):
        """
        This function returns the number populations
        defined by the demographic model
        """
        ddb = msprime.DemographyDebugger(**self.asdict())
        return len(ddb.epochs[0].populations)
        

    # TODO Rename
    def get_Ne_through_time_single_pop(self, end, start = 0, num_steps = 10):
        """
        This function returns the defined Ne for each individual
        sub population for any demographic model.
    
        parameter `end` specifies at which generation to stop 
        collecting effective population sizes among subpopulations

        `num_steps` parameter will determine how many points between 
        `start` and `end` to sample.
        
        This function will return a numpy ndarray that will contain the effective
        population sizes for each subpopulation individually across time.        
        """
        
        ddb = msprime.DemographyDebugger(**self.asdict())
        num_pops = self.num_pops()
        epoch_times = ddb.epoch_times()
        N_t = np.zeros([num_steps,num_pops])
        steps = np.linspace(start, end, num_steps)
        for j,t in enumerate(steps):
            N = self.get_N_M_at_t(t)["population_sizes"]
            N_t[j] = N
    
        return N_t

    # TODO Rename
    def get_Ne_through_time_all_pops(self, end, num_samples_per_location, num_steps = 10):
        """
        This function will calculate the true Ne for a population with
        multiple sub-populations. This is a function that takes a 
        demographic model and returns a
        function of time that gives inverse coalescence rate.
        
        num_samples_per_location should be a list containing the
        number of samples in each subpopulation organized by population index.

        num_steps parameter will determine how many points along the 
        time axis are returnes.
        """

        ddb = msprime.DemographyDebugger(**self.asdict())
        num_pops = self.num_pops()
        assert(len(num_samples_per_location) == num_pops)
        P = np.zeros([num_pops**2,num_pops**2])
        index_array = np.array(range(num_pops**2)).reshape([num_pops,num_pops])
        for x in range(num_pops):
            for y in range(num_pops):
                K_x = num_samples_per_location[x]
                K_y = num_samples_per_location[y]
                P[index_array[x,y],index_array[x,y]] = K_x * (K_y - self._delta(x,y))
        P = P / np.sum(P)
        r = np.zeros(num_steps)
        steps = np.linspace(0, end, num_steps)
        dt = steps[1] - steps[0]
        for j in range(num_steps):
            N_M = self._get_N_M_at_t(steps[j])
            pop_sizes = N_M["population_sizes"]
            mig_matrix = N_M["migration_matrix"]
            C = self._get_C_from_N(pop_sizes)
            Mp = self._get_Mp_from_M(mig_matrix)
            G = self._get_G_from_Mp_C(Mp, C)
            P = P * np.exp(dt * G)
            r[j] = np.sum(np.matmul(P,C)) / np.sum(P)
                
        return r

    # TODO Rename
    def _get_epoch_at_t(self, t):
        """
        Given a time, t (in generations), find and return the 
        The Epoch for which this belongs.
        If there is no epoch that exists for the given t 
        then it returns None.

        <class 'int'> -> <class 'msprime.simulations.Epoch'>
        """
        # lets set these as class variables that get 
        # set in the initialization of a subclass.
        ddb = msprime.DemographyDebugger(**self.asdict())
        epochs = ddb.epochs
        j = 0
        while epochs[j].end_time <= t:
            j += 1
        return j, epochs[j]

    # TODO Rename
    def _get_N_M_at_t(self, t):
        """
        Given a time, t (in generations), find and return 
        a dictionary containing:
        1: "population_sizes" the vector N which should represent effective population size
        for each populations at time t and,
        2: "migration_matrix" The migration matrix for for the populations at time t
        
        <class 'int'> -> <class 'dict'>
        """
    
        epochIndex, epoch = self._get_epoch_at_t(t)
        ddb = msprime.DemographyDebugger(**self.asdict())

        #N = [pop.start_size for pop in epoch.populations]
        N = ddb.population_size_history[:, epochIndex]
             
        for i,pop in enumerate(epoch.populations):
            s = t - epoch.start_time
            g = pop.growth_rate
            N[i] *= np.exp(-1 * g * s)

        return {"population_sizes":N, "migration_matrix":epoch.migration_matrix}

    def _delta(self, x, y):
        if x == y:
            return 1
        else:
            return 0
    
    # TODO Rename
    def _get_C_from_N(self, N):
        """
        Compute the matrix C, which reprents coalescent probabilities
        in each population, from Individual effective population sizes.
        """
        
        n = len(N)
        C = np.zeros([n**2,n**2])
        index_array = np.array(range(n**2)).reshape([n,n])
        for idx in range(n):
            C[index_array[idx,idx],index_array[idx,idx]] = 1 / (2 * N[idx])

        return C

    # TODO Rename
    def _get_G_from_Mp_C(self, Mp, C):
        """
        ????
        """
        I = np.eye(len(Mp[0]))
        G = (np.kron(Mp,I) + np.kron(I,Mp)) - C

        return G

    # TODO Rename
    def _get_Mp_from_M(self, M):
        """
        Simply compute Mp. which represents the **,
        From the migration matrix.
        """
        
        Mp = copy.deepcopy(M)
        for idx,row in enumerate(M):
            Mp[idx][idx] = -1 * sum(row)

        return Mp


# ###############################################################
# END CONSTRUCTION - GROUND TRUTH N_t
# ###############################################################

        
    def equals(self, other, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
        """
        Returns True if this model is equal to the specified model to the
        specified numerical tolerance (as defined by numpy.allclose).

        We use the 'equals' method here rather than the equality operator
        because we need to be able to specifiy the numerical tolerances.
        """
        try:
            self.verify_equal(other, rtol=rtol, atol=atol)
            return True
        except (UnequalModelsError, AttributeError):
            return False

    def verify_equal(self, other, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
        """
        Equivalent to the :func:`.equals` method, but raises a UnequalModelsError if the
        models are not equal rather than returning False.
        """
        mm1 = np.array(self.migration_matrix)
        mm2 = np.array(other.migration_matrix)
        if mm1.shape != mm2.shape:
            raise UnequalModelsError("Migration matrices different shapes")
        if not np.allclose(mm1, mm2, rtol=rtol, atol=atol):
            raise UnequalModelsError("Migration matrices differ")
        verify_population_configurations_equal(
            self.population_configurations, other.population_configurations,
            rtol=rtol, atol=atol)
        verify_demographic_events_equal(
            self.demographic_events, other.demographic_events,
            len(self.population_configurations),
            rtol=rtol, atol=atol)


def all_models():
    """
    Returns the list of all Model classes that are defined within the stdpopsim
    module.
    """
    ret = []
    for cls in Model.__subclasses__():
        mod = inspect.getmodule(cls).__name__
        if mod.startswith("stdpopsim"):
            ret.append(cls())
    return ret
