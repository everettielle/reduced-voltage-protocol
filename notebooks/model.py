import myokit
import pints
import numpy


class Model(pints.ForwardModel):
    def __init__(self, model, protocol):
        super().__init__()

        # Load model
        self.model = myokit.load_model(model)

        # Create a CVODE Simulation
        self.sim = myokit.Simulation(self.model)

        # Read data from DataLog class
        self.log = myokit.DataLog.load_csv(protocol).npview()

        # Extract 'time', 'current', 'voltage', and evaluate 'time_max'
        self.time = self.log['time']
        self.current = self.log['current']
        self.voltage = self.log['voltage']
        self.time_max = self.log['time'][-1] + 1

        # Apply data-clamp
        self.sim.set_fixed_form_protocol(self.time, self.voltage)

        # Set max step size
        self.sim.set_max_step_size(0.1)

    def n_parameters(self):
        return int(self.model.value('ikr.n_params'))

    def set_tolerance(self, tol):
        self.sim.set_tolerance(tol, tol)

    def simulate(self, parameters, times):
        # Reset to default time and state
        self.sim.reset()

        # Apply parameters
        for i, p in enumerate(parameters):
            self.sim.set_constant('ikr.p' + str(1 + i), p)

        # Run
        time_max = times[-1] + (times[-1] - times[-2])
        try:
            log = self.sim.run(time_max, log_times=times, log=['ikr.IKr'])
            return log['ikr.IKr']
        except myokit.SimulationError:
            print('Error evaluating with parameters: ' + str(parameters))
            return numpy.nan * times