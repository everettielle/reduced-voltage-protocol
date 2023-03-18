import pints
import numpy


class Boundaries_Model_A(pints.Boundaries):
    """
    A boundaries class that implements the maximum-rate boundaries used in
    Beattie et al.
    """

    # Limits for a-type parameters (untransformed)
    a_min = 1e-7
    a_max = 1e3

    # Limits for g-type parameters
    b_min = 1e-7
    b_max = 0.4

    # Limits for maximum rate coefficients
    km_min = 1.67e-5
    km_max = 1e3

    # Voltages used when determining maximum rate coefficients
    v_low = -120
    v_high = 60

    def __init__(self):
        self.g_min = 1e2 * 1e-3
        self.g_max = 5e5 * 1e-3

        # Univariate paramater bounds
        self._lower = numpy.array([
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.g_min,
        ])
        self._upper = numpy.array([
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.g_max,
        ])

    def n_parameters(self):
        return 9

    def check(self, parameters):
        # Check parameter boundaries
        if (numpy.any(parameters <= self._lower)
                or numpy.any(parameters >= self._upper)):
            return False

        # Check rate boundaries
        k1m = parameters[0] * numpy.exp(parameters[1] * self.v_high)
        if k1m <= self.km_min or k1m >= self.km_max:
            return False
        k2m = parameters[2] * numpy.exp(-parameters[3] * self.v_low)
        if k2m <= self.km_min or k2m >= self.km_max:
            return False
        k3m = parameters[4] * numpy.exp(parameters[5] * self.v_high)
        if k3m <= self.km_min or k3m >= self.km_max:
            return False
        k4m = parameters[6] * numpy.exp(-parameters[7] * self.v_low)
        if k4m <= self.km_min or k4m >= self.km_max:
            return False

        # All tests passed!
        return True

    def _sample_partial(self, v):
        """Samples a pair of kinetic parameters"""
        for i in range(100):
            a = numpy.exp(numpy.random.uniform(
                numpy.log(self.a_min), numpy.log(self.a_max)))
            b = numpy.random.uniform(self.b_min, self.b_max)
            km = a * numpy.exp(b * v)
            if km > self.km_min and km < self.km_max:
                return a, b
        raise ValueError('Too many iterations')

    def sample(self, n=1):
        points = numpy.zeros((n, 9))
        for i in range(n):
            points[i, 0:2] = self._sample_partial(self.v_high)
            points[i, 2:4] = self._sample_partial(-self.v_low)
            points[i, 4:6] = self._sample_partial(self.v_high)
            points[i, 6:8] = self._sample_partial(-self.v_low)
            points[i, 8] = numpy.random.uniform(self.g_min, self.g_max)
        return points


# ======== Boundary class implementation ========
class Boundaries_Model_B(pints.Boundaries):
    """
    A boundaries class that implements the maximum-rate boundaries used in
    Beattie et al.
    """

    # Limits for a-type parameters (untransformed)
    a_min = 1e-7
    a_max = 1e3

    # Limits for g-type parameters
    b_min = 1e-7
    b_max = 0.4

    # Limits for maximum rate coefficients
    km_min = 1.67e-5
    km_max = 1e3

    # Voltages used when determining maximum rate coefficients
    v_low = -120
    v_high = 60

    def __init__(self):
        self.g_min = 1e2 * 1e-3
        self.g_max = 5e5 * 1e-3

        # Univariate paramater bounds
        self._lower = numpy.array([
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.km_min, self.km_min,
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.g_min,
        ])
        self._upper = numpy.array([
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.km_max, self.km_max,
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.g_max,
        ])

    def n_parameters(self):
        return 11

    def check(self, parameters):

        # Check parameter boundaries
        if (numpy.any(parameters <= self._lower)
                or numpy.any(parameters >= self._upper)):
            return False

        # Check rate boundaries
        k1m = parameters[0] * numpy.exp(parameters[1] * self.v_high)
        if k1m <= self.km_min or k1m >= self.km_max:
            return False
        k2m = parameters[2] * numpy.exp(-parameters[3] * self.v_low)
        if k2m <= self.km_min or k2m >= self.km_max:
            return False
        k3m = parameters[6] * numpy.exp(parameters[7] * self.v_high)
        if k3m <= self.km_min or k3m >= self.km_max:
            return False
        k4m = parameters[8] * numpy.exp(-parameters[9] * self.v_low)
        if k4m <= self.km_min or k4m >= self.km_max:
            return False

        # For constant value p6, p7
        for i in [4, 5]:
            if numpy.any(parameters[i] <= self.km_min) or numpy.any(parameters[i] >= self.km_max):
                return False

        # All tests passed!
        return True

    def _sample_partial(self, v):
        for i in range(100):
            a = numpy.exp(numpy.random.uniform(
                numpy.log(self.a_min), numpy.log(self.a_max)))
            b = numpy.random.uniform(self.b_min, self.b_max)
            km = a * numpy.exp(b * v)
            if self.km_min <= km <= self.km_max:
                return a, b
        raise ValueError('Too many iterations')

    def sample(self, n=1):
        points = numpy.zeros((n, 11))
        for i in range(n):
            points[i, 0:2] = self._sample_partial(self.v_high)
            points[i, 2:4] = self._sample_partial(-self.v_low)
            points[i, 4:6] = numpy.random.uniform(self.km_min, self.km_max)
            points[i, 6:8] = self._sample_partial(self.v_high)
            points[i, 8:10] = self._sample_partial(-self.v_low)
            points[i, 10] = numpy.random.uniform(self.g_min, self.g_max)
        return points


class Boundaries_Model_15(pints.Boundaries):
    """
    A boundaries class that implements the maximum-rate boundaries used in
    Beattie et al.
    """

    # Limits for a-type parameters (untransformed)
    a_min = 1e-7
    a_max = 1e3

    # Limits for g-type parameters
    b_min = 1e-7
    b_max = 0.4

    # Limits for maximum rate coefficients
    km_min = 1.67e-5
    km_max = 1e3

    # Voltages used when determining maximum rate coefficients
    v_low = -120
    v_high = 60

    def __init__(self):
        self.g_min = 1e2 * 1e-3
        self.g_max = 5e5 * 1e-3

        # Univariate paramater bounds
        self._lower = numpy.array([
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.g_min,
        ])
        self._upper = numpy.array([
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.g_max,
        ])

    def n_parameters(self):
        return 17

    def check(self, parameters):
        # Check parameter boundaries
        if (numpy.any(parameters <= self._lower)
                or numpy.any(parameters >= self._upper)):
            return False

        # Check rate boundaries
        k1m = parameters[0] * numpy.exp(parameters[1] * self.v_high)
        if k1m <= self.km_min or k1m >= self.km_max:
            return False
        k2m = parameters[2] * numpy.exp(-parameters[3] * self.v_low)
        if k2m <= self.km_min or k2m >= self.km_max:
            return False
        k3m = parameters[4] * numpy.exp(parameters[5] * self.v_high)
        if k3m <= self.km_min or k3m >= self.km_max:
            return False
        k4m = parameters[6] * numpy.exp(-parameters[7] * self.v_low)
        if k4m <= self.km_min or k4m >= self.km_max:
            return False
        k5m = parameters[8] * numpy.exp(parameters[9] * self.v_high)
        if k5m <= self.km_min or k5m >= self.km_max:
            return False
        k6m = parameters[10] * numpy.exp(-parameters[11] * self.v_low)
        if k6m <= self.km_min or k6m >= self.km_max:
            return False
        k7m = parameters[12] * numpy.exp(parameters[13] * self.v_high)
        if k7m <= self.km_min or k7m >= self.km_max:
            return False
        k8m = parameters[14] * numpy.exp(-parameters[15] * self.v_low)
        if k8m <= self.km_min or k8m >= self.km_max:
            return False

        # All tests passed!
        return True

    def _sample_partial(self, v):
        """Samples a pair of kinetic parameters"""
        for i in range(100):
            a = numpy.exp(numpy.random.uniform(
                numpy.log(self.a_min), numpy.log(self.a_max)))
            b = numpy.random.uniform(self.b_min, self.b_max)
            km = a * numpy.exp(b * v)
            if km > self.km_min and km < self.km_max:
                return a, b
        raise ValueError('Too many iterations')

    def sample(self, n=1):
        points = numpy.zeros((n, 17))
        for i in range(n):
            points[i, 0:2] = self._sample_partial(self.v_high)
            points[i, 2:4] = self._sample_partial(-self.v_low)
            points[i, 4:6] = self._sample_partial(self.v_high)
            points[i, 6:8] = self._sample_partial(-self.v_low)
            points[i, 8:10] = self._sample_partial(self.v_high)
            points[i, 10:12] = self._sample_partial(-self.v_low)
            points[i, 12:14] = self._sample_partial(self.v_high)
            points[i, 14:16] = self._sample_partial(-self.v_low)
            points[i, 16] = numpy.random.uniform(self.g_min, self.g_max)
        return points


class Boundaries_Model_16(pints.Boundaries):
    """
    A boundaries class that implements the maximum-rate boundaries used in
    Beattie et al.
    """

    # Limits for a-type parameters (untransformed)
    a_min = 1e-7
    a_max = 1e3

    # Limits for g-type parameters
    b_min = 1e-7
    b_max = 0.4

    # Limits for maximum rate coefficients
    km_min = 1.67e-5
    km_max = 1e3

    # Voltages used when determining maximum rate coefficients
    v_low = -120
    v_high = 60

    def __init__(self):
        self.g_min = 1e2 * 1e-3
        self.g_max = 5e5 * 1e-3

        # Univariate paramater bounds
        self._lower = numpy.array([
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.km_min, self.km_min,
            self.g_min,
        ])
        self._upper = numpy.array([
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.km_max, self.km_max,
            self.g_max,
        ])

    def n_parameters(self):
        return 15

    def check(self, parameters):
        # Check parameter boundaries
        if (numpy.any(parameters <= self._lower)
                or numpy.any(parameters >= self._upper)):
            return False

        # Check rate boundaries
        k1m = parameters[0] * numpy.exp(parameters[1] * self.v_high)
        if k1m <= self.km_min or k1m >= self.km_max:
            return False
        k2m = parameters[2] * numpy.exp(-parameters[3] * self.v_low)
        if k2m <= self.km_min or k2m >= self.km_max:
            return False
        k3m = parameters[4] * numpy.exp(parameters[5] * self.v_high)
        if k3m <= self.km_min or k3m >= self.km_max:
            return False
        k4m = parameters[6] * numpy.exp(-parameters[7] * self.v_low)
        if k4m <= self.km_min or k4m >= self.km_max:
            return False
        k5m = parameters[8] * numpy.exp(parameters[9] * self.v_high)
        if k5m <= self.km_min or k5m >= self.km_max:
            return False
        k6m = parameters[10] * numpy.exp(-parameters[11] * self.v_low)
        if k6m <= self.km_min or k6m >= self.km_max:
            return False

        for i in [12, 13]:
            if numpy.any(parameters[i] <= self.km_min) or numpy.any(parameters[i] >= self.km_max):
                return False

        # All tests passed!
        return True

    def _sample_partial(self, v):
        """Samples a pair of kinetic parameters"""
        for i in range(100):
            a = numpy.exp(numpy.random.uniform(
                numpy.log(self.a_min), numpy.log(self.a_max)))
            b = numpy.random.uniform(self.b_min, self.b_max)
            km = a * numpy.exp(b * v)
            if km > self.km_min and km < self.km_max:
                return a, b
        raise ValueError('Too many iterations')

    def sample(self, n=1):
        points = numpy.zeros((n, 15))
        for i in range(n):
            points[i, 0:2] = self._sample_partial(self.v_high)
            points[i, 2:4] = self._sample_partial(-self.v_low)
            points[i, 4:6] = self._sample_partial(self.v_high)
            points[i, 6:8] = self._sample_partial(-self.v_low)
            points[i, 8:10] = self._sample_partial(self.v_high)
            points[i, 10:12] = self._sample_partial(-self.v_low)
            points[i, 12:14] = numpy.random.uniform(self.g_min, self.g_max)
            points[i, 14] = numpy.random.uniform(self.g_min, self.g_max)
        return points


class Boundaries_Model_25(pints.Boundaries):
    """
    A boundaries class that implements the maximum-rate boundaries used in
    Beattie et al.
    """

    # Limits for a-type parameters (untransformed)
    a_min = 1e-7
    a_max = 1e3

    # Limits for g-type parameters
    b_min = 1e-7
    b_max = 0.4

    # Limits for maximum rate coefficients
    km_min = 1.67e-5
    km_max = 1e3

    # Voltages used when determining maximum rate coefficients
    v_low = -120
    v_high = 60

    def __init__(self):
        self.g_min = 1e2 * 1e-3
        self.g_max = 5e5 * 1e-3

        # Univariate paramater bounds
        self._lower = numpy.array([
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.g_min,
        ])
        self._upper = numpy.array([
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.g_max,
        ])

    def n_parameters(self):
        return 21

    def check(self, parameters):
        # Check parameter boundaries
        if (numpy.any(parameters <= self._lower)
                or numpy.any(parameters >= self._upper)):
            return False

        # Check rate boundaries
        k1m = parameters[0] * numpy.exp(parameters[1] * self.v_high)
        if k1m <= self.km_min or k1m >= self.km_max:
            return False
        k2m = parameters[2] * numpy.exp(-parameters[3] * self.v_low)
        if k2m <= self.km_min or k2m >= self.km_max:
            return False
        k3m = parameters[4] * numpy.exp(parameters[5] * self.v_high)
        if k3m <= self.km_min or k3m >= self.km_max:
            return False
        k4m = parameters[6] * numpy.exp(-parameters[7] * self.v_low)
        if k4m <= self.km_min or k4m >= self.km_max:
            return False
        k5m = parameters[8] * numpy.exp(parameters[9] * self.v_high)
        if k5m <= self.km_min or k5m >= self.km_max:
            return False
        k6m = parameters[10] * numpy.exp(-parameters[11] * self.v_low)
        if k6m <= self.km_min or k6m >= self.km_max:
            return False
        k7m = parameters[12] * numpy.exp(parameters[13] * self.v_high)
        if k7m <= self.km_min or k7m >= self.km_max:
            return False
        k8m = parameters[14] * numpy.exp(-parameters[15] * self.v_low)
        if k8m <= self.km_min or k8m >= self.km_max:
            return False
        k9m = parameters[16] * numpy.exp(parameters[17] * self.v_high)
        if k9m <= self.km_min or k9m >= self.km_max:
            return False
        k10m = parameters[18] * numpy.exp(-parameters[19] * self.v_low)
        if k10m <= self.km_min or k10m >= self.km_max:
            return False
        # All tests passed!
        return True

    def _sample_partial(self, v):
        """Samples a pair of kinetic parameters"""
        for i in range(100):
            a = numpy.exp(numpy.random.uniform(
                numpy.log(self.a_min), numpy.log(self.a_max)))
            b = numpy.random.uniform(self.b_min, self.b_max)
            km = a * numpy.exp(b * v)
            if km > self.km_min and km < self.km_max:
                return a, b
        raise ValueError('Too many iterations')

    def sample(self, n=1):
        points = numpy.zeros((n, 21))
        for i in range(n):
            points[i, 0:2] = self._sample_partial(self.v_high)
            points[i, 2:4] = self._sample_partial(-self.v_low)
            points[i, 4:6] = self._sample_partial(self.v_high)
            points[i, 6:8] = self._sample_partial(-self.v_low)
            points[i, 8:10] = self._sample_partial(self.v_high)
            points[i, 10:12] = self._sample_partial(-self.v_low)
            points[i, 12:14] = self._sample_partial(self.v_high)
            points[i, 14:16] = self._sample_partial(-self.v_low)
            points[i, 16:18] = self._sample_partial(self.v_high)
            points[i, 18:20] = self._sample_partial(-self.v_low)
            points[i, 20] = numpy.random.uniform(self.g_min, self.g_max)
        return points


# ======== transformation function implementation ========
def transformation_model_a():
    """
    Creates and returns a :class:`pints.Transformation` suitable for use
    with the model by Beattie et al.
    """
    return pints.ComposedTransformation(
        pints.LogTransformation(n_parameters=1),  # p1 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p2 (b-type)
        pints.LogTransformation(n_parameters=1),  # p3 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p4 (b-type)
        pints.LogTransformation(n_parameters=1),  # p5 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p6 (b-type)
        pints.LogTransformation(n_parameters=1),  # p7 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p8 (b-type)
        pints.IdentityTransformation(n_parameters=1),  # p9 (maximum conductance)
    )


def transformation_model_b():
    return pints.ComposedTransformation(
        pints.LogTransformation(n_parameters=1),  # p1 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p2 (b-type)
        pints.LogTransformation(n_parameters=1),  # p3 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p4 (b-type)
        pints.IdentityTransformation(n_parameters=1),  # p5 (b-type)
        pints.IdentityTransformation(n_parameters=1),  # p6 (b-type)
        pints.LogTransformation(n_parameters=1),  # p7 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p8 (b-type)
        pints.LogTransformation(n_parameters=1),  # p9 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p10 (b-type)
        pints.IdentityTransformation(n_parameters=1),  # p11 (maximum conductance)
    )


def transformation_model_15():
    """
    Creates and returns a :class:`pints.Transformation` suitable for use
    with the model by Beattie et al.
    """
    return pints.ComposedTransformation(
        pints.LogTransformation(n_parameters=1),  # p1 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p2 (b-type)
        pints.LogTransformation(n_parameters=1),  # p3 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p4 (b-type)
        pints.LogTransformation(n_parameters=1),  # p5 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p6 (b-type)
        pints.LogTransformation(n_parameters=1),  # p7 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p8 (b-type)
        pints.LogTransformation(n_parameters=1),  # p9 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p10 (b-type)
        pints.LogTransformation(n_parameters=1),  # p11 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p12 (b-type)
        pints.LogTransformation(n_parameters=1),  # p13 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p14 (b-type)
        pints.LogTransformation(n_parameters=1),  # p15 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p16 (b-type)
        pints.IdentityTransformation(n_parameters=1),  # p17 (maximum conductance)
    )


def transformation_model_16():
    """
    Creates and returns a :class:`pints.Transformation` suitable for use
    with the model by Beattie et al.
    """
    return pints.ComposedTransformation(
        pints.LogTransformation(n_parameters=1),  # p1 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p2 (b-type)
        pints.LogTransformation(n_parameters=1),  # p3 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p4 (b-type)
        pints.LogTransformation(n_parameters=1),  # p5 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p6 (b-type)
        pints.LogTransformation(n_parameters=1),  # p7 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p8 (b-type)
        pints.LogTransformation(n_parameters=1),  # p9 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p10 (b-type)
        pints.LogTransformation(n_parameters=1),  # p11 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p12 (b-type)
        pints.IdentityTransformation(n_parameters=1),  # p13 (b-type)
        pints.IdentityTransformation(n_parameters=1),  # p14 (b-type)
        pints.IdentityTransformation(n_parameters=1),  # p15 (maximum conductance)
    )

def transformation_model_25():
    """
    Creates and returns a :class:`pints.Transformation` suitable for use
    with the model by Beattie et al.
    """
    return pints.ComposedTransformation(
        pints.LogTransformation(n_parameters=1),  # p1 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p2 (b-type)
        pints.LogTransformation(n_parameters=1),  # p3 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p4 (b-type)
        pints.LogTransformation(n_parameters=1),  # p5 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p6 (b-type)
        pints.LogTransformation(n_parameters=1),  # p7 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p8 (b-type)
        pints.LogTransformation(n_parameters=1),  # p9 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p10 (b-type)
        pints.LogTransformation(n_parameters=1),  # p11 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p12 (b-type)
        pints.LogTransformation(n_parameters=1),  # p13 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p14 (b-type)
        pints.LogTransformation(n_parameters=1),  # p15 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p16 (b-type)
        pints.LogTransformation(n_parameters=1),  # p17 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p18 (b-type)
        pints.LogTransformation(n_parameters=1),  # p19 (a-type)
        pints.IdentityTransformation(n_parameters=1),  # p20 (b-type)
        pints.IdentityTransformation(n_parameters=1),  # p21 (maximum conductance)
    )