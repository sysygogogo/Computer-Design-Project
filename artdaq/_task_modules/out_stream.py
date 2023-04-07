from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import ctypes
import numpy
from artdaq._lib import lib_importer
from artdaq._task_modules.write_functions import _write_raw
from artdaq.errors import check_for_error
from artdaq.constants import (RegenerationMode, ResolutionType)


class OutStream(object):
    """
    Exposes an output data stream on a DAQ task.

    The output data stream be used to control writing behavior and can be
    used in conjunction with writer classes to write samples to an
    ArtDAQ task.
    """
    def __init__(self, task):
        self._task = task
        self._handle = task._handle
        self._auto_start = False
        self._timeout = 10.0

        super(OutStream, self).__init__()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self._handle == other._handle and
                    self._auto_start == other._auto_start and
                    self._timeout == other._timeout)
        return False

    def __hash__(self):
        return hash((self._handle.value, self._auto_start, self._timeout))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return 'OutStream(task={0})'.format(self._task.name)

    @property
    def auto_start(self):
        """
        bool: Specifies if the "write" method automatically starts the
            stream's owning task if you did not explicitly start it
            with the DAQ Start Task method.
        """
        return self._auto_start

    @auto_start.setter
    def auto_start(self, val):
        self._auto_start = val

    @auto_start.deleter
    def auto_start(self):
        self._auto_start = False

    @property
    def timeout(self):
        """
        float: Specifies the amount of time in seconds to wait for
            the write method to write all samples. ArtDAQ performs a
            timeout check only if the write method must wait before it
            writes data. The write method returns an error if the time
            elapses. The default timeout is 10 seconds. If you set
            "timeout" to artdaq.WAIT_INFINITELY, the write method
            waits indefinitely. If you set timeout to 0, the write
            method tries once to write the submitted samples. If the
            write method could not write all the submitted samples, it
            returns an error and the number of samples successfully
            written in the number of samples written per channel
            output.
        """
        return self._timeout

    @timeout.setter
    def timeout(self, val):
        self._timeout = val

    @timeout.deleter
    def timeout(self):
        self._timeout = 10.0

    @property
    def regen_mode(self):
        """
        :class:`artdaq.constants.RegenerationMode`: Specifies whether
            to allow ArtDAQ to generate the same data multiple times.
        """
        val = ctypes.c_int()

        cfunc = lib_importer.windll. ArtDAQ_GetWriteRegenMode
        if cfunc.argtypes is None:
            with cfunc.arglock:
                if cfunc.argtypes is None:
                    cfunc.argtypes = [
                        lib_importer.task_handle, ctypes.POINTER(ctypes.c_int)]

        error_code = cfunc(
            self._handle, ctypes.byref(val))
        check_for_error(error_code)

        return RegenerationMode(val.value)

    @regen_mode.setter
    def regen_mode(self, val):
        val = val.value
        cfunc = lib_importer.windll.ArtDAQ_SetWriteRegenMode
        if cfunc.argtypes is None:
            with cfunc.arglock:
                if cfunc.argtypes is None:
                    cfunc.argtypes = [
                        lib_importer.task_handle, ctypes.c_int]

        error_code = cfunc(
            self._handle, val)
        check_for_error(error_code)

    def write(self, numpy_array):
        """
        Writes raw samples to the task or virtual channels you specify.

        The number of samples per channel to write is determined using the
        following equation:

        number_of_samples_per_channel = math.floor(
            numpy_array_size_in_bytes / (
                number_of_channels_to_write * raw_sample_size_in_bytes))

        Raw samples constitute the internal representation of samples in a
        device, read directly from the device or buffer without scaling or
        reordering. The native format of a device can be an 8-, 16-, or 32-bit
        integer, signed or unsigned.

        If you use a different integer size than the native format of the
        device, one integer can contain multiple samples or one sample can
        stretch across multiple integers. For example, if you use 32-bit
        integers, but the device uses 8-bit samples, one integer contains up to
        four samples. If you use 8-bit integers, but the device uses 16-bit
        samples, a sample might require two integers. This behavior varies from
        device to device. Refer to your device documentation for more
        information.

        ArtDAQ does not separate raw data into channels. It accepts data in
        an interleaved or non-interleaved 1D array, depending on the raw
        ordering of the device. Refer to your device documentation for more
        information.

        If the task uses on-demand timing, this method returns only after the
        device generates all samples. On-demand is the default timing type if
        you do not use the timing property on the task to configure a sample
        timing type. If the task uses any timing type other than on-demand,
        this method returns immediately and does not wait for the device to
        generate all samples. Your application must determine if the task is
        done to ensure that the device generated all samples.

        Use the "auto_start" property on the stream to specify if this method
        automatically starts the stream's owning task if you did not explicitly
        start it with the DAQ Start Task method.

        Use the "timeout" property on the stream to specify the amount of
        time in seconds to wait for the method to write all samples. ArtDAQ
        performs a timeout check only if the method must wait before it writes
        data. This method returns an error if the time elapses. The default
        timeout is 10 seconds. If you set timeout to artdaq.WAIT_INFINITELY,
        the method waits indefinitely. If you set timeout to 0, the method
        tries once to write the submitted samples. If the method could not
        write all the submitted samples, it returns an error and the number of
        samples successfully written.

        Args:
            numpy_array (numpy.ndarray): Specifies a 1D NumPy array that
                contains the raw samples to write to the task.
        Returns:
            int:

            Specifies the actual number of samples per channel successfully
            written to the buffer.
        """
        channels_to_write = self._task.channels
        number_of_channels = len(channels_to_write.channel_names)

        channels_to_write.ao_resolution_units = ResolutionType.BITS

        number_of_samples_per_channel, _ = divmod(
            numpy_array.nbytes, (
                number_of_channels * channels_to_write.ao_resolution / 8))

        return _write_raw(
            self._handle, number_of_samples_per_channel, self.auto_start,
            self.timeout, numpy_array)
