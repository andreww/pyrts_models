import numpy as np

DEFAULT_KNOTS = [1.00000,  0.96512,  0.92675,  0.88454,  0.83810,  0.78701,
                 0.73081,  0.66899,  0.60097,  0.52615,  0.44384,  0.35329,
                 0.25367,  0.14409,  0.02353, -0.010909, -0.25499, -0.41550,
                -0.59207, -0.78631, -1.00000]

class PyRTS:

    def __init__(self, filename=None, degree=None, knots=None,
                 inner_radius=3480.0, outer_radius=6346.691):
        """
        PyRTS objects represent seismic tomography of the mantle

        The serise of tomographic models such as "S20RTS",
        "S40RTS", and "SP12RTS" represent seismic velocities using
        the same basic parameterisation, with a spherical harmonic
        basis used for lateral velocity variation and cubic splines
        used to describe model variation in radius. The models also
        share a common file format to store model parameters. PyRTS
        objects allow this class of models to be reperesented and
        manipulated in python.

        Objects can be created 'empty' by passing in an integer
        spherical harmonic degree, or can be populated from an ".sph"
        file. All known models use the same radial parameterisation,
        but alternititve spline knots and depths for the CMB and Moho
        can be provided using additional optional arguments.

        Arguments:
        filename: a path-like object representing a ".sph" file used
              to initialise this model. Normally this is a string (optional)
        degree: integer representing the maximum spherical harmonic
              degree of this model (optional unless filename is None).
        knots: cubic spline knots in the range [1, -1] representing the
              radial parameterisation. If not provided a default used
              for all known models is used (optional list of floats)
        inner_radius: radius of the inside of the model in km (optional)
        outer_radius: radius of the outside of the model in km (optional)
        """
        # The locations of the knots are not stored in the data files
        # but the number is in the header. If no input is provided we
        # use the default set (corresponding to the published models).
        # Otherwise we just accept the provided list
        if knots is None:
            self.knots = DEFAULT_KNOTS
        else:
            self.knots = knots

        # If a degree has been passed we use this and allocate an
        # array for all the coefficents. If not, then the file reader
        # gets a chance to read in the degree and do the allocation
        # (otherwise it just checks that things match). 
        if degree is not None:
            self.degree = degree
            self._allocate_coefs()
        else:
            self.degree = None
            self.coefs = None

        # If it's provided we read in the coefficents from the file
        # if not allocated, this will allocate coefs too.
        if filename is not None:
            self.read_sph_file(filename)

        # At this point something must have allocated coefs. Check
        assert self.coefs is not None, \
            "You must provide a filename or degree"

        # Finally, store the top and bottom
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius


    def _allocate_coefs(self):
        """
        Create an empty array for the coefficents

        self.knots and self.degree must be set first
        """
        self.coefs = np.zeros((len(self.knots), 2, self.degree+1, 
                               self.degree+1))


    def read_sph_file(self, filename):
        """
        Read a sph file

        filename is a path-like object, if coefs has been allocated
        the size must match.
        """
        with open(filename, 'r') as fileobj:
            self.read_sph_stream(fileobj)


    def read_sph_stream(self, fileobj):
        """
        Read a stream representing sph file

        fileobj is a file-like object. This is normally called by
        read_sph_file.
        """
        header = None
        dataline = []
        ri = 0
        li = 0
        for line in fileobj:

            if header is None:
                header = line.split()
                degree_in_file = int(header[0])
                used_degrees_in_file = header[1].count('1')
                if self.degree is not None:
                    assert used_degrees_in_file - 1 == self.degree, \
                        "Wrong number of 1s in SH header"
                knots_in_file = int(header[2])
                used_knots_in_file = header[3].count('1')
                assert len(self.knots) == used_knots_in_file, \
                    "Wrong number of 1s in knots header"
                if self.coefs is None:
                    self.degree = used_degrees_in_file - 1
                    self._allocate_coefs()
                
            else:
                dataline.extend(line.split())
                # Lines can be a fixed number of coeffs or can
                # fill this harmonic
                if len(dataline) == (li * 2) + 1:
                    # We have all the data, process the line
                    assert ri <= len(self.knots), "Too many lines!"

                    mi = 0
                    for m, coef in enumerate(dataline):
                        if m == 0:
                            self.coefs[ri,0,li,mi] = float(coef)
                            mi = mi + 1
                        elif m%2 == 1:
                            # Odd number in list, real coef
                            self.coefs[ri,0,li,mi] = float(coef)
                            # don't increment mi!
                        else:
                            # even number in list, imag coef
                            self.coefs[ri,1,li,mi] = float(coef)
                            mi = mi + 1

                    li = li + 1
                    if li > self.degree:
                        li = 0
                        ri = ri + 1 
                    dataline = []

                assert len(dataline) < (li * 2) + 1, "Too much data"

