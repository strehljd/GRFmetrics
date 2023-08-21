from typing import Any
import c3d #c3d library
import numpy as np
from numpy import linalg as LA
from time import sleep

from itertools import product

class GRF_metrics:
    def __init__(self) -> None:
        pass
    ### TAKEN FROM metadata.py ###
    def print_metadata(self, reader):
        print('Header information:\n{}'.format(reader.header))
        for key, g in sorted(reader.group_items()):
            print('')
            for key, p in sorted(g.param_items()):
                self.print_param(g, p)

    def print_param_value(self, name, value):
        print(name, '=', value)

    def print_param_array(self, name, p, offset_in_elements):
        arr = []
        start = offset_in_elements
        end = offset_in_elements + p.dimensions[0]
        if p.bytes_per_element == 2:
            arr = p.int16_array
        elif p.bytes_per_element == 4:
            arr = p.float_array
        elif p.bytes_per_element == -1:
            return self.print_param_value(name, p.bytes[start:end])
        else:
            arr = p.int8_array
        print('{0} = {1}'.format(name, arr.flatten()[start:end]))


    def print_param(self,g, p):
        name = "{0.name}.{1.name}".format(g, p)
        print('{0}: {1.total_bytes}B {1.dimensions}'.format(name, p))

        if len(p.dimensions) == 0:
            val = None
            width = len(p.bytes)
            if width == 2:
                val = p.int16_value
            elif width == 4:
                val = p.float_value
            else:
                val = p.int8_value
            self.print_param_value(name, val)

        if len(p.dimensions) == 1 and p.dimensions[0] > 0:
            return self.print_param_array(name, p, 0)

        if len(p.dimensions) >= 2:
            offset = 0
            for coordinate in product(*map(range, reversed(p.dimensions[1:]))):
                subscript = ''.join(["[{0}]".format(x) for x in coordinate])
                self.print_param_array(name + subscript, p, offset)
                offset += p.dimensions[0]
    ### (END) TAKEN FROM metadata.py ###

    def import_data(self):
        is_failed = False   

        with open('Sub1_r26_n3.c3d', 'rb') as handle:
            reader = c3d.Reader(handle)
            self.print_metadata(reader)

        # safe data in a class object
        return  not is_failed

    def get_distance(points, GRF):
        # Calcualtes the vector of the distance between the GRF vector and the ankle joint along the sagittal plane
        vec = np.array([(1),(2),(3)])
        return vec
    
    def get_F_int(d,F_GRF):
        norm_F_GRF = LA.norm(F_GRF)
        F_int = (norm_F_GRF * d) / (5) # https://www.facebook.com/kevinakirbydpm/photos/a.554861454611102/3747740051989877/?type=3 / 5cm is assumed acc. to paper
        return F_int

    def calculate_vec_tibia():
        vec = np.array([(1),(2),(3)])
        return vec

    def project_vector(x,y):
        # projects vector x onto vector
        vec = y * np.dot(x, y) / np.dot(y, y) # https://en.wikipedia.org/wiki/Vector_projection
        return vec 

    if __name__ == "__main__":
        print("Shalom!")

        import_data()

        for i in range(1,10):
            # loop trough every frame

            # Calulate real tibial load
            points, GRF_vec = get_frame()
            tibia_vec = calculate_vec_tibia(points)
            d = get_distance(points, GRF)
            
            F_int = get_F_int(d,GRF)
            F_ext = project_vector(GRF_vec,tibia_vec)

            F_tot = F_int + F_ext #acc. to paper

