from typing import Any
import c3d  # c3d library
import numpy as np
from numpy import linalg as LA

from itertools import product
import matplotlib.pyplot as plot


class GRF_metrics:
    def __init__(self) -> None:
        pass

    ### TAKEN FROM metadata.py ###
    def print_metadata(self, reader):
        print("Header information:\n{}".format(reader.header))
        for key, g in sorted(reader.group_items()):
            print("")
            for key, p in sorted(g.param_items()):
                self.print_param(g, p)

    def print_param_value(self, name, value):
        print(name, "=", value)

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
        print("{0} = {1}".format(name, arr.flatten()[start:end]))

    def print_param(self, g, p):
        name = "{0.name}.{1.name}".format(g, p)
        print("{0}: {1.total_bytes}B {1.dimensions}".format(name, p))

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
                subscript = "".join(["[{0}]".format(x) for x in coordinate])
                self.print_param_array(name + subscript, p, offset)
                offset += p.dimensions[0]

    ### (END) TAKEN FROM metadata.py ###

    def import_data(self):
        is_failed = False

        with open("Sub1_r26_n3.c3d", "rb") as handle:
            reader = c3d.Reader(handle)

            #use python lists for looping as it is more lightweight (https://stackoverflow.com/questions/31250129/python-numpy-array-of-numpy-arrays)
            point_list = []
            analog_list = []
            for i, points, analog in reader.read_frames(): # this function has to be looped to get the data of each frame 
                #append points and analog values to class variable
                temp_points = points
                point_list.append(temp_points) 

                temp_analog = analog
                analog_list.append(temp_analog)

            self.points = np.asarray(point_list)
            self.analog = np.asanyarray(analog_list)

            #TODO Remove this debug print
            print(self.points.shape)
            print(self.analog.shape)
            #self.print_metadata(reader)



        # safe data in a class object
        return not is_failed

    def get_distance(points, GRF):
        # Calcualtes the vector of the distance between the GRF vector and the ankle joint along the sagittal plane
        vec = np.array([(1), (2), (3)])
        return vec

    def get_F_int(d, F_GRF):
        norm_F_GRF = LA.norm(F_GRF)
        F_int = (norm_F_GRF * d) / (
            5
        )  # https://www.facebook.com/kevinakirbydpm/photos/a.554861454611102/3747740051989877/?type=3 / 5cm is assumed acc. to paper
        return F_int

    def get_tibia_vector(self, frame_number):
        # specify frame number as 1-#last_frame
        fig = plot.figure()
        ax = plot.axes(projection='3d')

        for i in range(0,125):
            p_medial_malloulus = self.points[i][7][0:3] #take x,y,z (0:2) at frame (frame_number) for mallous medial (7)
            p_lateral_malloulus = self.points[i][6][0:3] #take x,y,z (0:2) at frame (frame_number) for mallous lateral (6)
            p_medial_femoral_epicondyle = self.points[i][5][0:3] #take x,y,z (0:2) at frame (frame_number) for epicondyle medial (5)
            p_lateral_femoral_epicondyle = self.points[i][4][0:3] #take x,y,z (0:2) at frame (frame_number) for epicondyle lateral (4)


            ax.scatter(p_medial_malloulus[0],p_medial_malloulus[1],p_medial_malloulus[2], marker=",", color="r")
            ax.scatter(p_lateral_malloulus[0],p_lateral_malloulus[1],p_lateral_malloulus[2], marker=",", color="b")
            ax.scatter(p_medial_femoral_epicondyle[0],p_medial_femoral_epicondyle[1],p_medial_femoral_epicondyle[2], marker="o", color="r")
            ax.scatter(p_lateral_femoral_epicondyle[0],p_lateral_femoral_epicondyle[1],p_lateral_femoral_epicondyle[2], marker="o", color="b")

        fig.savefig('temp.png', dpi=fig.dpi)

        print("p_medial_malloulus", p_medial_malloulus)
        print("p_lateral_malloulus", p_lateral_malloulus)
        print("p_medial_femoral_epicondyle", p_medial_femoral_epicondyle)
        print("p_lateral_femoral_epicondyle", p_lateral_femoral_epicondyle)


        p_ankle_center = 0.5*(self.calculate_vec_AtoB(p_medial_malloulus, p_lateral_malloulus)) + p_medial_malloulus
        p_knee_center = 0.5*(self.calculate_vec_AtoB(p_medial_femoral_epicondyle, p_lateral_femoral_epicondyle)) + p_medial_femoral_epicondyle

        tibia_vec = self.calculate_vec_AtoB(p_ankle_center, p_knee_center)
     
        return tibia_vec

    def calculate_vec_AtoB(point_A, point_B):
        # calculates the vector between 2 points A to B
        vec = np.zeros((3, 1))  # 3D vector
        vec[0] = point_B[0] - point_A[0]
        vec[1] = point_B[1] - point_A[1]
        vec[2] = point_B[2] - point_A[2]
        return vec

    def project_vector(x, y):
        # projects vector x onto vector
        vec = (
            y * np.dot(x, y) / np.dot(y, y)
        )  # https://en.wikipedia.org/wiki/Vector_projection
        return vec

    def real_tibial_load():
        # LOOP
        # Calulate real tibial load
        points, GRF_vec = get_frame()
        tibia_vec = calculate_vec_AtoB(point_ankle, point_knee)
        d = get_distance(points, GRF)

        F_int = get_F_int(d, GRF)
        F_ext = project_vector(GRF_vec, tibia_vec)

        F_tot = F_int + F_ext  # acc. to paper
        # ENDLOOP

        J = F_tot
        F_max = F_tot
        return J, F_max


# MAIN
GRF = GRF_metrics()
GRF_metrics.import_data(GRF)
print(GRF_metrics.get_tibia_vector(GRF,600))