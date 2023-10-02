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

    def import_data(self, file):
        is_failed = False
        self.file_name = file

        with open(file, "rb") as handle:
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
            print("[import] Marker date shape: ", self.points.shape)
            print("[import] Analog date shape: ", self.analog.shape)
            self.print_metadata(reader)



        # safe data in a class object
        return not is_failed


    def get_tibia_vector(self, frame_number):

        frame_number = frame_number-1 # as defined in c3d fileformat
        p_medial_malloulus = np.array(self.points[frame_number][7][0:3]).reshape(3,1) #take x,y,z (0:2) at frame (frame_number) for mallous medial (7)
        p_lateral_malloulus = np.array(self.points[frame_number][6][0:3]).reshape(3,1) #take x,y,z (0:2) at frame (frame_number) for mallous lateral (6)
        p_medial_femoral_epicondyle = np.array(self.points[frame_number][5][0:3]).reshape(3,1) #take x,y,z (0:2) at frame (frame_number) for epicondyle medial (5)
        p_lateral_femoral_epicondyle = np.array(self.points[frame_number][4][0:3]).reshape(3,1) #take x,y,z (0:2) at frame (frame_number) for epicondyle lateral (4)

        p_ankle_center = (0.5*(self.calculate_vec_AtoB(p_medial_malloulus, p_lateral_malloulus)) + p_medial_malloulus).reshape(3,1)
        p_knee_center = (0.5*(self.calculate_vec_AtoB(p_medial_femoral_epicondyle, p_lateral_femoral_epicondyle)) + p_medial_femoral_epicondyle).reshape(3,1)

        tibia_vec = (self.calculate_vec_AtoB(p_ankle_center, p_knee_center)).reshape(3,1)

        ###
        # # TODO remove if running
        # # plot
        # fig = plot.figure()
        # ax = plot.axes(projection='3d')
        # ax.scatter(p_medial_malloulus[0],p_medial_malloulus[1],p_medial_malloulus[2], marker=",", color="r")
        # ax.scatter(p_lateral_malloulus[0],p_lateral_malloulus[1],p_lateral_malloulus[2], marker=",", color="b")
        # ax.scatter(p_medial_femoral_epicondyle[0],p_medial_femoral_epicondyle[1],p_medial_femoral_epicondyle[2], marker="o", color="r")
        # ax.scatter(p_lateral_femoral_epicondyle[0],p_lateral_femoral_epicondyle[1],p_lateral_femoral_epicondyle[2], marker="o", color="b")
        # ax.scatter(p_ankle_center[0],p_ankle_center[1],p_ankle_center[2], marker="<", color="g")
        # ax.scatter(p_knee_center[0],p_knee_center[1],p_knee_center[2], marker=">", color="g")

        # x = np.array([p_ankle_center[0], p_knee_center[0]])
        # y = np.array([p_ankle_center[1], p_knee_center[1]])
        # z = np.array([p_ankle_center[2], p_knee_center[2]])
        # ax.plot3D(x, y, z)

        # # save as pic
        # fig.savefig('temp.png', dpi=fig.dpi)
        # ###

        return tibia_vec

    def calculate_vec_AtoB(self, point_A, point_B):
        # calculates the vector between 2 points A to B
        vec = np.zeros((3, 1))  # 3D vector

        vec[0] = point_B[0] - point_A[0]
        vec[1] = point_B[1] - point_A[1]
        vec[2] = point_B[2] - point_A[2]
        return vec

    def project_vector(self, x, y):
        # projects vector x onto vector y
        vec = (
            y * np.dot(x, y) / np.dot(y, y)
        )  # https://en.wikipedia.org/wiki/Vector_projection
        return vec
    
    def get_F_int(self, GRF, frame_number):
        """Calculate F_int.
        Assuming that the Moment of F_int counteracts the (external) GRF moment (generated by the GRF along the lever d).
        Uses 5cm as lever for the F_int (muscle forces) as specified in the paper. 
        See https://www.facebook.com/kevinakirbydpm/photos/a.554861454611102/3747740051989877/?type=3    
        """
        
        # print("Get frame: ", frame_number)

        frame_number = frame_number-1 # as defined in c3d fileformat
        p_medial_malloulus = np.array(self.points[frame_number][7][0:3]).reshape(3,1) #take x,y,z (0:2) at frame (frame_number) for mallous medial (7)
        p_lateral_malloulus = np.array(self.points[frame_number][6][0:3]).reshape(3,1) #take x,y,z (0:2) at frame (frame_number) for mallous lateral (6)
        p_ankle_center = (0.5*(self.calculate_vec_AtoB(p_medial_malloulus, p_lateral_malloulus)) + p_medial_malloulus).reshape(3,1)

        trans = p_ankle_center

        # normal vector of the sagittal plane
        b = self.calculate_vec_AtoB(p_medial_malloulus, p_lateral_malloulus)
        b = b / np.linalg.norm(b) # normalize to unit-vector
        b = b.T.reshape(3,)

        a = np.array([1, 0, 0]) # unit vector of x-axis


        # rotate unit vector a onto unit vector b (calculation based on https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d)
        v = np.cross(a,b)
        c = np.dot(a,b)
        v_x = np.array([[0, -v[2], v[1] ], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

        rot = np.eye(3) + v_x + (1/(1+c))* (v_x@v_x)

        # build transformation matrix
        transform = np.concatenate((rot, trans), axis=1)
        hom_transform = np.concatenate((transform, np.array([0, 0, 0, 1]).reshape(1,4)), axis=0)

        F_GRF = hom_transform @ np.hstack((GRF.T.reshape(3,),np.array([1]))) # transform into sagittal plane coordiante frame

        # print("F_GRF: ", F_GRF, " GRF: ", GRF)
        # print("HomTrafo: ", hom_transform)

        r = self.get_distance(GRF)


        # # M_{GRF} = r x F_{GRF}
        M_GRF = r * F_GRF[0:3]
        # # M_{int} = 5[cm] x F_{int}
        # # M_{GRF} = M_{int}
        # # -> F_{int} = (r x F_{GRF}) / (5[cm]) (Here only the component along the sagittal plane should be taken into account)
        F_int = M_GRF / 5
        # F_int = np.cross(r,F_GRF)[0] / 5
        
        # TODO: Remove debug print
        # print("F_int: ", F_int)
        return F_int
    
    def get_distance(self, GRF):#points
        """Calculates the lever of the GRF.
        Which is the distance between the GRF vector and the ankle joint along the sagittal plane.
        """
        # TODO Calculate lever
        lever = np.array([(1), (1), (1)])

        return lever


    def real_tibial_load(self):
        number_frames = self.analog.shape[0]-1
        print("#frames: ", number_frames)
        F_int = np.zeros((number_frames,3)) # Pre-allocate matrix
        F_ext = np.zeros((number_frames,3)) # Pre-allocate matrix
        F_tot = np.zeros((number_frames,3)) # Pre-allocate matrix
        
        # LOOP
        for frame_number in range(0,number_frames):
            # Calulate real tibial load based on the projected GRF
            GRF_vec = np.zeros((3,1))
            GRF_vec[0] = np.average(self.analog[frame_number][0])
            GRF_vec[1] = np.average(self.analog[frame_number][1])
            GRF_vec[2] = np.average(self.analog[frame_number][2])

            tibia_vec = self.get_tibia_vector(frame_number)

            # print("[real_tibial] GRF projected: ", F_ext) # TODO remove this debug print

            # Calculate the internal reaction force based on the (internal) muscle forces
            F_int[frame_number,:] = self.get_F_int(GRF_vec, frame_number)
            F_ext[frame_number,:] = self.project_vector(GRF_vec.reshape(3,), tibia_vec.reshape(3,))

            F_tot[frame_number,:] = F_int[frame_number,:] + F_ext[frame_number,:]  # acc. to paper TODO: Add F_int
        # ENDLOOP

        # forces to euclidian norm 
        F_int_abs = np.zeros((number_frames,1)) # Pre-allocate matrix
        for i in range(1,number_frames):
            F_int_abs[i] = LA.norm(F_int[i,:])
            
        F_ext_abs = np.zeros((number_frames,1)) # Pre-allocate matrix
        for i in range(1,number_frames):
            F_ext_abs[i] = LA.norm(F_ext[i,:])
            
        # F_tot is equal to F_tibia??
        F_tot_abs = np.zeros((number_frames,1)) # Pre-allocate matrix
        for i in range(1,number_frames):
            F_tot_abs[i] = LA.norm(F_tot[i,:])
            
            
        # get maximum F_tibia
        index_max = np.argmax(F_tot_abs)
        

        # plot forces for debug purposes
        plot.plot(F_int_abs, linestyle='dashed', label='F_int')
        plot.plot(F_ext_abs, linestyle='dashed', label='F_ext')
        plot.plot(F_tot_abs, linestyle='dashed', label='F_tibia')
        plot.plot(index_max, F_tot_abs[index_max], 'ro', label='F_tibia_max')
        plot.legend(loc='upper left')
        plot.title(self.file_name)
        plot.show()
        
        
        # define return values
        F_max = F_tot_abs[index_max]
        J = np.trapz(F_tot_abs, dx=1)
        print("J = ", J) # TODO remove debug print
        return J, F_max



# MAIN
# init
GRF = GRF_metrics()

# file_names = ["Sub1_r26_p0.c3d", "Sub2_r26_p0.c3d", "Sub3_r26_p0.c3d" ,"Sub4_r26_p0.c3d" , "Sub5_r26_p0.c3d"]
file_names = ["Sub5_func_cal.c3d", "Sub5_static_cal.c3d"]

# do everything
for file_name in file_names:
    GRF_metrics.import_data(GRF, file_name)
    GRF_metrics.real_tibial_load(GRF)