import c3d #c3d library
import numpy as np
from numpy import linalg as LA
from time import sleep

class GRF_metrics:

    def import_data():
        is_failed = False

        with open('Sub1_r26_n3.c3d', 'rb') as handle:
            reader = c3d.Reader(handle)
            for i, points, analog in reader.read_frames():
                print('frame {}: point {}, analog {}'.format(i, points.shape, analog.shape))
                sleep(0.05)

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

