
# coding: utf-8

# author: Yang Tong  
# email: bitsoal@gmail.com

# ## philosophy: 
# ### I. Suppose the prim_a1 and prim_a2, prim_b1 and prim_b2 are the in-plane lattice vectors of two 2D structures, we try to search for those sub-lattice vector pairs (a1, a2, b1, b2) satisfying the conditions: |a1| = |b1| && |a2| = |b2| && angle < a1, a2 > == angle < b1, b2 > up to the acceptable max lattice strain and angle tolerance.
# ### II. then we first filter the (a1, a2, b1, b2) pairs w.r.t area. The smaller, the better.
# ### III. Choose the one as the ideal one whose lattice length ratio is closest to 1 --> convenient for KPOINTS sampling.
# ### IV. Create supercells with (a1, a2) and (b1, b2) being the lattice vectors and stack them.
# 
# ### Note that the angle < a1, a2 > is confined within arccos(-0.6) ~ arccos(0.6)

# In[1]:


import math, itertools, os, random, pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pymatgen as mg
from TwoD_Structures import TwoD_Structure
from pymatgen import Lattice

import pprint


# def solve_quadratic_equation(a, b, c):
#     delta = b**2 - 4*a*c
#     if delta < 0:
#         return []
#     elif delta == 0:
#         return [-b/(2*a)]
#     else:
#         sqr_delta = pow(delta, 0.5)
#         return [(-b-sqr_delta)/(2*a), (-b+sqr_delta)/(2*a)]
#     
# def get_valid_m2(inner_boundary, outer_boundary):
#     """
#     inner_boundary are a list of roots of the quadratic equation: a*x**2 + b*x + c = l
#     outer_boundary are a list of roots of the quadratic equation: a*x**2 + b*x +c = L
#     here l < L.
#     This function return a list of integer solutions of the inequality: l<= a*x**2 + b*x +c <=L
#     """
#     if len(outer_boundary) == 0:
#         return []
#     elif len(outer_boundary) == 1:
#         if abs(outer_boundary[0]-int(outer_boundary[0])) < 1.0e-5:
#             return [int(outer_boundary[0])]
#         else:
#             return []
#     else:
#         if len(inner_boundary) <= 1:
#             return [i for i in range(int(outer_boundary[0])-1, int(outer_boundary[1])+2)]
#         else:
#             m2_list = [i for i in range(int(outer_boundary[0])-1, int(inner_boundary[0])+1)]
#             m2_list += [i for i in range(int(inner_boundary[1]), int(outer_boundary[1])+2)]
#             return list(set(m2_list))
#         
# def group_list(old_list, no_of_ele_per_group=10):
#     pass
#     

# get_valid_m2(inner_boundary=[1.0, 3.2], outer_boundary=[-4.6, 8.5])
# int(-0.23)

# In[2]:


class Heterostructure():
    
    """
    This class provides method find_best_matched_sub_latt_pairs and build_best_matched_heterostructure for the heterostructure
    construction:
        Given the angle tolerance and lattice strain tolerance, find_best_matched_sub_latt_pairs can search for the best sub-lattices
            of the input structure_1 and structure_2, namely parent 1 and parent 2. The heterostructure constructed based on such sub-lattices
            has a relatively small in-plane area and its in-plane lattice constant ratio is close to 1.
        build_best_matched_heterostructure will call find_best_matched_sub_latt_pairs first and then construct the heterostructure. It will
            return a tuple (supercell of parent 1, supercell of parent 2, heterostructure) for users' further manipulations.
            Note that parent 1 is on top of parent 2 while the heterostructure is built.
    input arguments:
        - structure_filename_1, structure_filename_2 (str): the files storing structure 1 and structure 2. They could in any file format
                                                            that pymatgen supports.
        - interlayer_spacing (float or int): the interlayer spacing of the to-be-built heterostructure. Default: 3
        - vacuum_thickness (float or int): the thickness of the vacuum layer in angstroms. Default: 20
        - min_length (float or int): the minmum lattice constant of the to-be-built heterostructure. Default: 0.1
        - max_length (float or int): the maximum lattice constant of the to-be-built heterostructure. Default: 20
        - angle_range_consine (a list of two float or int numbers): the lower and upper bound of the angle between the in-plane lattice
                                    vectors. Here the cosine value is adopted. Default: [-0.6, 0.6]
        - angle_tolerance_in_deg (float): the maximum difference of the in-plane lattice angle between supercells of parent 1 and 2.
                                    Here the value is in degree. Default: 1.0e-5
        - relative_strain_tolerance (float): the maximum strain that can be applied to the supercells of parent 1 and parent 2 while the
                                    heterostructure is constructed. Here the value is in percentage.
                                    e.g.
                                        0.01%: 0.01% tensile strain
                                        -0.02%: 0.02% compressive strain
                                    default: 1.0e-2
        - strain_on_which (a list of four quasi-boolean): Decide to which lattice vector the strain is applied while the heterostructure is built.
                                    the first number: strain onto the lattice vector a of the supercell of parent 1
                                    the second number: strain onto the lattice vector b of the supercell of parent 1
                                    the third number: strain onto the lattice vector a of the supercell of parent 2
                                    the fourth number: strain onto the lattice vector b of the supercell of parent 2
                                the quasi-boolean be either 1 or 0
                                    1: apply strain to the corresponding lattice vector
                                    0: don't apply strain to the corresponding lattice vector
                                additional rules:
                                    if the numbers of lattice vector a/b for both parents are the same, the strain will be equally
                                        borne by the two lattice vector a/b. e.g. [1, 0, 1, 1], [0, 0, 0, 1]
                                Default: [1, 1, 1, 1]
    """
    
    def __init__(self, structure_filename_1, structure_filename_2, 
                 interlayer_spacing=3, vacuum_thickness = 20,
                 min_length=0.1, max_length=20, 
                 angle_range_cosine=[-0.6, 0.6], angle_tolerance_in_deg=1.0e-5, 
                 relative_strain_tolerance=1.0e-2, strain_on_which=[1, 1, 1, 1]):
        self.structure_filename_1 = structure_filename_1
        self.structure_1 = TwoD_Structure.from_file(structure_filename_1)
        self.structure_filename_2 = structure_filename_2
        self.structure_2 = TwoD_Structure.from_file(structure_filename_2)
        self.min_length = min_length
        self.max_length = max_length
        self.angle_range_cosine = angle_range_cosine
        self.angle_tolerance_in_deg = angle_tolerance_in_deg
        self.relative_strain_tolerance = relative_strain_tolerance
        self.strain_on_which = strain_on_which
        self.interlayer_spacing = interlayer_spacing
        self.vacuum_thickness = vacuum_thickness
        
    def build_best_matched_heterostructure(self):
        """
        given the angle tolerance and the lattice mismatch tolerance, search for the best and smallest heterostructure.
        return: (supercell of parent 1, supercell of parent 2, heterostructure)
        Note that parent 1 is on top of parent 2 while the heterostructure is built.
        """
        best_a1_a2_b1_b2_pair = self.find_best_matched_sub_latt_pairs()
        if best_a1_a2_b1_b2_pair == False:
            print("Fail to build a heterostructure satisfying the given tolerances in angle and strain.")
            print("\t\t\t{}".format(self.structure_filename_1))
            print("\t\t\t{}\n".format(self.structure_filename_2))
            return False
        
        structure_1, structure_2 = self.structure_1.copy(), self.structure_2.copy()
        
        #build supercells
        a1_m1, a1_m2 = best_a1_a2_b1_b2_pair["a1_m1"], best_a1_a2_b1_b2_pair["a1_m2"]
        a2_m1, a2_m2 = best_a1_a2_b1_b2_pair["a2_m1"], best_a1_a2_b1_b2_pair["a2_m2"]
        b1_m1, b1_m2 = best_a1_a2_b1_b2_pair["b1_m1"], best_a1_a2_b1_b2_pair["b1_m2"]
        b2_m1, b2_m2 = best_a1_a2_b1_b2_pair["b2_m1"], best_a1_a2_b1_b2_pair["b2_m2"]
        structure_1.make_supercell(scaling_matrix=[[a1_m1, a1_m2, 0],
                                                   [a2_m1, a2_m2, 0], 
                                                   [0, 0, 1]])
        structure_2.make_supercell(scaling_matrix=[[b1_m1, b1_m2, 0], 
                                                   [b2_m1, b2_m2, 0], 
                                                   [0, 0, 1]])
        
        #apply strain to supercells
        strain_on_a1, strain_on_a2 = best_a1_a2_b1_b2_pair["strain_on_a1"], best_a1_a2_b1_b2_pair["strain_on_a2"]
        strain_on_b1, strain_on_b2 = best_a1_a2_b1_b2_pair["strain_on_b1"], best_a1_a2_b1_b2_pair["strain_on_b2"]
        structure_1 = structure_1.get_strained_structure([strain_on_a1, strain_on_a2, 0])
        structure_2 = structure_2.get_strained_structure([strain_on_b1, strain_on_b2, 0])
        
        #unify supercells such that they have the same lattice constant c
        TwoD_Structure.unify_structure(alignment_type="bottom", structure=structure_1, ref_structure=structure_2)
        
        
        #the lattice angles should be the same
        new_structures = Heterostructure.unify_in_plane_lattice_angle(structure_1=structure_1, structure_2=structure_2, 
                                                                      tolerance=self.angle_tolerance_in_deg)
        if new_structures == False:
            return False
        structure_1, structure_2 = new_structures
        
            
        
        #estimate and change the lattice constant c of the heterostructure
        thickness = structure_1.get_structure_thickness(thickness_is_cartesian=True)
        thickness += structure_2.get_structure_thickness(thickness_is_cartesian=True)
        thickness += self.interlayer_spacing
        total_c = thickness + self.vacuum_thickness
        structure_1.change_latt_c(new_latt_c=total_c, new_latt_c_is_cartesian=True)
        structure_2.change_latt_c(new_latt_c=total_c, new_latt_c_is_cartesian=True)
       
        
        #shift structure_1 upwards such that it is self.interlayer_spacing above structure 1
        shift = min(structure_2.cart_coords[:, 2]) + thickness - max(structure_1.cart_coords[:, 2])
        structure_1.shift_along_c(shift_dis=shift, shift_dis_is_cartesian=True)
        
        
        #check if the lattices are the same
        max_diff = np.max(np.abs(structure_1.lattice.matrix-structure_2.lattice.matrix))
        if max_diff > 1.0e-5:
            raise Exception("Lattices are not the same for {} and {}".format(self.structure_filename_1, self.structure_filename_2))
        
        
        #build heterostructures
        hetero_str = structure_1 + structure_2
        
        #align parent structures w.r.t the heterostructure
        TwoD_Structure.unify_structure(alignment_type="bottom", structure=structure_2, ref_structure=hetero_str)
        TwoD_Structure.unify_structure(alignment_type="top", structure=structure_1, ref_structure=hetero_str)
        
        return (structure_1, structure_2, hetero_str)

        
        
    def find_best_matched_sub_latt_pairs(self):
        
        a1, a2 = self.structure_1.lattice.matrix[0, :], self.structure_1.lattice.matrix[1, :]
        b1, b2 = self.structure_2.lattice.matrix[0, :], self.structure_2.lattice.matrix[1, :]
        
        a_sub_latt_list=Heterostructure.generate_sub_latt_vectors(a1=a1, a2=a2, min_length=self.min_length, max_length=self.max_length)
        b_sub_latt_list=Heterostructure.generate_sub_latt_vectors(a1=b1, a2=b2, min_length=self.min_length, max_length=self.max_length)
        
        
        input_args = {"sub_latt_list_of_a":  a_sub_latt_list, 
                      "sub_latt_list_of_b":  b_sub_latt_list, 
                      "angle_range_cosine": self.angle_range_cosine, 
                      "angle_tolerance_in_deg": self.angle_tolerance_in_deg, 
                      "relative_strain_tolerance": self.relative_strain_tolerance, 
                      "strain_on_which": self.strain_on_which}
        matched_a1_a2_b1_b2_pair_list = []
        matched_pair_iter = Heterostructure.generate_matched_a1_a2_b1_b2_pairs(**input_args)
        for matched_pair in matched_pair_iter:
            matched_a1_a2_b1_b2_pair_list.append(matched_pair)
            if len(matched_a1_a2_b1_b2_pair_list) == 5:
                break
                      
        
        #pprint.pprint(matched_a1_a2_b1_b2_pair_list)
        no_of_pairs = len(matched_a1_a2_b1_b2_pair_list)
        if no_of_pairs == 0:
            return False
        
        return Heterostructure.sort_a1_a2_b1_b2_pair_by_area(matched_a1_a2_b1_b2_pair_list)[0]
        
        
      
    @classmethod
    def generate_sub_latt_vectors(cls, a1, a2, min_length=0.1, max_length=20):
        """
        Given the lattice vectors of the 2D primitive cell, generate all sub-lattice vectors whose length is smaller than max_length.
        input argument:
            - a1, a2 (type @ np.ndarray or list or tuple): the lattice vectors of the primitive cell.
                if a1 or a2 have dimensions larger than 2, 
                the first two elements of a1 or a2 will be taken as x and y components, discarding others.
            - min_length (type @float): the min length of the searched sub lattice vectors (Default: 0)
            - max_length (type @ float): the max length of the searched sub lattice vectors (Default: 20)
        output:
            a list of length N, where N is the number of valid sub-lattice vectors. Each entry consists of a sub-lattice vector 
            of np.ndarry, its norm and m1 and m2 (sub-lattice vector = m1*a1 + m2*a2)
        """
        a1, a2 = np.array(a1)[:2], np.array(a2)[:2]
    
        sub_latts = []
        max_m1 = int(max_length/np.linalg.norm(a1, ord=2))+1
        max_m2 = int(max_length/np.linalg.norm(a2, ord=2))+1
        for m1 in range(-max_m1, max_m1+1):
            for m2 in range(-max_m2, max_m2+1):
                new_latt = m1*a1 + m2*a2
                norm = np.linalg.norm(new_latt, ord=2)
                if min_length <= norm <= max_length:
                    sub_latts.append([new_latt, norm, m1, m2])
        return sorted(sub_latts, key=lambda sub_lat: sub_lat[1])
    
    @classmethod
    def generate_matched_a1_a2_b1_b2_pairs(cls, sub_latt_list_of_a, sub_latt_list_of_b,
                                           angle_range_cosine=[-0.6, 0.6], angle_tolerance_in_deg=1.0e-5, 
                                           relative_strain_tolerance=1.0e-2, strain_on_which=[1, 1, 1, 1]):
        """
        pick any two sub-lattice vectors from sub_latt_list_of_a and assign them to the new lattice vectors denoted as a1 and a2.
        pick any two sub-lattice vectors from sub_latt_list_of_b and assign them to the new lattice vectors denoted as b1 and b2.
        These a1, a2, b1 and b2 form a (a1, a2, b1, b2) pair
        
        input arguments:
            - sub_latt_list_of_a: a list of sub-lattice vectors generated by function "generate_sub_latt_vectors"
            - sub_latt_list_of_b: a list of sub-lattice vectors generated by function "generate_sub_latt_vectors"
            - angle_range_cosine: apply a constraint to the angle made by lattice vectors. Default [-0.6, 0.6] 
            - angle_tolerance_in_deg: the tolerance between angle <a1, a2> and angle <b1, b2>. Default: 1.0e-5
            - relative_strain_tolerance: the max relative mismatch between lattice vectors. Default: 0.01
            - strain_on_which: determine whether strains can be applied to [a1, a2, b1, b2].
                            1: strain is allowed; 0: strain is not allowed
                            if a1 and b1 are equal, strain is equally borne by both. The same goes with a2 and b2 
            
        We add constraints on the (a1, a2, b1, b2) pairs:
            - the rotation from a2 to a1 and from b2 to b1 should be anti-clockwise
            - the difference between angle <a1, a2> and angle <b1, b2> should be equal up to the angle_tolerance
            - the applied strained on a1, a2, b1 and b2 should be below the given tolerance
        
        output: an iterable. Each time return a dictionary which has keys:
                a1, a2, b1, b2: lattice vectors of dimension 2
                a1_norm, a2_norm, b1_norm, b2_norm: norms of a1, a2, b1 and b2
                a1_m1, a1_m2, a2_m1, a2_m2, b1_m1, b1_m2, b2_m1, b2_m2: e.g. a1 = a1_m1*primitive_a1 + a1_m2*primitive_a2
                a_angle, b_angle: angle <a1, a2>, angle <b1, b2>
                strain_on_a1, strain_on_a2, strain_on_b1, strain_on_b2 in percentage (%)
                a1_a2_area, b1_b2_area,
                a1_a2_ratio, b1_b2_ratio: norm(a1)/norm(a2), norm(b1)/norm(b2)
                strained_a1_a2_ratio: norm(strained a1) / norm(strained a2)
                strained_area: strained_norm1 * strained_norm2 * sin((<a1, a2>+<b1, b2>)/2)
    """
                
        #pprint.pprint(sub_latt_list_of_a1)
        #pprint.pprint(sub_latt_list_of_a2)
        #pprint.pprint(sub_latt_list_of_b1)
        #pprint.pprint(sub_latt_list_of_b2)
        #for a1, b1 in zip(sub_latt_list_of_a1, sub_latt_list_of_b1):
        #    for a2, b2 in zip(sub_latt_list_of_a2, sub_latt_list_of_b2):
        #        a1_a2_pairs.append([a1, a2])
        #        b1_b2_pairs.append([b1, b2])
        
        strain_on_which_a1_b1 = [strain_on_which[0], strain_on_which[2]]
        strain_on_which_a2_b2 = [strain_on_which[1], strain_on_which[3]]
        for a1_ind, a1 in enumerate(sub_latt_list_of_a):
            for b1_ind, b1 in enumerate(sub_latt_list_of_b):
                strained_norm1, strain_on_a1, strain_on_b1 = Heterostructure.cal_strain(length_1=a1[1], length_2=b1[1], 
                                                                                        strain_on_which=strain_on_which_a1_b1)
                if max([abs(strain_on_a1), abs(strain_on_b1)]) > relative_strain_tolerance:
                    continue
                
                for a2 in sub_latt_list_of_a[:a1_ind+1]:
                    for b2 in sub_latt_list_of_b[:b1_ind+1]:
                        strained_norm2, strain_on_a2, strain_on_b2 = Heterostructure.cal_strain(length_1=a2[1], length_2=b2[1], 
                                                                                                strain_on_which=strain_on_which_a2_b2)
                        if max([abs(strain_on_a2), abs(strain_on_b2)]) > relative_strain_tolerance:
                            continue
                            
                        cos_a_angle = Heterostructure.find_cos_rotaion_angle(a1[0], a2[0], find_sign=False)
                        cos_a_sign = Heterostructure.find_cos_rotaion_angle(a1[0], a2[0], find_sign=True)
                        if cos_a_sign < 0 or angle_range_cosine[0] > cos_a_angle or angle_range_cosine[1] < cos_a_angle:
                            continue
                            
                        cos_b_angle = Heterostructure.find_cos_rotaion_angle(b1[0], b2[0], find_sign=False)
                        cos_b_sign = Heterostructure.find_cos_rotaion_angle(b1[0], b2[0], find_sign=True)
                        if cos_b_sign < 0 or angle_range_cosine[0] > cos_b_angle or angle_range_cosine[1] < cos_b_angle:
                            continue
                
                        #Check if angle <a1, a2> is equal to angle <b1, b2> within the given tolerance.
                        if abs(np.arccos(cos_a_angle) - np.arccos(cos_b_angle))*180/np.pi > angle_tolerance_in_deg:
                            continue
                
        
                        dict_ = {"a1": a1[0], "a1_norm": a1[1], "a1_m1": a1[2], "a1_m2":a1[3],
                                 "a2": a2[0], "a2_norm": a2[1], "a2_m1": a2[2], "a2_m2": a2[3],
                                 "b1": b1[0], "b1_norm": b1[1], "b1_m1": b1[2], "b1_m2": b1[3],
                                 "b2": b2[0], "b2_norm": b2[1], "b2_m1": b2[2], "b2_m2": b2[3],
                                 "a_angle": np.arccos(cos_a_angle)*180/np.pi, "b_angle": np.arccos(cos_b_angle)*180/np.pi, 
                                 "strain_on_a1": strain_on_a1, "strain_on_a2": strain_on_a2, 
                                 "strain_on_b1": strain_on_b1, "strain_on_b2": strain_on_b2, 
                                 "a1_a2_area": np.abs(np.cross(a1[0], a2[0])), 
                                 "b1_b2_area": np.abs(np.cross(b1[0], b2[0])), 
                                 "strained_a1_a2_ratio": strained_norm1 / strained_norm2,
                                 "a1_a2_ratio": a1[1]/a2[1], "b1_b2_ratio": b1[1]/b2[1]}
                        strained_area = strained_norm1 * strained_norm2 * np.abs(np.sin(np.deg2rad(dict_["a_angle"]+dict_["b_angle"])/2))
                        dict_["strained_area"] = strained_area
                
                        yield dict_
                
    @classmethod
    def cal_strain(cls, length_1, length_2, strain_on_which):
        """
        calculate the strain in percentage that is applied to length_1 or length_2 such that they have the same length.
        input arguments:
            - length_1 (float)
            - length_2 (float)
            - strain_on_which (a list of 2 quasi-boolean ): decide to which the strain is applied.
                [0, 1]: apply the strain to length_2
                [1, 0]: apply the strain to length_1
                [0, 0] or [1, 1]: the strain is equally borne by both length_1 and length_2
        """
        if strain_on_which[0] == strain_on_which[1]:
            strained_length = (length_1+length_2)/2
        elif strain_on_which[0] == 1:
            strained_length = length_2
        else:
            strained_length = length_1
        strain_on_length_1 = (strained_length-length_1)/length_1
        strain_on_length_2 = (strained_length-length_2)/length_2
        return (strained_length, strain_on_length_1, strain_on_length_2)
                
    @classmethod
    def sort_a1_a2_b1_b2_pair_by_area(cls, a1_a2_b1_b2_pairs_list):
        """
        sort (a1, a2, b1, b2) pairs by area in an ascending order
        input argument:
            - a1_a2_b1_b2_pairs_list: a list of dictionaries which are outputs of generate_matched_a1_a2_b1_b2_pairs
        """
        return sorted(a1_a2_b1_b2_pairs_list, key=lambda a1_a2_b1_b2_pair: a1_a2_b1_b2_pair["strained_area"])
    
    @classmethod
    def sort_a1_a2_b1_b2_pair_by_length_ratio(cls, a1_a2_b1_b2_pairs_list):
        """
        sort (a1, a2, b1, b2) pairs by norm(a1)/norm(b1) in an ascending order
        input argument:
            - a1_a2_b1_b2_pairs_list: a list of dictionaries which are outputs of generate_matched_a1_a2_b1_b2_pairs
        """
        return sorted(a1_a2_b1_b2_pairs_list, key=lambda a1_a2_b1_b2_pair: abs(a1_a2_b1_b2_pair["strained_a1_a2_ratio"]-1))
        
    @classmethod
    def find_cos_rotaion_angle(cls, vector1, vector2, find_sign=False):
        """
        Return the angle in radian by which vector2 is rotated such that rotated vector2 points 
        in the same direction as vector1.
        input arguments:
            - vector1, vector2: np.ndarray of dimesion 2
            - find_sign: If True, will return the sign of the rotation: return 1 (-1) if the rotaion is anti-clockwise (clockwise)
                     If False, return cos(angle) in radians between vector1 and vector2
        """
        #print(vector1, vector2)
        if find_sign:
            sign = 1 if np.cross(vector1, vector2) < 0 else -1
            return sign
        else:
            cos_angle = np.inner(vector1, vector2)/(np.linalg.norm(vector1, ord=2)*np.linalg.norm(vector2, ord=2))
            return cos_angle
        
    @classmethod
    def unify_in_plane_lattice_angle(cls, structure_1, structure_2, tolerance=1.0e-5):
        """
        If the difference in the in-plane (a-b) lattice angle between structure_1 and structure_2 is smaller than tolerance,
        change the lattice of structure_1 and structure_2 such that they share the same in-plane lattice angle. Note that
        the difference is equally borne by both structures
        If the difference is larger than tolerance, return False.
        input arguments:
            - structure_1, structure_2: structures of type pymatgen.Structure or its sub-class
            - tolerance (float): tolerance in degree. Default: 1.0e-5
        output:
            reutrn the modified structure_1 and structure_2
        """
        if abs(structure_1.lattice.gamma - structure_2.lattice.gamma) > tolerance:
            return False
        
        average_gamma = (structure_1.lattice.gamma + structure_2.lattice.gamma)/2
        
        species_1 = structure_1.species
        species_2 = structure_2.species
        frac_coords_1 = structure_1.frac_coords
        frac_coords_2 = structure_2.frac_coords
        new_angles_1 = list(structure_1.lattice.angles[:2]) + [average_gamma]
        new_angles_2 = list(structure_2.lattice.angles[:2]) + [average_gamma]
        new_lattice_1 = Lattice.from_lengths_and_angles(abc=structure_1.lattice.abc, ang=new_angles_1)
        new_lattice_2 = Lattice.from_lengths_and_angles(abc=structure_2.lattice.abc, ang=new_angles_2)
        
        new_structure_1 = TwoD_Structure(lattice=new_lattice_1, species=species_1, coords=frac_coords_1, coords_are_cartesian=False)
        new_structure_2 = TwoD_Structure(lattice=new_lattice_2, species=species_2, coords=frac_coords_2, coords_are_cartesian=False)
        return new_structure_1, new_structure_2


# In[3]:


if __name__ == "__main__":
    
    WS2_dict = {'@class': 'Structure',
                '@module': 'pymatgen.core.structure',
                'charge': None,
                'lattice': {'a': 3.2,
                            'alpha': 90.0,
                            'b': 3.2,
                            'beta': 90.0,
                            'c': 20.0,
                            'gamma': 119.99999999999999,
                            'matrix': [[3.2, 0.0, 1.9594348786357652e-16],
                                       [-1.5999999999999996, 2.771281292110204, 1.9594348786357652e-16],
                                       [0.0, 0.0, 20.0]],
                            'volume': 177.36200269505306},
                'sites': [{'abc': [0.6667, 0.33336, 0.5],
                           'label': 'W',
                           'species': [{'element': 'W', 'occu': 1.0}],
                           'xyz': [1.600064, 0.9238343315378577, 10.0]},
                          {'abc': [0.33332, 0.66665, 0.42155],
                           'label': 'S',
                           'species': [{'element': 'S', 'occu': 1.0}],
                           'xyz': [-1.5999999999793957e-05, 1.8474746733852674, 8.431]},
                          {'abc': [0.33332, 0.66665, 0.57845],
                           'label': 'S',
                           'species': [{'element': 'S', 'occu': 1.0}],
                           'xyz': [-1.5999999999793957e-05, 1.8474746733852674, 11.569]}]}

    HfSe2_dict = {'@class': 'Structure',
             '@module': 'pymatgen.core.structure',
             'charge': None,
             'lattice': {'a': 3.77,
                         'alpha': 90.0,
                         'b': 3.77,
                         'beta': 90.0,
                         'c': 20.0,
                         'gamma': 120.00000000000001,
                         'matrix': [[3.77, 0.0, 2.3084592163927607e-16],
                                    [-1.8849999999999996, 3.264915772267334, 2.3084592163927607e-16],
                                    [0.0, 0.0, 20.0]],
                         'volume': 246.17464922895695},
             'sites': [{'abc': [0.0, 0.0, 0.5],
                        'label': 'Hf',
                        'species': [{'element': 'Hf', 'occu': 1.0}],
                        'xyz': [0.0, 0.0, 10.0]},
                       {'abc': [0.33334, 0.66667, 0.57865],
                        'label': 'Se',
                        'species': [{'element': 'Se', 'occu': 1.0}],
                        'xyz': [1.8850000000236733e-05, 2.1766213978974633, 11.573]},
                       {'abc': [0.66666, 0.33333, 0.42135],
                        'label': 'Se',
                        'species': [{'element': 'Se', 'occu': 1.0}],
                        'xyz': [1.8849811500000002, 1.0882943743698705, 8.427]}]}
    
    if not os.path.isdir("test"):
        os.mkdir("test")
    TwoD_Structure.from_dict(HfSe2_dict).to(format="cif", filename=os.path.join("test", "HfSe2.cif"))
    TwoD_Structure.from_dict(WS2_dict).to(format="cif", filename=os.path.join("test", "WS2.cif"))
    
    heterostructure = Heterostructure(structure_filename_1=os.path.join("test", "HfSe2.cif"), 
                                      structure_filename_2=os.path.join("test", "WS2.cif"))
    best_pair = heterostructure.find_best_matched_sub_latt_pairs()
    if best_pair:
        import pprint
        pprint.pprint(best_pair)
        parent_1, parent_2, hetero = heterostructure.build_best_matched_heterostructure()
        parent_1.to(format="cif", filename=os.path.join("test", "HfSe2_supercell.cif"))
        parent_2.to(format="cif", filename=os.path.join("test", "WS2_supercell.cif"))
        hetero.to(format="cif", filename=os.path.join("test", "heterostructure.cif"))
    else:
        print("No heterostructure satisfying the given tough tolerance.")


# hetero = Heterostructure(structure_filename_1="test_cif/WSe2-CONTCAR.cif", 
#                          structure_filename_2="test_cif/SnSe2-CONTCAR.cif", 
#                          relative_strain_tolerance=0.005, 
#                          max_length=20)

# hetero.find_best_matched_sub_latt_pairs()

# sorted([{"a": 1, "b":2}, {"a":0, "b": 3}], key=lambda dict_: dict_["b"])
