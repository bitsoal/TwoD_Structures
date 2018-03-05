
# coding: utf-8

# In[1]:


import numpy as np
import spglib

from pymatgen import Structure, Lattice
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.structure_matcher import StructureMatcher

import os, copy


# In[2]:


class TwoD_Structure(Structure):
    
    def __init__(self, lattice, species, coords, charge=None, validate_proximity=False, 
                 to_unit_cell=False, coords_are_cartesian=False, site_properties=None, is_standard_2D=False):
        self.is_standard_2D = is_standard_2D
        self.input_arg_dict = {"lattice": lattice, "species": species, "coords": coords, 
                               "charge": charge, "validate_proximity": validate_proximity, 
                               "to_unit_cell":to_unit_cell, "coords_are_cartesian":coords_are_cartesian, 
                               "site_properties": site_properties}
        super(TwoD_Structure, self).__init__(**self.input_arg_dict)
        #self.map_into_cell()
        #self._make_all_atoms_together()
        
    def get_standard_structure(self):
        self.map_into_cell()
        self._make_all_atoms_together()
        self.center_structure()
        standard_structure = self._get_orthogonal_c_twoD_structure()
        standard_structure.map_into_cell()
        standard_structure.is_standard_2D = True
        return standard_structure
        
    
    def _get_orthogonal_c_twoD_structure(self):
        """
        This method can only be called after calling map_into_cell and _make_all_atoms_together.
        map_into_cell first maps all atoms into the cell.
        _make_all_atoms_together deals with the case where part of atoms at the cell top whereas the other at the cell bottom.
        _make_all_atoms_together is required beacause the atoms at the cell top will be separated from the atoms at the cell bottom
            after the lattice vectors's transformation
        Idea about the basis orthogonalization:
        Suppose the old and new basises are 3x3 matrixes, say old_basis and new_basis. Each row of them is a lattice vector.
        We define the transformation matrix as follows: new_basis = transformation_matrix * old_basis.
        If the a point in the old_basis has a fractional coordinate old_frac_coord, then its associated fractional coordinate
            in the new_basis is:
                                new_frac_coord = old_frac_coord * inverse(transformation_matrix)
        output:
            this method will return an orthogonalized structure whose lattice c is perpendicular to the a-b plane and is as long
                as the old lattice c.
        """
        angles = self.lattice.angles
        new_angles = [90, 90, angles[-1]]
        old_latt_vectors = self.lattice.matrix
        
        new_latt_c = np.cross(old_latt_vectors[0], old_latt_vectors[1])
        new_latt_c = new_latt_c * self.lattice.c / np.linalg.norm(new_latt_c)
        new_latt_vectors = np.copy(old_latt_vectors)
        new_latt_vectors[2] = new_latt_c
        
        inv_transformation_matrix = np.matmul(old_latt_vectors, np.linalg.inv(new_latt_vectors))
        
        new_frac_coords = np.zeros(self.frac_coords.shape)
        for atom_ind, old_frac_coord in enumerate(self.frac_coords):
            new_frac_coords[atom_ind] = np.dot(old_frac_coord, inv_transformation_matrix)
            
        lattice = Lattice.from_lengths_and_angles(abc=self.lattice.abc, ang=new_angles)
        
        input_argument_dict = copy.deepcopy(self.input_arg_dict)
        input_argument_dict["lattice"] = lattice
        input_argument_dict["coords"] = new_frac_coords
        input_argument_dict["coords_are_cartesian"] = False
        return TwoD_Structure(**input_argument_dict)
    
    
    def center_structure(self):
        """
        center the structure in the C direction.
        """
        shift_dis = self.lattice.c/2 - np.mean(self.cart_coords[:, 2])
        self.shift_along_c(shift_dis)
                
    
    def map_into_cell(self):
        """
        map all atoms into the cell.
        """
        species = self.species
        frac_coords = self.frac_coords
        for atom_ind, coord in enumerate(frac_coords):
            new_coord = [self._map_into_0_to_1_range(coord_) for coord_ in coord]
            self[atom_ind] = species[atom_ind], new_coord
            
    def _make_all_atoms_together(self, critical_frac_c_diff=0.5):
        """
        This method deals with the case where some atoms are at the top of the cell, while the other at the bottom.
        modification: map the atoms at the cell top downwards.
        The atoms at the cell top are those whose c coordinate is more than 
            critical_frac_c_diff away from the c coordinate of the lowest atom
        critical_frac_c_diff (float): default to 0.5
        """
        max_c, min_c = max(self.frac_coords[:, 2]), min(self.frac_coords[:, 2])
        
        species = self.species
        frac_coords = self.frac_coords
            
        for atom_ind, coord in enumerate(frac_coords):
            if coord[2]-min_c > critical_frac_c_diff:
                self[atom_ind] = species[atom_ind], list(coord[:2]) + [coord[2]-1]
                
            
    def shift_along_c(self, shift_dis):
        """
        Shift all atoms along the c direction by a distance shift_dis.
        Note that shift_dis is an absolute value in angstrom, not a fractional number.
        positve shift_dis <--> shift upwards,
        negative shift_dis <--> shift downwards.
        """
        frac_shift_dis = shift_dis/self.lattice.c
        species = self.species
        frac_coords = self.frac_coords
        
        for atom_ind, frac_coord in enumerate(frac_coords):
            self[atom_ind] = species[atom_ind], list(frac_coord[:2]) + [frac_coord[2]+frac_shift_dis]
        
        
            
    def _map_into_0_to_1_range(self, number):
        while number >= 1:
            number -= 1
            
        while number < 0:
            number += 1
            
        return number
    
    def turn_structure_upside_down(self, center=None):
        """
        Create a new TwoD_Structure object by turnning the old structure upside down along the C direction.
        optional input argument:
            - center (float): turn the old structure upside down with respect to center in the C direction.
                            Default: the C coordinate of the geometrical center of the old structure.
                            Note that it should be a fractional number relative to the lattice vector C.
        Note that this method is only available for standard 2D structures which can be obatined by calling method get_standard_structure
        output:
            a tuple - (new_structure, center)
        """
        assert self.is_standard_2D, "Error: this method is only available for standard 2D structure which can be obtained by calling method get_standard_structure"
        
        if center == None:
            center = np.mean(self.frac_coords[:, 2])
            
        species = self.species
        new_structure = self.copy()
        for atom_ind, frac_coord in enumerate(new_structure.frac_coords):
            new_coord = list(frac_coord[:2]) + [2*center-frac_coord[2]]
            new_structure[atom_ind] = species[atom_ind], new_coord
            
        new_structure.is_standard_2D = True
            
        return (new_structure, center)
    
    @classmethod
    def from_file(cls, filename, primitive=False, sort=False, merge_tol=0.0):
        structure = Structure.from_file(filename=filename, primitive=primitive, sort=sort, merge_tol=merge_tol)
        input_argument_dict = {"lattice": structure.lattice, 
                               "species": structure.species, 
                               "coords": structure.frac_coords, 
                               "coords_are_cartesian": False, 
                               "charge": structure.charge}
        
        return TwoD_Structure(**input_argument_dict)
    
    @classmethod
    def from_dict(cls, dictionary):
        """
        Reconsitite a TwoD_Structure object from a dictionary reprentation of TwoD_Structure or pymatgen.Structure
        input argument:
            dictionary (dict): Dictionary representation of structure
        """
        structure = Structure.from_dict(dictionary)
        input_argument_dict = {"lattice": structure.lattice, 
                               "species": structure.species, 
                               "coords": structure.frac_coords, 
                               "coords_are_cartesian": False, 
                               "charge": structure.charge}
        
        return TwoD_Structure(**input_argument_dict)
    
    def get_structure_thickness(self, return_cartesian_thickness=True):
        """
        return the thickness of the 2D structure
        optional input argument:
            - return_cartesian_thickness (bool): if True, return the absolute thickness in angstrom
                                            if False, return the fractional value of the thickness relative to the lattice c
                                            Default: True
        """
        if self.is_standard_2D:
            standard_str = self
        else:
            standard_str = self.get_standard_structure()
            
        cartesian_thickness = max(standard_str.cart_coords[:, 2])-min(standard_str.cart_coords[:, 2])
        if return_cartesian_thickness:
            return cartesian_thickness
        else:
            return cartesian_thickness / standard_str.lattice.c
        
    def get_vacuum_thickness(self, return_cartesian_thickness=True):
        """
        return the vacuum thickness.
        optional input argument:
            - return_cartesian_thickness (bool): if True, return the absolute thickness in angstrom
                                            if False, return the fractional value of the thickness relative to the lattice c
                                            Default: True
        Note that this method is only available if the 2D structure is a standard 2D structure by calling get_standard_structure method
        """
        assert self.is_standard_2D, "Error: the get_vacuum_thickness is only available for standard 2D structure obtained by get_standard_structure method"
        
        thickness = self.lattice.c - self.get_structure_thickness(return_cartesian_thickness=True)
        if return_cartesian_thickness:
            return thickness
        else:
            return thickness / self.lattice.c
        
    def change_vacuum_thickness(self, new_vacuum_thickness):
        """
        change the vacuum thickness.
        input argument:
            - new_vacuum_thickness (float): the new vacuum thickness in angstroms.
                The vacuum layer should be thicker than the slab. If thinner, raise Error.
        Note that this method is only available if the 2D structure is a standard 2D structure by calling get_standard_structure method
        output:
            return a new standard TwoD_Structure object with the assigned new_vacuum_thickness.
        """
        assert self.get_structure_thickness() < new_vacuum_thickness, "Error: the vacuum layer must         be thicker than the slab ({})".format(self.get_structure_thickness())
        
        assert self.is_standard_2D, "Error: the get_vacuum_thickness is only available for standard         2D structure obtained by get_standard_structure method"
               
        new_lattice_c = self.get_structure_thickness() + new_vacuum_thickness
        new_lattice_matrix = np.copy(self.lattice.matrix)
        new_lattice_matrix[2, 2] = new_lattice_c
        new_lattice = Lattice(matrix=new_lattice_matrix)
        
        input_argument_dict = copy.deepcopy(self.input_arg_dict)
        input_argument_dict.update({"lattice": new_lattice, 
                                    "species": self.species, 
                                    "coords": self.cart_coords, 
                                    "coords_are_cartesian": True})
        new_structure = TwoD_Structure(**input_argument_dict)
        return new_structure.get_standard_structure()
        
        
    
    def get_spglib_input(self):
        """
        Create and return a cell structure of spglib.
        the input argument of spglib, cell, has the format as follows:
        cell: Crystal structrue given in tuple.
            In the case given by a tuple, it has to follow the form below,
            (Lattice parameters in a 3x3 array (see the detail below),
             Fractional atomic positions in an Nx3 array,
             Integer numbers to distinguish species in a length N array,
             (optional) Collinear magnetic moments in a length N array),
            where N is the number of atoms.
            Lattice parameters are given in the form:
                [[a_x, a_y, a_z],
                 [b_x, b_y, b_z],
                 [c_x, c_y, c_z]]
        """
        if self.is_standard_2D:
            standard_str = self
        else:
            standard_str = self.get_standard_structure()
        
        lattice = standard_str.lattice.matrix
        
        positions = [list(coord) for coord in standard_str.frac_coords]
        
        #code atom species with integers
        species = standard_str.species
        unique_species = list(set(species))
        species_dict = {ele: i for ele, i in zip(unique_species, range(1, len(unique_species)+1))}
        numbers = [species_dict[ele] for ele in species]
        
        return (lattice, positions, numbers)
    
    def is_top_and_bottom_surfaces_equivalent(self, symprec=1e-5):
        """    Check if the top and bottom surfaces are equivalent.
        return True if equivalent; False otherwise.
        input argument:
            symprec (float): Default to 1e-5
        philosophy:
            step 1: Check if the input cell is planar. Return True if it is. Otherwise, proceed step 2
            step 2: Get all symmetry operations of the input cell using spglib.get_symmetry
            step 3: Pick up the highest atom. return True if there is a symmetry operation under 
                    which it is mapped to the lowest atom
        """
        if self.is_standard_2D:
            standard_str = self
        else:
            standard_str = self.get_standard_structure()
        
        #step 1
        if standard_str.get_structure_thickness() < symprec:
            return True
        
        #step 2
        cell = standard_str.get_spglib_input()
        symmetry_operations = spglib.get_symmetry(cell=cell, symprec=symprec)
        
        #step 3
        #find the species and fractional coordinate of the top atom.
        c_coordinates_list = [cell[1][i][2] for i in range(len(cell[1]))]
        top_atom_index = c_coordinates_list.index(min(c_coordinates_list))
        top_atom_coord = np.array(cell[1][top_atom_index])
        top_atom_species = cell[2][top_atom_index]
        
        half_frac_thickness = 0.5 * standard_str.get_structure_thickness(return_cartesian_thickness=False)
        for rotation, translation in zip(symmetry_operations["rotations"], symmetry_operations["translations"]):
            #in spglib, new_vector[3x1] = rotation[3x3] * vector[3x1] + translation[3x1]
            #np.dot(a, b), if a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b
            #so no need to reshape top_atom_coord from a 1-D array to a 3x1 column vector
            rotation, translation = np.array(rotation), np.array(translation).reshape(3)
            mapped_atom_coord = np.dot(rotation, top_atom_coord) + translation
            
            if abs(mapped_atom_coord[-1] - top_atom_coord[-1]) > half_frac_thickness:
                #print(rotation, translation)
                #print(top_atom_coord, mapped_atom_coord)
                return True
        return False
    
    
    def TwoD_Structure_2_Slab(self, tolerance=1.0e-10):
        """
        Return the associated pymatgen.core.surface.Slab object.
        Note that this method is only available if the 2D structure is a standard 2D structure 
            which is created by get_standard_structure method
        The pymatgen.core.surface.SlabGenerator is used to generate all (001)-terminated surfaces.
        Among those geneated pymatgen.core.surface.Slab objects, the right is picked up by comparing the distance_matrix,
        in-plane lattice length, lattice angles, atomic species with the parent TwoD_Structure object.
        Input argument:
            tolerance (float): the tolerance for the distance_matrix comparisons.
                            This is also the tolerance for determining the primitive unit cell
                            Default: 1.0e-10
        output:
            If the correct slab is found, return it; Otherwise, return False
        """
        assert self.is_standard_2D, "Error: method Two_Structure_2_Slab is only available if the 2D structure is a         standard 2D structure which is created by get_standard_structure method"
        
        slabgen_input_dict = {"initial_structure": self, 
                              "miller_index":(0, 0, 1), 
                              "min_slab_size": self.get_structure_thickness(), 
                              "min_vacuum_size": self.get_vacuum_thickness()}
        
        slab_list = SlabGenerator(**slabgen_input_dict).get_slabs(tol=tolerance)
        
        target_slab_list = []
        for slab in slab_list:
            
            #Check if the atom species are the same
            if slab.species != self.species:
                continue
            
            #See whether the slab has the correct distance_matrix
            distance_diff_max = np.max(np.abs(slab.distance_matrix - self.distance_matrix))
            if distance_diff_max > tolerance:
                continue
            
            #In principle, the correct slab and its parent TwoD_Structure structure should just differ by a constant \
            #shift along the C direction.
            coord_diff = np.abs(slab.cart_coords - self.cart_coords)
            in_plane_coord_diff_max = np.max(coord_diff[:, :2])
            out_of_plane_shift_diff_max = max(coord_diff[:, 2]) - min(coord_diff[:, 2])
            if in_plane_coord_diff_max > tolerance or out_of_plane_shift_diff_max > tolerance:
                continue
            
            
             
            #Check if the in-plane lattice vectors are as long as the parent structure
            if abs(slab.lattice.a - self.lattice.a) > tolerance or abs(slab.lattice.b - self.lattice.b) > tolerance:
                continue
                
            #Since the miller index is (0, 0, 1), so the lattice angles should not change.
            if sum([abs(ang0 - ang1) for ang0, ang1 in zip(slab.lattice.angles, self.lattice.angles)]) > tolerance:
                continue
                
            target_slab_list.append(slab)
            
        if len(target_slab_list) == 1:
            return target_slab_list[0]
        else:
            return False
                
            

