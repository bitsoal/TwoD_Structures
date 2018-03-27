
# coding: utf-8

# In[ ]:


# author: Yang Tong
# email: bitsoal@gmail.com


# In[1]:


import numpy as np
import spglib

from pymatgen import Structure, Lattice
from pymatgen.core.surface import SlabGenerator

import os, copy


# In[1]:


class TwoD_Structure(Structure):
    
    def __init__(self, lattice, species, coords, charge=None, validate_proximity=False, 
                 to_unit_cell=False, coords_are_cartesian=False, site_properties=None):
        input_arg_dict = {"lattice": lattice, "species": species, "coords": coords, 
                          "charge": charge, "validate_proximity": validate_proximity, 
                          "to_unit_cell":to_unit_cell, "coords_are_cartesian":coords_are_cartesian, 
                          "site_properties": site_properties}
        super(TwoD_Structure, self).__init__(**input_arg_dict)
        
        self.standardize_structure() # standardize the input 2D structures
      
    def copy(self):
        return copy.deepcopy(self)
        
    def standardize_structure(self):
        """
        The method is called in __init__ to standardize the input 2D structures
        What will be done:
            1. orthogonalize the lattice vectors <--> the lattice vector c is normal to the a-b plane
            2. center the structure along the C direction.
            3. map all atoms into the cell
        """
        self.make_c_normal_to_ab_plane()
        self.center_structure()
        self.map_into_cell()
        
    
    def make_c_normal_to_ab_plane(self):
        """
        This method can only be called after calling map_into_cell and make_all_atoms_together.
        
        Idea about the basis orthogonalization:
        Suppose the old and new basises are 3x3 matrixes, say old_basis and new_basis. Each row of them is a lattice vector.
        We define the transformation matrix as follows: new_basis = transformation_matrix * old_basis.
        If the a point in the old_basis has a fractional coordinate old_frac_coord, then its associated fractional coordinate
            in the new_basis is:
                                new_frac_coord = old_frac_coord * inverse(transformation_matrix)
        What will be done:
            This method orthogonalizes the structure such that its lattice c is perpendicular to the a-b plane and is as long
                as the old lattice c. And set is_c_normal_to_ab_plane to True.
            Before orthogonalizing the lattice vectors, map_into_cell and make_all_atoms_together will be called first:
                - map_into_cell first maps all atoms into the cell.
                - make_all_atoms_together deals with the case where part of atoms at the cell top whereas the other at 
                the cell bottom. make_all_atoms_together is required beacause the atoms at the cell top will be separated 
                from the atoms at the cell bottom after the lattice vectors's transformation
                
        Note that this method only orthogonalizes the lattice vectors. If you want to orthogonalize the lattice vectors as well
            as center the structure, use method standardize_structure instead.
        """
        self.map_into_cell()
        self.make_all_atoms_together()
        
        species = self.species
        new_angles = [90, 90, self.lattice.gamma]
        old_latt_vectors = self.lattice.matrix
        
        new_latt_c = np.cross(old_latt_vectors[0], old_latt_vectors[1])
        new_latt_c = new_latt_c * self.lattice.c / np.linalg.norm(new_latt_c)
        new_latt_vectors = np.copy(old_latt_vectors)
        new_latt_vectors[2] = new_latt_c
        
        inv_transformation_matrix = np.matmul(old_latt_vectors, np.linalg.inv(new_latt_vectors))
        
        new_frac_coords = np.zeros(self.frac_coords.shape)
        for atom_ind, old_frac_coord in enumerate(self.frac_coords):
            new_frac_coords[atom_ind] = np.dot(old_frac_coord, inv_transformation_matrix)
            
        new_lattice = Lattice.from_lengths_and_angles(abc=self.lattice.abc, ang=new_angles)
        self.modify_lattice(new_lattice)
        
        for atom_ind in range(len(species)):
            self[atom_ind] = species[atom_ind], new_frac_coords[atom_ind]
            
        
    
    
    def center_structure(self):
        """
        center the structure in the C direction.

        """
        shift_dis = 0.5 - np.mean(self.frac_coords[:, 2])
        self.shift_along_c(shift_dis)
                
    
    def map_into_cell(self):
        """
        map all atoms into the cell and reset mapped to True
        """
        species = self.species
        frac_coords = self.frac_coords
        for atom_ind, coord in enumerate(frac_coords):
            new_coord = [self._map_into_0_to_1_range(coord_) for coord_ in coord]
            self[atom_ind] = species[atom_ind], new_coord
            
    def make_all_atoms_together(self, critical_frac_c_diff=0.5):
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
                
                
            
    def shift_along_c(self, shift_dis, shift_dis_is_cartesian=False):
        """
        Shift all atoms along the c direction by a distance shift_dis.
        
        input arguments:
            - shift_dis (float): positve shift_dis <--> shift upwards; negative shift_dis <--> shift downwards.
            - shift_dis_is_cartesian (bool): If False, shift_dis is fractional relative to lattice vector c;
                                            If True, shift_dis is in angstroms.
                                             Default: False
        """
        
        species = self.species
        frac_coords = self.frac_coords
        
        if shift_dis_is_cartesian:
            shift_dis /= self.lattice.c
        
        for atom_ind, frac_coord in enumerate(frac_coords):
            self[atom_ind] = species[atom_ind], list(frac_coord[:2]) + [frac_coord[2]+shift_dis]
            
            
    def _map_into_0_to_1_range(self, number):
        while number >= 1:
            number -= 1
            
        while number < 0:
            number += 1
            
        return number
    
    def turn_structure_upside_down(self):
        """
        Turn the structure upside down along the C direction with respect to the geometry center.
        """
        
        center = np.mean(self.frac_coords[:, 2])
            
        species = self.species
        frac_coords = np.copy(self.frac_coords)
        for atom_ind, frac_coord in enumerate(frac_coords):
            new_coord = list(frac_coord[:2]) + [2*center-frac_coord[2]]
            self[atom_ind] = species[atom_ind], new_coord
                
    
    @classmethod
    def from_file(cls, filename, primitive=False, sort=False, merge_tol=0.0):
        structure = Structure.from_file(filename=filename, primitive=primitive, sort=sort, merge_tol=merge_tol)
        input_argument_dict = {"lattice": structure.lattice, 
                               "species": structure.species, 
                               "coords": structure.frac_coords, 
                               "coords_are_cartesian": False, 
                               "charge": structure.charge}
        
        return TwoD_Structure(**input_argument_dict)
    
    def to(self, **kwargs):
        structure = Structure.from_dict(self.as_dict())
        structure.to(**kwargs)
    
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
    
    @classmethod
    def unify_structure(cls, structure, ref_structure, alignment_type="bottom", upside_down=False):
        """
        Take ref_structure as a reference and change structure from the three aspects below:
            1. modify the lattice vector c of structure to that of ref_structure.
            2. turn the structure upside down if upside_down is True;
            3. align the 2D structure to ref_structure such that they share the same lowest/highest c coordinate or their
                geometry center coincide with each other along the c direction.
        input arguments:
            - structure, ref_structure: TwoD_Structure objects
            - aligment_type (str): three options - "bottom", "top", "center". Default: "bottom"
            - upside_down (bool): whether to turn structure upside down. Default: False.
        """
        assert isinstance(structure, TwoD_Structure), "Error: input structure is not an instance of TwoD_Structure"
        assert isinstance(ref_structure, TwoD_Structure), "Error: input ref_structure is not an instance of TwoD_Structure"
        
        #step 1: compare the lattice constant c
        ref_latt_c, old_latt_c = ref_structure.lattice.c, structure.lattice.c
        if old_latt_c != ref_latt_c:
            structure.change_latt_c(new_latt_c=ref_latt_c, new_latt_c_is_cartesian=True)
            
        #step 2: whether to flip the structure
        if upside_down:
            structure.turn_structure_upside_down()
            
        #step 3: alignment
        TwoD_Structure.align_along_c(structure=structure, ref_structure=ref_structure, alignment_type=alignment_type)
        
        
    
    @classmethod
    def align_along_c(cls, structure, ref_structure, alignment_type="bottom"):
        """
        align structure to ref_structure such that they share the same lowest/highest c coordinate or their geometry centers coincide
        with each other along the c direction.
        input arguments:
            - structure, ref_structure: TwoD_Structure
            - aligment_type (str): three options - "bottom", "top", "center". Default: "bottom"
        """
        assert isinstance(structure, TwoD_Structure), "Error: input structure is not an instance of TwoD_Structure"
        assert isinstance(ref_structure, TwoD_Structure), "Error: input ref_structure is not an instance of TwoD_Structure"
        
        alignment_type = alignment_type.lower().strip()
        assert alignment_type in ["bottom", "top", "center"], "Error: alignment_type should be 'bottom', 'top' or 'center for align_two_structures.'"
        
        if alignment_type == "top":
            shift_dis = max(ref_structure.cart_coords[:, 2]) - max(structure.cart_coords[:, 2])
        elif alignment_type == "bottom":
            shift_dis = min(ref_structure.cart_coords[:, 2]) - min(structure.cart_coords[:, 2])
        else:
            shift_dis = np.mean(ref_structure.cart_coords[:, 2]) - np.mean(structure.cart_coords[:, 2])
            
        frac_shift_dis = shift_dis / structure.lattice.c
        
        species = structure.species
        frac_coords = np.copy(structure.frac_coords)
        for atom_ind, frac_coord in enumerate(frac_coords):
            structure[atom_ind] = species[atom_ind], list(frac_coord[:2]) + [frac_coord[2] + frac_shift_dis]
        
    
    def get_structure_thickness(self, thickness_is_cartesian=True):
        """
        return the thickness of the 2D structure.
        optional input argument:
            thickness_is_cartesian (bool): If True, return the thickness in angstroms
                                            If False, return the thickness in the fractional form relative to the lattice vector c
                                            Default: True
        """
        
        if thickness_is_cartesian:
            return max(self.cart_coords[:, 2]) - min(self.cart_coords[:, 2])
        else:
            return max(self.frac_coords[:, 2]) - min(self.frac_coords[:, 2])
        
    def get_vacuum_thickness(self, thickness_is_cartesian=True):
        """
        return the vacuum thickness.
        optional input argument:
            thickness_is_cartesian (bool): If True, return the thickness in angstroms
                                            If False, return the thickness in the fractional form relative to the lattice vector c
                                            Default: True
        """
        
        if thickness_is_cartesian:
            return self.lattice.c - self.get_structure_thickness(thickness_is_cartesian=True)
        else:
            return 1 - self.get_structure_thickness(thickness_is_cartesian=False)
        
        
        
    def change_vacuum_thickness(self, new_vacuum_thickness, thickness_is_cartesian=True):
        """
        change the vacuum thickness and at the end the standardize_structure will be called to standardize this structure.
        input argument:
            - new_vacuum_thickness (float): the new vacuum thickness in angstroms.
                The vacuum layer should be thicker than the slab. If thinner, raise Error.
            - thickness_is_cartesian (bool): If True, new_vacuum_thickness is in angstroms.
                                            If False, new_vacuum_thickness is fractional relative to the lattice vector c
        """
        assert self.get_structure_thickness(thickness_is_cartesian) < new_vacuum_thickness, "Error: the vacuum layer must         be thicker than the slab ({})".format(self.get_structure_thickness())
        
        if not thickness_is_cartesian:
            new_vacuum_thickness *= self.lattice.c
             
        new_lattice_c = self.get_structure_thickness() + new_vacuum_thickness
        new_lattice_matrix = np.copy(self.lattice.matrix)
        new_lattice_matrix[2, 2] = new_lattice_c
        new_lattice = Lattice(matrix=new_lattice_matrix)
        
        species = self.species
        cart_coords = np.copy(self.cart_coords)
        frac_coords = np.copy(self.frac_coords)
        
        self.modify_lattice(new_lattice)
        
        for atom_ind in range(len(species)):
            self[atom_ind] = species[atom_ind], list(frac_coords[atom_ind, :2]) + [cart_coords[atom_ind, 2]/self.lattice.c]
            
        self.standardize_structure()
        
    def change_latt_c(self, new_latt_c, new_latt_c_is_cartesian=True):
        """
        Change the length of the lattice vector c and at the end the standardize_structure will 
            be called to standardize this structure.
        input arguments:
            - new_latt_c (float): the new length of the lattice vector c
            - new_latt_c_is_cartesian (bool): if True, new_latt_c is in angstroms;
                                            if False, new_latt_c fractional relative to the old lattice vector c
                                            Default: True
        """
        structure_thickness = self.get_structure_thickness(thickness_is_cartesian=True)
        
        if new_latt_c_is_cartesian:
            new_vacuum_thickness = new_latt_c - structure_thickness
        else:
            new_vacuum_thickness = new_latt_c * self.lattice.c - structure_thickness
            
        self.change_vacuum_thickness(new_vacuum_thickness=new_vacuum_thickness, thickness_is_cartesian=True)
         
    
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
        
        lattice = self.lattice.matrix
        
        positions = [list(coord) for coord in self.frac_coords]
        
        #code atom species with integers
        species = self.species
        unique_species = list(set(species))
        species_dict = {ele: i for ele, i in zip(unique_species, range(1, len(unique_species)+1))}
        numbers = [species_dict[ele] for ele in species]
        
        return (lattice, positions, numbers)
    
    def are_top_and_bottom_surfaces_equivalent(self, symprec=1e-5):
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
        
        #step 1
        if self.get_structure_thickness(thickness_is_cartesian=True) < symprec:
            return True
        
        #step 2
        cell = self.get_spglib_input()
        symmetry_operations = spglib.get_symmetry(cell=cell, symprec=symprec)
        
        #step 3
        #find the species and fractional coordinate of the top atom.
        c_coordinates_list = [cell[1][i][2] for i in range(len(cell[1]))]
        top_atom_index = c_coordinates_list.index(max(c_coordinates_list))
        top_atom_coord = np.array(cell[1][top_atom_index])
        top_atom_species = cell[2][top_atom_index]
        
        half_frac_thickness = 0.5 * self.get_structure_thickness(thickness_is_cartesian=False)
        for rotation, translation in zip(symmetry_operations["rotations"], symmetry_operations["translations"]):
            #in spglib, new_vector[3x1] = rotation[3x3] * vector[3x1] + translation[3x1]
            #np.dot(a, b), if a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b
            #so no need to reshape top_atom_coord from a 1-D array to a 3x1 column vector
            rotation, translation = np.array(rotation), np.array(translation).reshape(3)
            mapped_atom_coord = np.dot(rotation, top_atom_coord) + translation
            
            if abs(mapped_atom_coord[-1] - top_atom_coord[-1]) > half_frac_thickness:
                return True
        return False
    
    
    
    def TwoD_Structure_2_Slab(self, tolerance=1.0e-5):
        """
        Return the associated pymatgen.core.surface.Slab object.
        The pymatgen.core.surface.SlabGenerator is used to generate all (001)-terminated surfaces.
        Among those geneated pymatgen.core.surface.Slab objects, the right is picked up by comparing the distance_matrix,
        in-plane lattice length, lattice angles, atomic species with the parent TwoD_Structure object.
        Input argument:
            tolerance (float): the tolerance for the distance_matrix comparisons.
                            This is also the tolerance for determining the primitive unit cell
                            Default: 1.0e-5
        output:
            If the correct slab is found, return it; Otherwise, return False
        """
        
        slab_thickness = self.get_structure_thickness(thickness_is_cartesian=True)
        if slab_thickness < 0.1:
            print("Note that the actual slab thickness is less than 1, this 2D material may be strictly planar. reset it to 0.1")
            slab_thickness = 0.1
        slabgen_input_dict = {"initial_structure": self, 
                              "miller_index":(0, 0, 1), 
                              "min_slab_size": slab_thickness, 
                              "min_vacuum_size": self.get_vacuum_thickness(thickness_is_cartesian=True)}
        
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
                
            

