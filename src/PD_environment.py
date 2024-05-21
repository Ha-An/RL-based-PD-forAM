from config import * 
import mesh_processor as mp  
import PD_interface

def extract_features(mesh):
    # (1) Volume
    part_volume = mesh.volume

    # (2) Size of the bounding box
    bounding_box = mesh.bounding_box.extents 

    # (3) Concavity
    concavity = mesh.convex_hull.volume - part_volume
    
    # (4) Support volume
    Utility = PD_interface.Utility()
    mesh = [mesh]
    mesh, sup_vol = Utility.orientation(Utility.create_obj(mesh)) 
  
    return part_volume, concavity, bounding_box, sup_vol, mesh 



def create_env():
    
    # Import the initial model
    processor = mp.MeshProcessor() 
    processor.load_mesh(MESH_PATH) 
    mesh = [processor.mesh]
    
    part_volume, concavity, bounding_box, sup_vol, mesh = extract_features(mesh[0])
    '''
    print("Initial model ==== ")
    print("Validation (Watertight): ", processor.mesh.is_watertight)
    print("Volume: ", part_volume)
    print("concavity: ", concavity)
    print("bounding_box: ", bounding_box)
    print("support_volume: ", support_volume)
    '''

    # processor.pyvista_visualize(processor.mesh)
    # Create a PD tree
    PD_tree = {1: {"Vol": part_volume, "BB-X": bounding_box[0], "BB-Y": bounding_box[1], "BB-Z": bounding_box[2],
                   "Conc": concavity, "SupVol": sup_vol[0], "Mesh":mesh[0] }}

    # Create a list of decomposed parts
    part_list = [1]
    return PD_tree, part_list


# def cap_current_state(PD_tree, decomposed_parts):
#     # Capture the current state of the PD environment
#     # BUILD ORIENTATION DETERMINATION

#     return state


def deter_build_orientation(trimesh_model):   
    Utility = PD_interface.Utility()
    build_orientation, sup_vol = Utility.orientation(
        Utility.create_obj(trimesh_model))

    # SET THE CURRENT BUILD ORIENTATION TO THE DEFAULT

    return build_orientation, sup_vol


def decompose_parts(ACTION, part_list, PD_tree):
    Utility=PD_interface.Utility()
    MeshProcessor=mp.MeshProcessor()
    Part=part_list[round(ACTION[0])] 

    # ACTION[0] : PART ID of the part to be decomposed
    # ACTION[1] : CUTTING PLANE COORDINATE & ANGLE 
    plain_normal = [ACTION[1],ACTION[2],ACTION[3]]
    
    meshes = MeshProcessor.trimesh_cut(PD_tree[Part]['Mesh'], plain_normal)

 
    if len(meshes) > 0 :
        i = 1
        for mesh in meshes:
            part_volume, concavity, bounding_box,sup_vol,mesh = extract_features(mesh)
            
            #print("Mesh{} Validation (Watertight): ".format(i), mesh.is_watertight)
            #print("Mesh{} Volume: ".format(i), part_volume)
            # mp.processor.pyvista_visualize(mesh)
            PartID = Part*10+i
            i += 1
            print(bounding_box)
            # Update the PD tree
            PD_tree[PartID] = {"Vol": part_volume, "BB-X":bounding_box[0],"BB-Y": bounding_box[1],"BB-Z": bounding_box[2],
                               "Conc": concavity, "SupVol":sup_vol[0],"Mesh":mesh[0]}

            # Update the list of decomposed parts
            part_list.append(PartID)
        part_list.remove(Part)
  
    total_supvol=0
    for part in part_list:
        total_supvol=total_supvol+PD_tree[part]["SupVol"]
        print(f"{part}:{PD_tree[part]["SupVol"]}")

    print("=================")
    print("Sum of SupVol:",total_supvol)
    print("=================")

    return PD_tree, part_list, total_supvol


def cal_reward(min_volume_of_surrport_struct):
    # Calculate the reward based on the current state
    
    return -min_volume_of_surrport_struct

'''
PD_tree, part_list=create_env()
PD_tree, part_list,reward=decompose_parts([1,0,0,0,-20,-20,100],part_list,PD_tree)
PD_tree, part_list,reward=decompose_parts([12,0,0,0,-200,-20,100],part_list,PD_tree)
'''