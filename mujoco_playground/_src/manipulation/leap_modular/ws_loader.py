

import numpy as np
import open3d as o3d

class WSLoader:
  def __init__(self,Finger="FF"):
    if Finger not in [ "FF", "TH"]:
      raise ValueError("Invalid Finger type. Must be one of FF, TH.")

    self.Finger = Finger

    self.ws_points = np.load(f"{Finger}.npy").reshape(-1, 3)

    self_offsets = {
      "MF": 0.0422842,
      "RF": 0.0422842*2
    }

  def get_goal(self,finger):
    if self.Finger == "Finger":
      if finger not in ["FF","MF","RF"]:
        raise ValueError( "Invalid finger type. Must be one of FF, MF, RF." )
      return self._get_goal_for_ff_mf_rf(finger)

    elif self.Finger == "TH":
      if finger not in ["TH"]:
        raise ValueError( "Invalid finger type. Must be TH." )
      return self._get_goal_for_th()

  def _get_goal_for_ff_mf_rf(self,finger):
    random_index = np.random.randint(0, array.shape[0])
    point =  self.ws_points( random_index )
    if finger in [ "MF", "RF" ]:
      point[0] += self.offsets[finger]

    return point

  def _get_goal_for_th(self):
    random_index = np.random.randint(0, array.shape[0])
    return self.ws_points( random_index )


if __name__ == "__main__":
  workspace_points = np.load("TH.npy").reshape(-1, 3)

  print( workspace_points.shape )
  print(workspace_points)


  # Create an Open3D PointCloud object
  point_cloud = o3d.geometry.PointCloud()

  # Assign the points to the PointCloud object
  point_cloud.points = o3d.utility.Vector3dVector(workspace_points)

  # Visualize the PointCloud
  o3d.visualization.draw_geometries([point_cloud])

