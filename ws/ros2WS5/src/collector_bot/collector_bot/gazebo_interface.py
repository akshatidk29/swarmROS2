"""Async wrappers for Gazebo spawn / delete services."""

from gazebo_msgs.srv import DeleteEntity, SpawnEntity

# Dropped objects are grayscale (already delivered, don't need camera detection).
# No collision tags so robots can drive through the basket area freely.
# Sizes match arena.world exactly: cube 0.15, sphere radius 0.10.

_CUBE_SDF = """<?xml version="1.0"?>
<sdf version="1.6">
<model name="{name}">
  <static>true</static>
  <link name="link">
    <visual name="v">
      <geometry><box><size>0.15 0.15 0.15</size></box></geometry>
      <material>
        <ambient>0.4 0.4 0.4 1</ambient>
        <diffuse>0.5 0.5 0.5 1</diffuse>
        <specular>0.3 0.3 0.3 1</specular>
      </material>
    </visual>
  </link>
</model>
</sdf>"""

_SPHERE_SDF = """<?xml version="1.0"?>
<sdf version="1.6">
<model name="{name}">
  <static>true</static>
  <link name="link">
    <visual name="v">
      <geometry><sphere><radius>0.10</radius></sphere></geometry>
      <material>
        <ambient>0.4 0.4 0.4 1</ambient>
        <diffuse>0.5 0.5 0.5 1</diffuse>
        <specular>0.3 0.3 0.3 1</specular>
      </material>
    </visual>
  </link>
</model>
</sdf>"""


class GazeboInterface:
    """Non-blocking Gazebo service calls (returns Futures)."""

    def __init__(self, node):
        self._del = node.create_client(DeleteEntity, '/delete_entity')
        self._spn = node.create_client(SpawnEntity,  '/spawn_entity')

    def delete_async(self, name):
        req = DeleteEntity.Request()
        req.name = name
        return self._del.call_async(req)

    def spawn_async(self, name, obj_type, x, y):
        sdf = (_CUBE_SDF if obj_type == 'cube' else _SPHERE_SDF).format(name=name)
        req = SpawnEntity.Request()
        req.name  = name
        req.xml   = sdf
        req.initial_pose.position.x = float(x)
        req.initial_pose.position.y = float(y)
        req.initial_pose.position.z = 0.15
        return self._spn.call_async(req)
