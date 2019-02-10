'''
The program lets cozmo recognize custom walls with the custom markers.
A 3D map is generated to show Cozmo's surroundings including the
walls and other recognizable objects. The user can control the Cozmo.
When it finds the cube, it will make it light blue.
Author: Tony Pan
Date: 01-28-2019
'''

import time

import cozmo
from cozmo.objects import CustomObject, CustomObjectMarkers, CustomObjectTypes


def handle_object_appeared(evt, **kw):
    # This will be called whenever an EvtObjectAppeared is dispatched -
    # whenever an Object comes into view.
    if int(evt.obj.object_id) == 5:
        # cozmo.robot.Robot.say_text(text="North Wall").wait_for_completed()
        print("Cozmo started seeing a %s" % str(evt.obj.object_id))
    elif int(evt.obj.object_id) == 6:
        # cozmo.robot.Robot.say_text(text="East Wall").wait_for_completed()
        print("Cozmo started seeing a %s" % str(evt.obj.object_id))
    elif int(evt.obj.object_id) == 7:
        # cozmo.robot.Robot.say_text(text="South Wall").wait_for_completed()
        print("Cozmo started seeing a %s" % str(evt.obj.object_id))
    elif int(evt.obj.object_id) == 8:
        # cozmo.robot.Robot.say_text(text="West Wall").wait_for_completed()
        print("Cozmo started seeing a %s" % str(evt.obj.object_id))
    elif evt.obj.object_type == cozmo.objects.LightCube:
        cozmo.lights.blue_light


def handle_object_disappeared(evt, **kw):
    # This will be called whenever an EvtObjectDisappeared is dispatched -
    # whenever an Object goes out of view.
    if isinstance(evt.obj, CustomObject):
        print("Cozmo stopped seeing a %s" % str(evt.obj.object_id))


def custom_objects(robot: cozmo.robot.Robot):
    # Add event handlers for whenever Cozmo sees a new object
    robot.add_event_handler(cozmo.objects.EvtObjectAppeared, handle_object_appeared)
    robot.add_event_handler(cozmo.objects.EvtObjectDisappeared, handle_object_disappeared)

    # define a unique wall (150mm x 120mm (x10mm thick for all walls)
    # with a 50mm x 30mm Circles2 image on front and back
    north_wall = robot.world.define_custom_wall(CustomObjectTypes.CustomType02,
                                                CustomObjectMarkers.Circles2,
                                                300, 150,
                                                63, 63, True)

    # define a unique wall (150mm x 120mm (x10mm thick for all walls)
    # with a 50mm x 30mm Circles2 image on front and back
    east_wall = robot.world.define_custom_wall(CustomObjectTypes.CustomType03,
                                                CustomObjectMarkers.Circles3,
                                                300, 150,
                                                63, 63, True)

    # define a unique wall (150mm x 120mm (x10mm thick for all walls)
    # with a 50mm x 30mm Circles2 image on front and back
    south_wall = robot.world.define_custom_wall(CustomObjectTypes.CustomType04,
                                                CustomObjectMarkers.Circles4,
                                                300, 150,
                                                63, 63, True)

    # define a unique wall (150mm x 120mm (x10mm thick for all walls)
    # with a 50mm x 30mm Circles2 image on front and back
    west_wall = robot.world.define_custom_wall(CustomObjectTypes.CustomType05,
                                                CustomObjectMarkers.Circles5,
                                                300, 150,
                                                63, 63, True)

    if ((north_wall is not None) and (east_wall is not None) and
            (south_wall is not None) and (west_wall is not None)):
        print("All objects defined successfully!")
    else:
        print("One or more object definitions failed!")
        return

    while True:
        time.sleep(0.1)


cozmo.run_program(custom_objects, use_3d_viewer=True, use_viewer=True)
