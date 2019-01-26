'''This example demonstrates how you can define custom objects.

The example defines several custom objects (2 cubes, a wall and a box). When
Cozmo sees the markers for those objects he will report that he observed an
object of that size and shape there.

You can adjust the markers, marker sizes, and object sizes to fit whatever
object you have and the exact size of the markers that you print out.
'''

import time

import cozmo
from cozmo.objects import CustomObject, CustomObjectMarkers, CustomObjectTypes


def handle_object_appeared(evt, **kw):
    # This will be called whenever an EvtObjectAppeared is dispatched -
    # whenever an Object comes into view.
    if isinstance(evt.obj, CustomObject):
        if evt.obj.object_id == 0:
            cozmo.robot.Robot.say_text("North Wall").wait_for_completed()
            print("Cozmo started seeing a %s" % str(evt.obj.object_id))
        elif evt.obj.object_id == 1:
            cozmo.robot.Robot.say_text("East Wall").wait_for_completed()
            print("Cozmo started seeing a %s" % str(evt.obj.object_id))
        elif evt.obj.object_id == 2:
            cozmo.robot.Robot.say_text("South Wall").wait_for_completed()
            print("Cozmo started seeing a %s" % str(evt.obj.object_id))
        elif evt.obj.object_id == 3:
            cozmo.robot.Robot.say_text("West Wall").wait_for_completed()
            print("Cozmo started seeing a %s" % str(evt.obj.object_id))


# def handle_object_disappeared(evt, **kw):
    # This will be called whenever an EvtObjectDisappeared is dispatched -
    # whenever an Object goes out of view.
    # if isinstance(evt.obj, CustomObject):
        # print("Cozmo stopped seeing a %s" % str(evt.obj.object_type))


def custom_objects(robot: cozmo.robot.Robot):
    # Add event handlers for whenever Cozmo sees a new object
    robot.add_event_handler(cozmo.objects.EvtObjectAppeared, handle_object_appeared)
    # robot.add_event_handler(cozmo.objects.EvtObjectDisappeared, handle_object_disappeared)

    # define a unique wall (150mm x 120mm (x10mm thick for all walls)
    # with a 50mm x 30mm Circles2 image on front and back
    north_wall = robot.world.define_custom_wall(CustomObjectTypes.CustomType02,
                                                CustomObjectMarkers.Circles2,
                                                150, 300,
                                                63, 63, True)

    # define a unique wall (150mm x 120mm (x10mm thick for all walls)
    # with a 50mm x 30mm Circles2 image on front and back
    east_wall = robot.world.define_custom_wall(CustomObjectTypes.CustomType02,
                                                CustomObjectMarkers.Circles3,
                                                150, 300,
                                                63, 63, True)

    # define a unique wall (150mm x 120mm (x10mm thick for all walls)
    # with a 50mm x 30mm Circles2 image on front and back
    south_wall = robot.world.define_custom_wall(CustomObjectTypes.CustomType02,
                                                CustomObjectMarkers.Circles4,
                                                150, 300,
                                                63, 63, True)

    # define a unique wall (150mm x 120mm (x10mm thick for all walls)
    # with a 50mm x 30mm Circles2 image on front and back
    west_wall = robot.world.define_custom_wall(CustomObjectTypes.CustomType02,
                                                CustomObjectMarkers.Circles5,
                                                150, 300,
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
