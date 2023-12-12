from lxml import etree
from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite import quadruped, walker, humanoid


def make_quadruped_walk(seed):
    xml_string = quadruped.make_model(
        floor_size=quadruped._DEFAULT_TIME_LIMIT * quadruped._WALK_SPEED
    )

    mjcf = etree.fromstring(xml_string)
    floor = mjcf.find("./worldbody/geom")
    floor.set("material", "")
    floor.set("rgba", "0 0 0 1")
    xml_string = etree.tostring(mjcf, pretty_print=True)

    physics = quadruped.Physics.from_xml_string(xml_string, common.ASSETS)
    task = quadruped.Move(desired_speed=quadruped._WALK_SPEED, random=seed)
    environment_kwargs = {}
    return control.Environment(
        physics,
        task,
        time_limit=quadruped._DEFAULT_TIME_LIMIT,
        control_timestep=quadruped._CONTROL_TIMESTEP,
        **environment_kwargs
    )


def make_walker_walk(seed):
    xml_string = common.read_model("walker.xml")

    mjcf = etree.fromstring(xml_string)
    floor = mjcf.find("./worldbody/geom")
    floor.set("material", "")
    floor.set("rgba", "0 0 0 1")
    xml_string = etree.tostring(mjcf, pretty_print=True)

    physics = walker.Physics.from_xml_string(xml_string, common.ASSETS)

    task = walker.PlanarWalker(move_speed=walker._WALK_SPEED, random=seed)
    environment_kwargs = {}
    return control.Environment(
        physics,
        task,
        time_limit=walker._DEFAULT_TIME_LIMIT,
        control_timestep=walker._CONTROL_TIMESTEP,
        **environment_kwargs
    )


def make_humanoid_walk(seed):
    xml_string = common.read_model("humanoid.xml")

    mjcf = etree.fromstring(xml_string)
    floor = mjcf.find("./worldbody/geom")
    floor.set("material", "")
    floor.set("rgba", "0 0 0 1")
    xml_string = etree.tostring(mjcf, pretty_print=True)

    physics = humanoid.Physics.from_xml_string(xml_string, common.ASSETS)

    task = humanoid.Humanoid(
        move_speed=humanoid._WALK_SPEED, pure_state=False, random=seed
    )
    environment_kwargs = {}
    return control.Environment(
        physics,
        task,
        time_limit=humanoid._DEFAULT_TIME_LIMIT,
        control_timestep=humanoid._CONTROL_TIMESTEP,
        **environment_kwargs
    )
