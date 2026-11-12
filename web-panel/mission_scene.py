"""Revalidate stationary tray identity against fresh depth after contents change."""
from team22_bringup.tray_tracking import tray_support


def observed_objects(packet):
    objects={o['kind']:o for o in packet['scene']['objects'] if o.get('status')=='observed' or
             (o['kind'].startswith('tray_') and o.get('status')=='geometry_confirmed' and
              o.get('current_depth_support',{}).get('confirmed') and o.get('age_sim_s',100)<.45)}
    for obj in packet['scene']['objects']:
        if obj['kind'].startswith('tray_') and obj['kind'] not in objects and obj.get('age_sim_s',100)<15:
            support=tray_support(obj,packet['obstacles'])
            if support['confirmed']:
                objects[obj['kind']]=dict(obj,status='geometry_confirmed',current_depth_support=support,
                    identity_source='previous_color_with_current_three_or_more_wall_depth',confirmed_stamp=packet['scene']['stamp'])
    return objects
