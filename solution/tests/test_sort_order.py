import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/team22_bringup'))
from team22_bringup.sort_order import measured_dimensions, prioritize_by_size


def item(name, dimensions, bounds, mode='lift'):
    return dict(kind=name, approach_mode=mode, target=dict(
        geometry={'dimensions_mm': dimensions}, bounds=bounds))


class SortOrderTests(unittest.TestCase):
    def scene(self):
        return [
            item('mouse', [112., 58., 40.], [[.42, -.23, 0.], [.49, -.10, .04]], 'edge_slide'),
            item('drive', [154., 80., 199.], [[.28, -.39, 0.], [.47, -.25, .20]]),
            item('cable', [160., 128., 42.], [[.30, .26, 0.], [.47, .46, .04]], 'edge_slide'),
        ]

    def test_size_is_volume_not_longest_side(self):
        objects = self.scene()[1:]
        self.assertEqual([x['kind'] for x in prioritize_by_size(objects, .22)], ['cable', 'drive'])

    def test_accessibility_overrides_smallest_size(self):
        result = prioritize_by_size(self.scene(), .22)
        self.assertEqual([x['kind'] for x in result], ['cable', 'drive', 'mouse'])
        self.assertEqual(result[-1]['order_priority']['access_after'], ['drive'])

    def test_mouse_is_not_forced_last_if_unobstructed(self):
        objects = self.scene()[::2]
        self.assertEqual([x['kind'] for x in prioritize_by_size(objects, .22)], ['mouse', 'cable'])

    def test_names_do_not_determine_priority(self):
        objects = self.scene()
        for obj, name in zip(objects, ('big_box', 'tiny_part', 'other')):
            obj['kind'] = name
        result = prioritize_by_size(objects, .22)
        self.assertEqual([x['kind'] for x in result], ['other', 'tiny_part', 'big_box'])

    def test_changed_sizes_change_the_order(self):
        objects = self.scene()[1:]
        objects[0]['target']['geometry']['dimensions_mm'] = [10., 10., 10.]
        self.assertEqual([x['kind'] for x in prioritize_by_size(objects, .22)], ['drive', 'cable'])

    def test_missing_edge_uses_size_without_inventing_obstacles(self):
        self.assertEqual([x['kind'] for x in prioritize_by_size(self.scene())], ['mouse', 'cable', 'drive'])

    def test_lift_grasp_waits_for_a_bottle_in_its_approach(self):
        objects = [
            item('small_jar', [41., 42., 135.], [[.37, -.48, 0.], [.42, -.435, .135]]),
            item('front_bottle', [67., 20., 194.], [[.26, -.445, 0.], [.34, -.386, .194]]),
            item('inboard_box', [154., 80., 199.], [[.28, -.39, 0.], [.47, -.25, .199]]),
        ]
        result = prioritize_by_size(objects, .22)
        self.assertEqual([obj['kind'] for obj in result], ['front_bottle', 'small_jar', 'inboard_box'])
        self.assertEqual(result[1]['order_priority']['access_after'], ['front_bottle'])

    def test_fallback_bounds_and_unknown_sizes(self):
        target = {'bounds': [[0., 0., 0.], [.1, .2, .3]]}
        self.assertEqual(measured_dimensions(target), (.1, .2, .3))
        objects = [{'kind': 'unknown', 'target': {}}, {'kind': 'measured', 'target': target}]
        self.assertEqual([x['kind'] for x in prioritize_by_size(objects)], ['measured', 'unknown'])

    def test_does_not_mutate_saved_queue(self):
        objects = self.scene()
        prioritize_by_size(objects, .22)
        self.assertEqual([x['kind'] for x in objects], ['mouse', 'drive', 'cable'])
        self.assertTrue(all('order_priority' not in obj for obj in objects))


if __name__ == '__main__':
    unittest.main()
