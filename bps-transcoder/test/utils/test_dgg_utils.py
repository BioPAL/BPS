import unittest

from bps.transcoder.utils.dgg_utils import dgg_search_tiles


class FusionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_dgg_search_tiles(self):
        # [lat_min, lat_max, lon_min, lon_max]
        latlon_coverage = [2.0, 3.0, 10.0, 11.0]

        (
            tiles_name_list,
            geotransform_dict,
            tile_extent_lonlat_dict,
            tiles_dict,
        ) = dgg_search_tiles(latlon_coverage, True)


if __name__ == "__main__":
    unittest.main()
