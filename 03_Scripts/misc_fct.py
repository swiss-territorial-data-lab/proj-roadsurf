import sys


def test_crs(crs1, crs2 = "EPSG:2056"):
    '''
    Take the crs of two dataframes and compare them. If they are not the same, stop the script.
    '''


    try:
        assert(crs1 == crs2), "CRS mismatch between the two files."
    except Exception as e:
        print(e)
        sys.exit(1)