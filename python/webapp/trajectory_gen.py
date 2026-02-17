"""
Coordinate conversion utilities for FastTracker Web GUI.

Trajectory computation is handled by the C++ executable.
This module provides only lat/lon <-> meters conversion.
"""

import math


def latlon_to_meters(lat1, lon1, lat2, lon2):
    """Convert lat/lon to relative meters. lat1/lon1 is the reference point (origin).

    Uses cos(lat1) for longitude scaling to ensure exact round-trip with
    metersToLatLon(x, y, ref_lat=lat1, ref_lon=lon1) in C++.
    """
    R = 6371000  # Earth radius in meters
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)

    dlat = lat2_r - lat1_r
    dlon = math.radians(lon2) - math.radians(lon1)

    # Use lat1 (reference) for cosine factor — matches C++ metersToLatLon
    dx = R * dlon * math.cos(lat1_r)
    dy = R * dlat

    return dx, dy


def meters_to_latlon(x, y, ref_lat, ref_lon):
    """Convert meters back to lat/lon relative to a reference point."""
    R = 6371000
    ref_lat_r = math.radians(ref_lat)

    dlat = y / R
    dlon = x / (R * math.cos(ref_lat_r))

    return ref_lat + math.degrees(dlat), ref_lon + math.degrees(dlon)
