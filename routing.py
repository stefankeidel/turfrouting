import six
import sys

sys.modules['sklearn.externals.six'] = six

import requests
import json
import mlrose
from glom import glom
from turfpy.transformation import circle
from turfpy.measurement import bbox  # , distance
from geojson import Point, Feature


def main():
    # user_zones = get_zones_for_user(308884)
    point = Point((10.028393,53.622673))
    zone_data = get_zones_for_point(point)

    zones_to_visit = []

    # add original start point to the list
    zones_to_visit.append(
        {
            "latitude": glom(point, "coordinates.1"),
            "longitude": glom(point, "coordinates.0"),
            "name": "Starting Point",
        }
    )

    # kill zones we already own and can't be revisited. No need to go there now
    for z in zone_data:
        if glom(z, "currentOwner.id", default=None) != 308884:
            zones_to_visit.append(
                {
                    "latitude": z["latitude"],
                    "longitude": z["longitude"],
                    "name": z["name"],
                }
            )

    distances = get_distances(zones_to_visit)
    fitness_dists = mlrose.TravellingSales(distances=distances)

    problem_fit = mlrose.TSPOpt(
        length=len(zones_to_visit), fitness_fn=fitness_dists, maximize=False
    )

    # Solve problem using the genetic algorithm
    best_state, best_fitness = mlrose.genetic_alg(
        problem_fit, mutation_prob=0.3, max_attempts=1000
    )

    print("The best state found is: ", best_state)
    print("The fitness at the best state is: ", best_fitness)

    # rotate until 0 is at beginning, and add 0 again at the end to create a loop, so [2 0 1 3] => [0 1 3 2 0]
    start = 0
    for i, v in enumerate(best_state):
        if v == 0:
            start = i
    best_state_ordered = list(best_state[start:]) + list(best_state[:start]) + [0]

    zones_to_visit_ordered = []
    for s in best_state_ordered:
        zones_to_visit_ordered.append(zones_to_visit[s])

    _dump_cxb_link(point, zones_to_visit_ordered)
    # print("Reference:")
    # _dump_cxb_link(point, zones_to_visit)


def get_zones_for_user(user_id):
    """Call to turf API to get some info about the user."""
    url = "https://api.turfgame.com/unstable/users"
    payload = [{"id": user_id}]
    response = requests.post(url, data=json.dumps(payload))

    # let's assume we found one? otherwise this will probably except
    user_data = response.json()[0]

    # dump some data as assurance we know what we're doing? ;-)
    print(
        "Rank {} user '{}' currently holds {} zones.".format(
            glom(user_data, "rank"),
            glom(user_data, "name"),
            len(glom(user_data, "zones")),
        )
    )

    return glom(user_data, "zones")


def get_zones_for_point(point, radius=2, cap=30):
    """
    For a given starting point (for example, your house or current location)
    get all zones in a rectangular zone around said point.

    Point is to be specified as geojson.Point

    Optional "radius" kwarg can be used to specify radius of the circle in kilometers. Defaults to 2
    """
    # first we draw a circle, around the point, then determine a bbox around said circle
    box = bbox(circle(center=Feature(geometry=point), radius=radius))
    url = "https://api.turfgame.com/unstable/zones"
    payload = [
        {
            "northEast": {"latitude": box[3], "longitude": box[2],},
            "southWest": {"latitude": box[1], "longitude": box[0],},
        }
    ]

    response = requests.post(url, data=json.dumps(payload))
    zone_data = response.json()

    print(
        "We have found {} zones for the given point {},{}. Capping at {}".format(
            len(zone_data), glom(point, "coordinates.1"), glom(point, "coordinates.0"), cap
        )
    )

    return zone_data[:cap]


def get_distances(zone_data):
    """
    For all pairs in zone list, get some sort of distance and return
    a list of tuples (index1, index2, distance)
    """

    # all relevant index pairs?
    distances = []

    for i, _i in enumerate(zone_data):
        for j, _j in enumerate(zone_data):
            if j <= i:
                continue

            dist = distance(
                zone_data[i]["longitude"],
                zone_data[i]["latitude"],
                zone_data[j]["longitude"],
                zone_data[j]["latitude"],
            )

            # print(
            #     "Distance between {} and {} is {}".format(
            #         zone_data[i]["name"], zone_data[j]["name"], dist
            #     )
            # )

            distances.append((i, j, dist,))

            print(f"{zone_data[i]['name']} done")

    print("Pairwise distance calculation done!")

    return distances


def distance(p1lon, p1lat, p2lon, p2lat):
    url = "https://brouter.cxberlin.net/brouter?lonlats={},{}|{},{}&profile=trekking&alternativeidx=0&format=geojson".format(
        p1lon, p1lat, p2lon, p2lat
    )

    response = requests.get(url, timeout=20)
    geoj = response.json()

    return float(glom(geoj, "features.0.properties.track-length"))


def _dump_cxb_link(point, zone_data):
    """Dump link for route planning, mostly for debugging O_o"""

    print(
        "https://routing.cxberlin.net/#map=14/{}/{}/standard&lonlats={}&profile=trekking".format(
            glom(point, "coordinates.1"),
            glom(point, "coordinates.0"),
            ";".join(
                ["{},{}".format(x["longitude"], x["latitude"]) for x in zone_data]
            ),
        )
    )


if __name__ == "__main__":
    main()
