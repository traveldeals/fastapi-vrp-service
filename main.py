import logging
import requests
import os
from typing import List, Tuple
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Logging Initialization
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mapping_agent")

app = FastAPI(
    title="Dispatch AI Mapping Agent",
    description="Production-grade core microservice for geocoding and VRP sequence optimization.",
    version="1.0.0"
)

# Enable CORS for easier testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_demo():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    html_path = os.path.join(current_dir, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


# -------------------------------------------------------------
# 1. PYDANTIC VALIDATION SCHEMAS
# -------------------------------------------------------------

class LocationInput(BaseModel):
    id: str = Field(..., description="Unique job or invoice identifier")
    name: str = Field(..., description="Friendly name of the stop or client")
    address: str = Field(..., description="Full text street address (Street, City, State, Zip)")

class OptimizeRequest(BaseModel):
    depot: LocationInput = Field(..., description="Starting and ending warehouse hub (Index 0)")
    stops: List[LocationInput] = Field(..., description="List of unsorted delivery locations")

    @field_validator('stops')
    @classmethod
    def validate_stop_count(cls, v):
        if len(v) == 0:
            raise ValueError("The delivery stop list cannot be empty.")
        if len(v) > 100:
            raise ValueError("Agent batch limits optimization to 100 stops per execution loop.")
        return v

class OptimizedStopResponse(BaseModel):
    sequence_position: int
    id: str
    name: str
    address: str
    coordinates: Tuple[float, float] = Field(..., description="(Latitude, Longitude)")

class OptimizeResponse(BaseModel):
    status: str
    total_stops: int
    optimized_manifest: List[OptimizedStopResponse]

# -------------------------------------------------------------
# 2. CORE LOGISTICS UTILITIES
# -------------------------------------------------------------

def geocode_address(address: str) -> Tuple[float, float]:
    """Converts plain text address into (Lat, Lon) coordinates.
    Tries Photon geocoding API first (friendly to Cloud Run IPs),
    then falls back to OpenStreetMap Nominatim.
    """
    # Try Photon (Komoot) API first
    photon_url = f"https://photon.komoot.io/api/?q={requests.utils.quote(address)}&limit=1"
    try:
        response = requests.get(photon_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data and "features" in data and len(data["features"]) > 0:
                coords = data["features"][0]["geometry"]["coordinates"]
                return float(coords[1]), float(coords[0]) # (Latitude, Longitude)
    except Exception as e:
        logger.warning(f"Photon geocoding failed, trying Nominatim fallback: {str(e)}")

    # Fallback to Nominatim
    headers = {"User-Agent": "DispatchAgent/1.0 (agent@yourdomain.com)"}
    nominatim_url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(address)}&format=json&limit=1"
    try:
        response = requests.get(nominatim_url, headers=headers, timeout=5)
        data = response.json()
        if data and len(data) > 0:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        logger.error(f"Nominatim geocoding fallback failed: {str(e)}")
        
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=f"Address could not be geocoded: {address}"
    )


def build_distance_matrix(coordinates: List[Tuple[float, float]]) -> List[List[int]]:
    """Generates street travel duration matrix (in seconds) via OSRM."""
    coord_string = ";".join([f"{lon},{lat}" for lat, lon in coordinates])
    url = f"http://router.project-osrm.org/table/v1/driving/{coord_string}?sources=all&destinations=all"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if response.status_code != 200 or "durations" not in data:
            raise HTTPException(status_code=502, detail="OSRM routing server failed matrix compilation.")
        # Google OR-Tools callbacks require integer constraints
        return [[int(cell) for cell in row] for row in data['durations']]
    except Exception as e:
        logger.error(f"Routing Matrix failure: {str(e)}")
        raise HTTPException(status_code=503, detail="OSRM routing service unavailable.")

# -------------------------------------------------------------
# 3. ROUTE CONTROLLER ENDPOINT
# -------------------------------------------------------------

@app.post(
    "/api/v1/route/optimize", 
    response_model=OptimizeResponse, 
    status_code=status.HTTP_200_OK,
    summary="Accepts unstructured addresses and outputs a mathematically optimized closed-loop sequence"
)
async def optimize_delivery_route(payload: OptimizeRequest):
    logger.info(f"Processing route batch request. Operational stops: {len(payload.stops)}")

    # Step A: Consolidate data structure. Depot hub is pinned to Index 0
    raw_locations = [payload.depot] + payload.stops
    geocoded_coords = [geocode_address(loc.address) for loc in raw_locations]

    # Step B: Fetch driving duration matrix
    distance_matrix = build_distance_matrix(geocoded_coords)
    num_locations = len(geocoded_coords)

    # Step C: Setup OR-Tools sequence manager (Nodes, Vehicles, Start_Node, End_Node)
    manager = pywrapcp.RoutingIndexManager(num_locations, 1, [0], [0])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Apply Optimization Policy
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="Optimization engine failed to settle a valid operational route."
        )

    # Step D: Construct response manifest matrix
    optimized_manifest = []
    index = routing.Start(0)
    sequence_counter = 0

    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        corresponding_location = raw_locations[node_index]
        
        optimized_manifest.append(
            OptimizedStopResponse(
                sequence_position=sequence_counter,
                id=corresponding_location.id,
                name=corresponding_location.name,
                address=corresponding_location.address,
                coordinates=geocoded_coords[node_index]
            )
        )
        sequence_counter += 1
        index = solution.Value(routing.NextVar(index))

    # Append structural closed-loop return to depot
    optimized_manifest.append(
        OptimizedStopResponse(
            sequence_position=sequence_counter,
            id=payload.depot.id,
            name=f"{payload.depot.name} (Return Depot)",
            address=payload.depot.address,
            coordinates=geocoded_coords[0]
        )
    )

    return OptimizeResponse(
        status="Success",
        total_stops=len(optimized_manifest) - 2,
        optimized_manifest=optimized_manifest
    )
