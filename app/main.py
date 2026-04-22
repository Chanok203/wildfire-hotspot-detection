from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import time

from app.services.vision import HotspotInstance
from app.utils.response import jsend_success, jsend_fail, jsend_error
from app.schemas.hotspot import CreateInstanceRequest
from ultralytics import YOLO

app = FastAPI(title="Wildfire hotspot Detection API", version="1.0.0")

active_instances = {}

model = YOLO("app/weights/best.pt")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return jsend_success({"message": "Wildfire hotspot Detection API is running"})


@app.get("/api/v1/hotspot-detection")
async def get_all_instances():
    data = []
    for drone_id, instance in active_instances.items():
        if not instance.is_running:
            continue
        data.append(
            {
                "id": drone_id,
                "start_time": instance.start_time,
                "remaining_sec": max(
                    0, int(instance.duration - (time.time() - instance.start_time))
                ),
            }
        )

    inactive = [k for k, v in active_instances.items() if not v.is_running]
    for k in inactive:
        del active_instances[k]

    return jsend_success({"instances": data})


@app.post("/api/v1/hotspot-detection")
async def create_instance(payload: CreateInstanceRequest):
    drone_id = payload.id
    input_url = payload.input_url
    output_url = payload.output_url

    if not drone_id or not input_url or not output_url:
        return jsend_fail(
            {"message": f"Missing required fields: id, input_url, output_url"}
        )

    if drone_id in active_instances and active_instances[drone_id].is_running:
        return jsend_fail(
            {"message": f"Instance {drone_id} is already running"}, status_code=409
        )

    instance = HotspotInstance(drone_id, model, input_url, output_url)
    instance.start()
    active_instances[drone_id] = instance
    return jsend_success({"id": drone_id, "status": "started"}, status_code=201)


@app.get("/api/v1/hotspot-detection/{drone_id}/snapshot")
async def get_snapshot(drone_id: str):
    instance = active_instances.get(drone_id)
    if not instance or not instance.is_running:
        return jsend_fail(
            {"message": "Instance not found or not running"}, status_code=404
        )

    img_bytes = instance.get_snapshot_image()
    if img_bytes is None:
        return jsend_error("Frame not captured yet", code=503)

    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")


@app.delete("/api/v1/hotspot-detection/{drone_id}")
async def delete_instance(drone_id: str):
    instance = active_instances.get(drone_id)
    if not instance:
        return jsend_fail({"message": "Instance not found"}, status_code=404)

    instance.stop()
    del active_instances[drone_id]

    return jsend_success({"message": f"Instance {drone_id} deleted"})


@app.delete("/api/v1/hotspot-detection")
async def delete_all_instances():
    count = len(active_instances)
    for drone_id in list(active_instances.keys()):
        active_instances[drone_id].stop()
        del active_instances[drone_id]

    return jsend_success({"deleted_count": count})


@app.get("/api/v1/hotspot-detection/{drone_id}")
async def get_instance(drone_id: str):
    instance = active_instances.get(drone_id)

    if not instance or not instance.is_running:
        return jsend_fail(
            {"message": "Instance not found or not running"}, status_code=404
        )

    return jsend_success(
        {
            "id": drone_id,
            "is_running": instance.is_running,
            "start_time": instance.start_time,
            "duration": instance.duration,
            "remaining_sec": max(
                0, int(instance.duration - (time.time() - instance.start_time))
            ),
        }
    )


@app.patch("/api/v1/hotspot-detection/{drone_id}")
async def extend_instance_time(drone_id: str):
    instance = active_instances.get(drone_id)

    if not instance or not instance.is_running:
        return jsend_fail({"message": "Instance not found"}, status_code=404)

    # เรียกใช้เมธอดที่เราเพิ่งเพิ่ม
    instance.extend_duration()

    return jsend_success(
        {
            "id": drone_id,
            "message": "Heartbeat received, timer reset",
            "remaining_sec": int(instance.duration),
        }
    )
