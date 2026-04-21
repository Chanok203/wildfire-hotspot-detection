from pydantic import BaseModel, Field


class CreateInstanceRequest(BaseModel):
    id: str = Field(..., example="drone_01", description="ID ของโดรน")
    input_url: str = Field(
        ..., example="rtsp://127.0.0.1:8554/drone_01", description="RTSP URL ต้นทาง"
    )
    output_url: str = Field(
        ...,
        example="rtsp://127.0.0.1:8554/AI/drone_01",
        description="RTSP URL ปลายทางที่ AI จะส่งไป",
    )
